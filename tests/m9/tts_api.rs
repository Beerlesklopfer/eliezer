//! `POST /v1/audio/speech` for speech synthesis models, and the voice endpoints (`/v1/audio/voices`).
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use eliezer_api::{App, managed_router};
use eliezer_backend::{supervisor::*, *};
use eliezer_core::{
    config::{Config, DevicePolicy},
    scheduler::{Clock, Instance, State},
};
use serde_json::{Value, json};
use std::{
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use tower::ServiceExt;

struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "eliezer-tts-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    /// The voice directory of the tests (it does not exist until the first upload).
    fn voices(&self) -> PathBuf {
        self.0.join("voices")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The fake backend with a record of the passages it was asked for and of the songs.
struct Recording {
    inner: FakeBackend,
    passages: Mutex<Vec<SpeechRequest>>,
    songs: Mutex<Vec<MusicRequest>>,
    /// The nth passage fails with this error.
    fail_at: Mutex<Option<(usize, BackendError)>>,
    /// The nth passage waits for a permit of this semaphore before it is read.
    gate: Mutex<Option<(usize, Arc<tokio::sync::Semaphore>)>>,
    /// From the nth passage on the answers have this sample rate.
    rate_at: Mutex<Option<(usize, u32)>>,
    /// Passages that were read to the end, and passages whose request was dropped while it ran.
    finished: AtomicU64,
    cancelled: Arc<AtomicU64>,
}
/// Counts a request that is dropped before it is finished.
struct CancelGuard(Arc<AtomicU64>, bool);
impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.1 {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
}
impl Backend for Recording {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        self.inner.start(spec)
    }
    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()> {
        self.inner.stop(running)
    }
    fn health<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, bool> {
        self.inner.health(running)
    }
    fn chat<'a>(
        &'a self,
        running: &'a Running,
        request: ChatRequest,
    ) -> BackendFuture<'a, tokio::sync::mpsc::Receiver<Result<Chunk, BackendError>>> {
        self.inner.chat(running, request)
    }
    fn music<'a>(
        &'a self,
        running: &'a Running,
        request: MusicRequest,
    ) -> BackendFuture<'a, Vec<u8>> {
        self.songs.lock().unwrap().push(request.clone());
        self.inner.music(running, request)
    }
    fn speak<'a>(
        &'a self,
        running: &'a Running,
        request: SpeechRequest,
    ) -> BackendFuture<'a, Vec<u8>> {
        let index = {
            let mut passages = self.passages.lock().unwrap();
            passages.push(request.clone());
            passages.len() - 1
        };
        if let Some((at, error)) = *self.fail_at.lock().unwrap()
            && at == index
        {
            return Box::pin(async move { Err(error) });
        }
        let gate = match &*self.gate.lock().unwrap() {
            Some((at, gate)) if *at == index => Some(gate.clone()),
            _ => None,
        };
        let rate = match *self.rate_at.lock().unwrap() {
            Some((at, rate)) if index >= at => Some(rate),
            _ => None,
        };
        let cancelled = self.cancelled.clone();
        let inner = self.inner.speak(running, request);
        Box::pin(async move {
            let mut guard = CancelGuard(cancelled, false);
            if let Some(gate) = gate {
                // A closed semaphore never happens in the tests; a permit is not given back.
                gate.acquire().await.unwrap().forget();
            }
            let wav = inner.await;
            guard.1 = true;
            self.finished.fetch_add(1, Ordering::SeqCst);
            match (wav, rate) {
                (Ok(wav), Some(rate)) => {
                    let mut pcm = eliezer_speech::pcm::read_wav(&wav).unwrap();
                    pcm.rate = rate;
                    Ok(eliezer_speech::pcm::write_wav(&pcm))
                }
                (wav, _) => wav,
            }
        })
    }
}

fn model(key: &str) -> Model {
    Model {
        instance: Instance {
            key: key.into(),
            state: State::Unloaded,
            bytes: 4_000_000_000,
            priority: 0,
            pinned: false,
            image: true,
            in_flight: 0,
            last_used_ms: 0,
        },
        default_spec: LoadSpec {
            model: key.into(),
            context: 32768,
            parallel: 1,
            bytes: 4_000_000_000,
            device: DevicePolicy::Auto,
        },
        max_context: 32768,
    }
}

/// The fake WAV of the backend: 800 samples at 8000 Hz (100 ms).
const PASSAGE_SAMPLES: usize = 800;

fn base(voices: &Path, settings: &str) -> String {
    format!(
        r#"
[server]
listen="127.0.0.1:1234"
[gpu]
vendor="none"
measure="none"
vram_budget_gb=24.0
reserve_gb=2.0
[backend.audiocpp]
program="/usr/bin/audiocpp_server"
port_range=[24700,24709]
voices_dir="{}"
max_passage_chars=40
pause_ms=100
paragraph_pause_ms=300
{settings}
"#,
        voices.display()
    )
}
const READER: &str = r#"
[[model]]
key="reader"
backend="audiocpp"
kind="tts"
source={file="/var/lib/eliezer/models/audio-cpp/Chatterbox-GGUF/chatterbox-q8_0.gguf"}
vram_gb=4.0
extra={language="de"}
"#;
const MUSIC: &str = r#"
[[model]]
key="composer"
backend="audiocpp"
kind="music"
source={file="/var/lib/eliezer/models/music"}
vram_gb=10.0
"#;

fn setup(text: &str, keys: Vec<String>) -> (Router, Arc<Recording>) {
    let config = Arc::new(Config::parse(text).unwrap());
    let backend = Arc::new(Recording {
        inner: FakeBackend::new(FakeConfig::default()),
        passages: Mutex::default(),
        songs: Mutex::default(),
        fail_at: Mutex::default(),
        gate: Mutex::default(),
        rate_at: Mutex::default(),
        finished: AtomicU64::new(0),
        cancelled: Arc::default(),
    });
    let names: Vec<String> = config.model.iter().map(|m| m.key.clone()).collect();
    let supervisor = Supervisor::spawn(
        names.iter().map(|k| model(k)).collect(),
        backend.clone(),
        Arc::new(TestClock),
        Settings {
            budget: 22_000_000_000,
            ram_budget: 0,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    );
    let app = App::new(
        config,
        supervisor,
        backend.clone(),
        Arc::new(eliezer_gpu::sources(
            eliezer_core::config::GpuVendor::None,
            eliezer_core::config::ProbeMethod::None,
            0,
            1,
            "/nonexistent".into(),
        )),
        keys,
    );
    (managed_router(app), backend)
}

struct Answer {
    status: StatusCode,
    kind: String,
    bytes: Vec<u8>,
}
impl Answer {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.bytes).unwrap_or(Value::Null)
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}
async fn send(app: &Router, request: Request<Body>) -> Answer {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let kind = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = to_bytes(response.into_body(), 64 << 20).await.unwrap();
    Answer {
        status,
        kind,
        bytes: bytes.to_vec(),
    }
}
async fn speak(app: &Router, body: Value) -> Answer {
    send(
        app,
        Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}
async fn get(app: &Router, uri: &str) -> Answer {
    send(
        app,
        Request::builder().uri(uri).body(Body::empty()).unwrap(),
    )
    .await
}
async fn delete(app: &Router, uri: &str) -> Answer {
    send(
        app,
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .body(Body::empty())
            .unwrap(),
    )
    .await
}

/// A multipart body: text fields and one file.
fn form(fields: &[(&str, &str)], file: Option<&[u8]>) -> (String, Vec<u8>) {
    let boundary = "----eliezer-test-boundary";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    if let Some(file) = file {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(file);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}
async fn upload(app: &Router, fields: &[(&str, &str)], file: Option<&[u8]>) -> Answer {
    upload_with(app, fields, file, None).await
}
async fn upload_with(
    app: &Router,
    fields: &[(&str, &str)],
    file: Option<&[u8]>,
    key: Option<&str>,
) -> Answer {
    let (kind, body) = form(fields, file);
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/audio/voices")
        .header("content-type", kind);
    if let Some(key) = key {
        request = request.header("authorization", format!("Bearer {key}"));
    }
    send(app, request.body(Body::from(body)).unwrap()).await
}

/// A recording of a tone: `seconds` long at 44.1 kHz, as a WAV file.
fn sample(seconds: f64) -> Vec<u8> {
    let rate = 44_100;
    let tone: Vec<i16> = (0..(seconds * f64::from(rate)) as usize)
        .map(|n| {
            ((n as f64 * 220.0 * std::f64::consts::TAU / f64::from(rate)).sin() * 9000.0) as i16
        })
        .collect();
    eliezer_speech::wav_at(&tone, rate)
}
fn mode(path: &Path) -> u32 {
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

// ---- reading aloud ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_text_is_read_in_passages_and_joined_with_the_pauses_of_the_configuration() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    // three sentences of 30 characters: one passage each (limit 40), then a new paragraph
    let text = "Der erste Satz ist hier drin. Der zweite Satz ist hier drin. Der dritte Satz ist hier drin.\n\nAbsatz zwei, nur ein kurzer Satz.";
    let answer = speak(&app, json!({"model": "reader", "input": text})).await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.text());
    assert_eq!(answer.kind, "audio/wav");
    let passages = backend.passages.lock().unwrap().clone();
    let sent: Vec<&str> = passages.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(
        sent,
        [
            "Der erste Satz ist hier drin.",
            "Der zweite Satz ist hier drin.",
            "Der dritte Satz ist hier drin.",
            "Absatz zwei, nur ein kurzer Satz."
        ]
    );
    // the model's language applies, nothing else was invented
    assert!(passages.iter().all(|p| p.language.as_deref() == Some("de")));
    assert!(
        passages
            .iter()
            .all(|p| p.voice.is_none() && p.speed.is_none() && p.seed.is_none())
    );
    // 4 passages of 800 samples; two pauses of 100 ms and one paragraph pause of 300 ms at 8 kHz; none at the end
    let pcm = eliezer_speech::pcm::read_wav(&answer.bytes).unwrap();
    assert_eq!(pcm.rate, 8000);
    assert_eq!(pcm.samples.len(), 4 * PASSAGE_SAMPLES + 2 * 800 + 2400);
    // the pauses are silent
    assert!(
        pcm.samples[PASSAGE_SAMPLES..PASSAGE_SAMPLES + 800]
            .iter()
            .all(|s| *s == 0)
    );
}

#[tokio::test]
async fn a_chapter_of_many_passages_is_read_in_order() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let sentences: Vec<String> = (0..60)
        .map(|n| format!("Satz Nummer {n} steht hier."))
        .collect();
    let answer = speak(
        &app,
        json!({"model": "reader", "input": sentences.join(" ")}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK);
    let passages = backend.passages.lock().unwrap().clone();
    // one sentence per passage (28 characters, two do not fit into 40)
    assert_eq!(passages.len(), 60);
    for (n, p) in passages.iter().enumerate() {
        assert_eq!(p.text, sentences[n]);
        assert!(p.text.chars().count() <= 40);
    }
    let pcm = eliezer_speech::pcm::read_wav(&answer.bytes).unwrap();
    assert_eq!(pcm.samples.len(), 60 * PASSAGE_SAMPLES + 59 * 800);
}

#[tokio::test]
async fn the_request_settings_reach_the_model_and_override_the_model_defaults() {
    let t = Temp::new();
    let text = base(&t.voices(), "")
        + &READER.replace(
            "extra={language=\"de\"}",
            "extra={language=\"de\",speed=0.9,default_voice=\"mama\",pause_ms=0,paragraph_pause_ms=0}",
        );
    let (app, backend) = setup(&text, vec![]);
    let (kind, body) = form(
        &[("name", "mama"), ("transcript", "Es war einmal.")],
        Some(&sample(2.0)),
    );
    let up = send(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/audio/voices")
            .header("content-type", kind)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(up.status, StatusCode::CREATED, "{}", up.text());
    upload(&app, &[("name", "papa")], Some(&sample(1.0))).await;
    // defaults: the voice, the speed, the language of the model
    let answer = speak(&app, json!({"model": "reader", "input": "Hallo Welt."})).await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.text());
    {
        let sent = backend.passages.lock().unwrap().clone();
        let voice = sent[0].voice.as_ref().unwrap();
        assert_eq!(voice.path, t.voices().join("mama.wav"));
        assert_eq!(voice.transcript.as_deref(), Some("Es war einmal."));
        assert_eq!(sent[0].speed, Some(0.9));
        assert_eq!(sent[0].language.as_deref(), Some("de"));
        // no pauses configured: the joined audio is the passage only
        let pcm = eliezer_speech::pcm::read_wav(&answer.bytes).unwrap();
        assert_eq!(pcm.samples.len(), PASSAGE_SAMPLES);
    }
    // the request wins
    let answer = speak(
        &app,
        json!({"model": "reader", "input": "Hallo Welt.", "voice": "papa", "speed": 1.5,
               "language": "en", "seed": 4294967295_u64, "response_format": "wav"}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK);
    let sent = backend.passages.lock().unwrap().clone();
    let last = sent.last().unwrap();
    assert_eq!(
        last.voice.as_ref().unwrap().path,
        t.voices().join("papa.wav")
    );
    assert_eq!(last.voice.as_ref().unwrap().transcript, None);
    assert_eq!((last.speed, last.seed), (Some(1.5), Some(4_294_967_295)));
    assert_eq!(last.language.as_deref(), Some("en"));
    // an empty voice is the default voice
    speak(
        &app,
        json!({"model": "reader", "input": "Hallo.", "voice": ""}),
    )
    .await;
    let sent = backend.passages.lock().unwrap().clone();
    assert_eq!(
        sent.last().unwrap().voice.as_ref().unwrap().path,
        t.voices().join("mama.wav")
    );
}

#[tokio::test]
async fn without_a_voice_the_models_own_speaker_is_used() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let answer = speak(
        &app,
        json!({"model": "reader", "input": "Hallo.", "voice": null}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK);
    assert!(backend.passages.lock().unwrap()[0].voice.is_none());
}

#[tokio::test]
async fn every_wrong_request_is_refused_before_a_passage_is_read() {
    let t = Temp::new();
    let settings = "max_input_chars=100";
    let (app, backend) = setup(&(base(&t.voices(), settings) + READER + MUSIC), vec![]);
    upload(&app, &[("name", "mama")], Some(&sample(1.0))).await;
    let good = json!({"model": "reader", "input": "Hallo Welt."});
    let long = "a".repeat(101);
    let hostile_voices = [
        "../mama",
        "../../etc/passwd",
        "a/b",
        "/etc/passwd",
        "mama/../mama",
        "mama.wav",
        ".",
        "..",
        "mama\0",
        "ma ma",
        "nobody",
        "MAMA",
        &"m".repeat(500),
    ];
    let mut cases: Vec<(String, Value, StatusCode)> = vec![
        (
            "no input".into(),
            json!({"model": "reader"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "empty input".into(),
            json!({"model": "reader", "input": "  "}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "input not text".into(),
            json!({"model": "reader", "input": 5}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "input too long".into(),
            json!({"model": "reader", "input": long}),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (
            "no letters".into(),
            json!({"model": "reader", "input": "* * * ---"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "unknown model".into(),
            json!({"model": "nothing", "input": "Hallo."}),
            StatusCode::NOT_FOUND,
        ),
        (
            "two models, none named".into(),
            json!({"input": "Hallo."}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "voice not text".into(),
            json!({"model": "reader", "input": "Hallo.", "voice": 5}),
            StatusCode::BAD_REQUEST,
        ),
    ];
    for voice in hostile_voices {
        cases.push((
            format!("voice {voice:?}"),
            json!({"model": "reader", "input": "Hallo.", "voice": voice}),
            StatusCode::BAD_REQUEST,
        ));
    }
    for (field, value) in [
        ("response_format", json!("mp3")),
        ("response_format", json!("flac")),
        ("speed", json!(0)),
        ("speed", json!(5)),
        ("speed", json!("fast")),
        ("language", json!("deutsch")),
        ("language", json!("DE")),
        ("language", json!(5)),
        ("seed", json!(-1)),
        ("seed", json!(4294967296_u64)),
        ("seed", json!("abc")),
        ("seed", json!(1.5)),
    ] {
        let mut body = good.clone();
        body[field] = value.clone();
        cases.push((format!("{field} {value}"), body, StatusCode::BAD_REQUEST));
    }
    for (name, body, expected) in cases {
        let answer = speak(&app, body).await;
        assert_eq!(answer.status, expected, "{name}: {}", answer.text());
        assert!(answer.kind.contains("json"), "{name}: {}", answer.kind);
    }
    assert!(backend.passages.lock().unwrap().is_empty());
    // not JSON at all
    let answer = send(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "text/plain")
            .body(Body::from("hello"))
            .unwrap(),
    )
    .await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    // the good request works, and so does the known voice
    assert_eq!(speak(&app, good.clone()).await.status, StatusCode::OK);
    let mut known = good;
    known["voice"] = json!("mama");
    assert_eq!(speak(&app, known).await.status, StatusCode::OK);
}

#[tokio::test]
async fn a_body_beyond_the_limit_of_the_server_is_too_large() {
    let t = Temp::new();
    let (app, backend) = setup(
        &(base(&t.voices(), "max_input_chars=400000") + READER),
        vec![],
    );
    let answer = speak(
        &app,
        json!({"model": "reader", "input": "a ".repeat(1_100_000)}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(backend.passages.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_failing_passage_fails_the_request_and_the_endpoint_still_chooses_by_kind() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER + MUSIC), vec![]);
    *backend.fail_at.lock().unwrap() = Some((1, BackendError::Crashed));
    let text = "Der erste Satz ist hier drin. Der zweite Satz ist hier drin. Der dritte Satz ist hier drin.";
    let answer = speak(&app, json!({"model": "reader", "input": text})).await;
    assert_eq!(answer.status, StatusCode::BAD_GATEWAY, "{}", answer.text());
    assert_eq!(answer.json()["error"]["type"], "backend_error");
    assert_eq!(backend.passages.lock().unwrap().len(), 2);
    let next = backend.passages.lock().unwrap().len();
    *backend.fail_at.lock().unwrap() = Some((next, BackendError::InvalidRequest));
    let answer = speak(&app, json!({"model": "reader", "input": "Hallo."})).await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    *backend.fail_at.lock().unwrap() = None;
    // the same endpoint composes with the music model, as before
    let answer = speak(
        &app,
        json!({"model": "composer", "input": "[verse] la", "instructions": "calm"}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.text());
    assert_eq!(backend.songs.lock().unwrap().len(), 1);
    // a music request does not reach the reader and a reading does not reach the music model
    let before = backend.passages.lock().unwrap().len();
    let answer = speak(&app, json!({"model": "composer", "input": "la"})).await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    assert_eq!(backend.passages.lock().unwrap().len(), before);
    assert_eq!(backend.songs.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn the_only_speech_model_needs_no_name_and_other_kinds_are_refused() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    assert_eq!(
        speak(&app, json!({"input": "Hallo."})).await.status,
        StatusCode::OK
    );
    assert_eq!(backend.passages.lock().unwrap().len(), 1);
    let llm = format!(
        "{}{}",
        base(&t.voices(), "") + READER,
        r#"
[backend.llamacpp]
enabled=true
[[model]]
key="writer"
backend="llamacpp"
kind="llm"
source={file="/var/lib/eliezer/models/w.gguf"}
size_gb=5.0
arch={layers=2,kv_heads=2,head_dim=64}
"#
    );
    let (app, _) = setup(&llm, vec![]);
    let answer = speak(&app, json!({"model": "writer", "input": "Hallo."})).await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn the_model_list_says_which_models_read_aloud() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let list = get(&app, "/api/v1/models").await.json();
    let entry = &list["models"][0];
    assert_eq!(entry["key"], "reader");
    assert_eq!(entry["type"], "tts");
    assert_eq!(entry["capabilities"]["speech_synthesis"], true);
    assert_eq!(entry["capabilities"]["voice_cloning"], true);
    assert!(entry["max_context_length"].is_null());
}

// ---- voices ---------------------------------------------------------------------------------------------------

#[tokio::test]
async fn voices_are_uploaded_listed_replaced_and_deleted() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    // nothing yet, and no directory
    let answer = get(&app, "/v1/audio/voices").await;
    assert_eq!(answer.status, StatusCode::OK);
    assert_eq!(answer.json(), json!({"voices": []}));
    assert!(!t.voices().exists());
    let answer = upload(&app, &[("name", "mama")], Some(&sample(3.0))).await;
    assert_eq!(answer.status, StatusCode::CREATED, "{}", answer.text());
    assert_eq!(answer.json(), json!({"name": "mama", "replaced": false}));
    upload(
        &app,
        &[("name", "Papa_2"), ("transcript", " Hallo Welt. ")],
        Some(&sample(1.0)),
    )
    .await;
    // the directory is private, the files too; the sample is mono 16 bit at 24 kHz
    assert_eq!(mode(&t.voices()), 0o700);
    assert_eq!(mode(&t.voices().join("mama.wav")), 0o600);
    let pcm = eliezer_speech::pcm::read_wav(&std::fs::read(t.voices().join("mama.wav")).unwrap())
        .unwrap();
    assert_eq!((pcm.rate, pcm.channels), (24_000, 1));
    assert!(
        (pcm.samples.len() as i64 - 72_000).abs() < 1500,
        "{}",
        pcm.samples.len()
    );
    assert_eq!(
        std::fs::read_to_string(t.voices().join("Papa_2.txt")).unwrap(),
        "Hallo Welt."
    );
    assert!(!t.voices().join("mama.txt").exists());
    // no leftovers of the upload
    let mut files: Vec<String> = std::fs::read_dir(t.voices())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    assert_eq!(files, ["Papa_2.txt", "Papa_2.wav", "mama.wav"]);
    // the list holds names only
    let answer = get(&app, "/v1/audio/voices").await;
    assert_eq!(answer.json(), json!({"voices": ["Papa_2", "mama"]}));
    let shown = answer.text();
    assert!(!shown.contains('/') && !shown.contains(".wav") && !shown.contains("Hallo"));
    // an existing name is never overwritten without replace=true
    let before = std::fs::read(t.voices().join("mama.wav")).unwrap();
    let answer = upload(&app, &[("name", "mama")], Some(&sample(1.0))).await;
    assert_eq!(answer.status, StatusCode::CONFLICT);
    assert_eq!(std::fs::read(t.voices().join("mama.wav")).unwrap(), before);
    let answer = upload(
        &app,
        &[("name", "mama"), ("replace", "false")],
        Some(&sample(1.0)),
    )
    .await;
    assert_eq!(answer.status, StatusCode::CONFLICT);
    let answer = upload(
        &app,
        &[("name", "mama"), ("replace", "yes")],
        Some(&sample(1.0)),
    )
    .await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    let answer = upload(
        &app,
        &[("name", "Papa_2"), ("replace", "true")],
        Some(&sample(1.0)),
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK);
    assert_eq!(answer.json(), json!({"name": "Papa_2", "replaced": true}));
    // the old transcript went with the old sample
    assert!(!t.voices().join("Papa_2.txt").exists());
    // delete
    let answer = delete(&app, "/v1/audio/voices/mama").await;
    assert_eq!(answer.status, StatusCode::OK);
    assert!(!t.voices().join("mama.wav").exists());
    assert_eq!(
        delete(&app, "/v1/audio/voices/mama").await.status,
        StatusCode::NOT_FOUND
    );
    upload(
        &app,
        &[("name", "x"), ("transcript", "words")],
        Some(&sample(1.0)),
    )
    .await;
    assert!(t.voices().join("x.txt").exists());
    delete(&app, "/v1/audio/voices/x").await;
    assert!(!t.voices().join("x.txt").exists() && !t.voices().join("x.wav").exists());
    assert_eq!(
        get(&app, "/v1/audio/voices").await.json(),
        json!({"voices": ["Papa_2"]})
    );
}

#[tokio::test]
async fn an_uploaded_sample_is_never_served_back() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    upload(&app, &[("name", "mama")], Some(&sample(1.0))).await;
    let sample = std::fs::read(t.voices().join("mama.wav")).unwrap();
    // some bytes of the sound itself, after the header
    let probe = &sample[100..140];
    for uri in [
        "/v1/audio/voices/mama",
        "/v1/audio/voices/mama.wav",
        "/v1/audio/voices?name=mama",
        "/voices/mama.wav",
        "/v1/audio/voices",
    ] {
        let answer = get(&app, uri).await;
        assert!(
            !answer.bytes.windows(probe.len()).any(|w| w == probe),
            "{uri}"
        );
        assert!(!answer.kind.contains("audio"), "{uri}: {}", answer.kind);
    }
}

#[tokio::test]
async fn bad_voice_uploads_are_refused_and_leave_nothing_behind() {
    let t = Temp::new();
    let settings =
        "voice_max_file_mb=1\nvoice_max_seconds=5\nvoice_name_max_chars=10\nmax_voices=2";
    let (app, _) = setup(&(base(&t.voices(), settings) + READER), vec![]);
    let ok = sample(1.0);
    for name in [
        "",
        "../x",
        "a/b",
        "a\\b",
        ".",
        "..",
        ".x",
        "x.wav",
        "a b",
        "a\0b",
        "Müller",
        "elevenchars",
        "a:b",
        "$(id)",
        "x\ny",
        "..%2Fx",
    ] {
        let answer = upload(&app, &[("name", name)], Some(&ok)).await;
        assert_eq!(
            answer.status,
            StatusCode::BAD_REQUEST,
            "{name:?}: {}",
            answer.text()
        );
    }
    assert_eq!(
        upload(&app, &[], Some(&ok)).await.status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        upload(&app, &[("name", "a")], None).await.status,
        StatusCode::BAD_REQUEST
    );
    // not audio, nothing, too long a recording, too big a file
    let answer = upload(&app, &[("name", "a")], Some(b"this is not audio at all")).await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        upload(&app, &[("name", "a")], Some(b"")).await.status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        upload(&app, &[("name", "a")], Some(&sample(8.0)))
            .await
            .status,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let big = vec![0_u8; (1 << 20) + 1];
    assert_eq!(
        upload(&app, &[("name", "a")], Some(&big)).await.status,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let long_transcript = "w".repeat(9000);
    assert_eq!(
        upload(
            &app,
            &[("name", "a"), ("transcript", &long_transcript)],
            Some(&ok)
        )
        .await
        .status,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    // not multipart
    let answer = send(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/audio/voices")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap(),
    )
    .await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        get(&app, "/v1/audio/voices").await.json(),
        json!({"voices": []})
    );
    // the number of voices is limited, replacing is not adding
    assert_eq!(
        upload(&app, &[("name", "a")], Some(&ok)).await.status,
        StatusCode::CREATED
    );
    assert_eq!(
        upload(&app, &[("name", "b")], Some(&ok)).await.status,
        StatusCode::CREATED
    );
    assert_eq!(
        upload(&app, &[("name", "c")], Some(&ok)).await.status,
        StatusCode::CONFLICT
    );
    assert_eq!(
        upload(&app, &[("name", "a"), ("replace", "true")], Some(&ok))
            .await
            .status,
        StatusCode::OK
    );
    // only the voices are in the directory
    let mut files: Vec<String> = std::fs::read_dir(t.voices())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    assert_eq!(files, ["a.wav", "b.wav"]);
}

#[tokio::test]
async fn delete_checks_the_name_against_the_voices() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    upload(&app, &[("name", "mama")], Some(&sample(1.0))).await;
    // a file next to the voices that a path trick would reach
    let secret = t.0.join("secret.wav");
    std::fs::write(&secret, "keep").unwrap();
    for uri in [
        "/v1/audio/voices/..%2Fsecret",
        "/v1/audio/voices/..%2Fsecret.wav",
        "/v1/audio/voices/%2E%2E",
        "/v1/audio/voices/a%2Fb",
        "/v1/audio/voices/mama.wav",
        "/v1/audio/voices/",
        "/v1/audio/voices/%00",
    ] {
        let answer = delete(&app, uri).await;
        assert!(
            matches!(
                answer.status,
                StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED
            ),
            "{uri}: {}",
            answer.status
        );
    }
    let long = format!("/v1/audio/voices/{}", "m".repeat(500));
    assert_eq!(delete(&app, &long).await.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        delete(&app, "/v1/audio/voices/nobody").await.status,
        StatusCode::NOT_FOUND
    );
    assert_eq!(std::fs::read_to_string(&secret).unwrap(), "keep");
    assert!(t.voices().join("mama.wav").exists());
}

#[tokio::test]
async fn links_and_odd_files_in_the_voice_directory_are_not_voices() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    upload(&app, &[("name", "mama")], Some(&sample(1.0))).await;
    let outside = t.0.join("outside.wav");
    std::fs::write(&outside, "not a voice of the service").unwrap();
    std::os::unix::fs::symlink(&outside, t.voices().join("evil.wav")).unwrap();
    std::fs::create_dir(t.voices().join("dir.wav")).unwrap();
    std::fs::write(t.voices().join("bad name.wav"), "x").unwrap();
    std::fs::write(t.voices().join("notes.txt"), "x").unwrap();
    std::fs::write(t.voices().join(".hidden.wav"), "x").unwrap();
    assert_eq!(
        get(&app, "/v1/audio/voices").await.json(),
        json!({"voices": ["mama"]})
    );
    for name in ["evil", "dir", "notes"] {
        let answer = speak(
            &app,
            json!({"model": "reader", "input": "Hallo.", "voice": name}),
        )
        .await;
        assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{name}");
        assert_eq!(
            delete(&app, &format!("/v1/audio/voices/{name}"))
                .await
                .status,
            StatusCode::NOT_FOUND
        );
    }
    assert!(backend.passages.lock().unwrap().is_empty());
    assert!(outside.exists());
}

#[tokio::test]
async fn the_voice_endpoints_need_the_api_key_like_the_other_modifying_endpoints() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec!["secret-key".into()]);
    assert_eq!(
        get(&app, "/v1/audio/voices").await.status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        upload_with(&app, &[("name", "mama")], Some(&sample(1.0)), None)
            .await
            .status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        upload_with(&app, &[("name", "mama")], Some(&sample(1.0)), Some("wrong"))
            .await
            .status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        delete(&app, "/v1/audio/voices/mama").await.status,
        StatusCode::UNAUTHORIZED
    );
    assert!(!t.voices().exists());
    let answer = upload_with(
        &app,
        &[("name", "mama")],
        Some(&sample(1.0)),
        Some("secret-key"),
    )
    .await;
    assert_eq!(answer.status, StatusCode::CREATED);
    // the key is not echoed anywhere
    assert!(!answer.text().contains("secret-key"));
    let request = Request::builder()
        .uri("/v1/audio/voices")
        .header("authorization", "Bearer secret-key")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        send(&app, request).await.json(),
        json!({"voices": ["mama"]})
    );
    let request = Request::builder()
        .method("DELETE")
        .uri("/v1/audio/voices/mama")
        .header("authorization", "Bearer secret-key")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, request).await.status, StatusCode::OK);
    // reading aloud needs it as well
    let answer = speak(&app, json!({"model": "reader", "input": "Hallo."})).await;
    assert_eq!(answer.status, StatusCode::UNAUTHORIZED);
}

// ---- streaming -------------------------------------------------------------------------------------------------

const FOUR: &str = "Der erste Satz ist hier drin. Der zweite Satz ist hier drin. Der dritte Satz ist hier drin.\n\nAbsatz zwei, nur ein kurzer Satz.";
/// Bytes of one fake passage (16 bit samples).
const PASSAGE_BYTES: usize = PASSAGE_SAMPLES * 2;
/// 4 passages, two pauses of 100 ms and a paragraph pause of 300 ms at 8 kHz, in bytes.
const FOUR_BYTES: usize = 4 * PASSAGE_BYTES + (2 * 800 + 2400) * 2;

/// The answer as it arrives: the head of the response and its body as a stream of data frames.
struct Streamed {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    body: axum::body::BodyDataStream,
}
impl Streamed {
    fn header(&self, name: &str) -> &str {
        self.headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
    }
    /// The next frame; `None` at the end, `Some(Err)` when the transfer was aborted.
    async fn next(&mut self) -> Option<Result<axum::body::Bytes, axum::Error>> {
        use futures_core::Stream;
        std::future::poll_fn(|cx| std::pin::Pin::new(&mut self.body).poll_next(cx)).await
    }
    /// Everything up to the end; `Err` with the bytes before it when the transfer was aborted.
    async fn rest(&mut self) -> Result<Vec<u8>, Vec<u8>> {
        let mut bytes = Vec::new();
        while let Some(frame) = self.next().await {
            match frame {
                Ok(frame) => bytes.extend_from_slice(&frame),
                Err(_) => return Err(bytes),
            }
        }
        Ok(bytes)
    }
}
async fn open(app: &Router, body: Value) -> Streamed {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let (head, body) = response.into_parts();
    Streamed {
        status: head.status,
        headers: head.headers,
        body: body.into_data_stream(),
    }
}
/// Waits until `done` holds; the tests have no fixed sleeps, only this bounded wait for a task to notice a change.
async fn until(done: impl Fn() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while !done() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the condition did not become true");
}

#[tokio::test]
async fn a_streamed_wav_has_an_open_length_and_the_samples_of_the_whole_answer() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let whole = speak(&app, json!({"model": "reader", "input": FOUR})).await;
    let mut streamed = open(
        &app,
        json!({"model": "reader", "input": FOUR, "stream": true}),
    )
    .await;
    assert_eq!(streamed.status, StatusCode::OK);
    assert_eq!(streamed.header("content-type"), "audio/wav");
    // no length: a chunked body
    assert_eq!(streamed.header("content-length"), "");
    assert_eq!(streamed.header("x-audio-sample-rate"), "8000");
    assert_eq!(streamed.header("x-audio-channels"), "1");
    assert_eq!(streamed.header("x-audio-bits"), "16");
    let bytes = streamed.rest().await.unwrap();
    assert_eq!(bytes.len(), 44 + FOUR_BYTES);
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[4..8], &[0xFF; 4]);
    assert_eq!(&bytes[8..16], b"WAVEfmt ");
    assert_eq!(&bytes[22..24], &1_u16.to_le_bytes());
    assert_eq!(&bytes[24..28], &8000_u32.to_le_bytes());
    assert_eq!(&bytes[28..32], &16000_u32.to_le_bytes());
    assert_eq!(&bytes[36..40], b"data");
    assert_eq!(&bytes[40..44], &[0xFF; 4]);
    // the same audio as the non-streamed answer, apart from the two sizes
    assert_eq!(&bytes[44..], &whole.bytes[44..]);
    let pcm = eliezer_speech::pcm::read_wav(&bytes).unwrap();
    assert_eq!(pcm.rate, 8000);
    assert_eq!(pcm.samples.len() * 2, FOUR_BYTES);
}

#[tokio::test]
async fn a_streamed_pcm_answer_is_raw_samples_with_the_format_in_headers() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let whole = speak(&app, json!({"model": "reader", "input": FOUR})).await;
    let mut streamed = open(
        &app,
        json!({"model": "reader", "input": FOUR, "stream": true, "response_format": "pcm"}),
    )
    .await;
    assert_eq!(streamed.status, StatusCode::OK);
    assert_eq!(streamed.header("content-type"), "audio/pcm");
    assert_eq!(streamed.header("x-audio-sample-rate"), "8000");
    assert_eq!(streamed.header("x-audio-channels"), "1");
    assert_eq!(streamed.header("x-audio-bits"), "16");
    let bytes = streamed.rest().await.unwrap();
    // no header of any kind: exactly the samples of the WAV answer
    assert_eq!(bytes, whole.bytes[44..]);
}

#[tokio::test]
async fn pcm_without_stream_is_one_complete_body() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let whole = speak(&app, json!({"model": "reader", "input": FOUR})).await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "reader", "input": FOUR, "response_format": "pcm"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let h = |name: &str| response.headers()[name].to_str().unwrap().to_owned();
    assert_eq!(h("content-type"), "audio/pcm");
    assert_eq!(h("content-length"), FOUR_BYTES.to_string());
    assert_eq!(h("x-audio-sample-rate"), "8000");
    assert_eq!(h("x-audio-channels"), "1");
    assert_eq!(h("x-audio-bits"), "16");
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    assert_eq!(bytes.as_ref(), &whole.bytes[44..]);
}

#[tokio::test]
async fn the_stream_and_format_fields_are_checked() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    for body in [
        json!({"model": "reader", "input": "Hallo.", "stream": "yes"}),
        json!({"model": "reader", "input": "Hallo.", "response_format": "mp3"}),
        json!({"model": "reader", "input": "Hallo.", "stream": true, "response_format": "flac"}),
    ] {
        let answer = speak(&app, body).await;
        assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{}", answer.text());
    }
    // `stream: false` and a missing field are the plain answer, and the limits still apply when streaming
    let answer = speak(
        &app,
        json!({"model": "reader", "input": "Hallo.", "stream": false}),
    )
    .await;
    assert_eq!(answer.kind, "audio/wav");
    assert_eq!(
        eliezer_speech::pcm::read_wav(&answer.bytes).unwrap().rate,
        8000
    );
    let text = base(&t.voices(), "max_input_chars=10") + READER;
    let (app, backend) = setup(&text, vec![]);
    let answer = speak(
        &app,
        json!({"model": "reader", "input": "Dieser Text ist zu lang.", "stream": true}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(backend.passages.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_first_bytes_are_sent_before_the_last_passage_is_read() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    // the last of the four passages waits until the test lets it go
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    *backend.gate.lock().unwrap() = Some((3, gate.clone()));
    let mut streamed = open(
        &app,
        json!({"model": "reader", "input": FOUR, "stream": true}),
    )
    .await;
    assert_eq!(streamed.status, StatusCode::OK);
    let first = streamed.next().await.unwrap().unwrap();
    // header, the first passage and the pause after it
    assert_eq!(first.len(), 44 + PASSAGE_BYTES + 1600);
    assert!(backend.finished.load(Ordering::SeqCst) < 4);
    // the last passage is not read before it is allowed to
    until(|| backend.passages.lock().unwrap().len() == 4).await;
    assert_eq!(backend.finished.load(Ordering::SeqCst), 3);
    gate.add_permits(1);
    let rest = streamed.rest().await.unwrap();
    assert_eq!(first.len() + rest.len(), 44 + FOUR_BYTES);
    assert_eq!(backend.finished.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn a_client_that_goes_away_stops_the_passages_that_follow() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let sentences: Vec<String> = (0..60)
        .map(|n| format!("Satz Nummer {n} steht hier."))
        .collect();
    // the third passage never finishes by itself
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    *backend.gate.lock().unwrap() = Some((2, gate.clone()));
    let mut streamed = open(
        &app,
        json!({"model": "reader", "input": sentences.join(" "), "stream": true}),
    )
    .await;
    assert!(streamed.next().await.unwrap().is_ok());
    until(|| backend.passages.lock().unwrap().len() == 3).await;
    drop(streamed);
    // the running request is dropped, not waited for
    until(|| backend.cancelled.load(Ordering::SeqCst) == 1).await;
    gate.add_permits(100);
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert_eq!(backend.passages.lock().unwrap().len(), 3);
    // the model is free again for the next job
    let answer = speak(&app, json!({"model": "reader", "input": "Noch ein Satz."})).await;
    assert_eq!(answer.status, StatusCode::OK);
}

#[tokio::test]
async fn an_error_after_the_first_bytes_aborts_the_transfer() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    *backend.fail_at.lock().unwrap() = Some((2, BackendError::InvalidRequest));
    let mut streamed = open(
        &app,
        json!({"model": "reader", "input": FOUR, "stream": true}),
    )
    .await;
    // the status is out before the failure is known
    assert_eq!(streamed.status, StatusCode::OK);
    let bytes = streamed.rest().await.unwrap_err();
    // what was read before the failure was sent, nothing after it
    assert_eq!(bytes.len(), 44 + 2 * (PASSAGE_BYTES + 1600));
    assert_eq!(backend.passages.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn an_error_in_the_first_passage_is_an_ordinary_error_answer() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    *backend.fail_at.lock().unwrap() = Some((0, BackendError::Crashed));
    let answer = speak(
        &app,
        json!({"model": "reader", "input": FOUR, "stream": true}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::BAD_GATEWAY);
    assert_eq!(answer.json()["error"]["type"], "backend_error");
}

#[tokio::test]
async fn the_sample_rate_is_the_one_of_the_first_passage() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    // every passage answers at 16 kHz: the stream, its header and the pauses follow that rate (100 ms = 1600 frames)
    *backend.rate_at.lock().unwrap() = Some((0, 16000));
    let mut streamed = open(
        &app,
        json!({"model": "reader", "input": FOUR, "stream": true}),
    )
    .await;
    assert_eq!(streamed.header("x-audio-sample-rate"), "16000");
    let bytes = streamed.rest().await.unwrap();
    assert_eq!(&bytes[24..28], &16000_u32.to_le_bytes());
    assert_eq!(&bytes[28..32], &32000_u32.to_le_bytes());
    assert_eq!(bytes.len(), 44 + 4 * PASSAGE_BYTES + (2 * 1600 + 4800) * 2);
}

#[tokio::test]
async fn a_passage_with_another_sample_rate_ends_the_stream_instead_of_being_converted() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    *backend.rate_at.lock().unwrap() = Some((1, 16000));
    let mut streamed = open(
        &app,
        json!({"model": "reader", "input": FOUR, "stream": true, "response_format": "pcm"}),
    )
    .await;
    assert_eq!(streamed.header("x-audio-sample-rate"), "8000");
    let bytes = streamed.rest().await.unwrap_err();
    // only the first passage and its pause got through
    assert_eq!(bytes.len(), PASSAGE_BYTES + 1600);
    // the same mix is refused in the plain answer too (its second passage is the first one that differs)
    let next = backend.passages.lock().unwrap().len();
    *backend.rate_at.lock().unwrap() = Some((next + 1, 16000));
    let answer = speak(&app, json!({"model": "reader", "input": FOUR})).await;
    assert_eq!(answer.status, StatusCode::BAD_GATEWAY);
}

// ---- streaming over a real connection --------------------------------------------------------------------------

/// Serves the router on a loopback port and returns the address.
async fn serve(app: Router) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    addr
}
async fn connect(addr: std::net::SocketAddr, body: &Value) -> tokio::net::TcpStream {
    use tokio::io::AsyncWriteExt;
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = body.to_string();
    let request = format!(
        "POST /v1/audio/speech HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(request.as_bytes()).await.unwrap();
    socket
}
async fn read_to_end(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    let _ = socket.read_to_end(&mut bytes).await;
    bytes
}
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[tokio::test]
async fn a_stream_is_chunked_and_ends_with_the_closing_chunk() {
    let t = Temp::new();
    let (app, _) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let addr = serve(app).await;
    let mut socket = connect(
        addr,
        &json!({"model": "reader", "input": FOUR, "stream": true, "response_format": "pcm"}),
    )
    .await;
    let bytes = read_to_end(&mut socket).await;
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(400)]).to_lowercase();
    assert!(head.starts_with("http/1.1 200"), "{head}");
    assert!(head.contains("transfer-encoding: chunked"), "{head}");
    assert!(!head.contains("content-length"), "{head}");
    assert!(head.contains("x-audio-sample-rate: 8000"), "{head}");
    assert!(bytes.ends_with(b"0\r\n\r\n"));
}

#[tokio::test]
async fn a_failed_stream_closes_the_connection_without_the_closing_chunk() {
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    *backend.fail_at.lock().unwrap() = Some((2, BackendError::InvalidRequest));
    let addr = serve(app).await;
    let mut socket = connect(
        addr,
        &json!({"model": "reader", "input": FOUR, "stream": true, "response_format": "pcm"}),
    )
    .await;
    let bytes = read_to_end(&mut socket).await;
    assert!(bytes.starts_with(b"HTTP/1.1 200"));
    assert!(contains(&bytes, b"\r\n\r\n"));
    // the client sees a transfer that did not finish properly
    assert!(!bytes.ends_with(b"0\r\n\r\n"));
}

#[tokio::test]
async fn a_closed_connection_stops_the_synthesis() {
    use tokio::io::AsyncReadExt;
    let t = Temp::new();
    let (app, backend) = setup(&(base(&t.voices(), "") + READER), vec![]);
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    *backend.gate.lock().unwrap() = Some((1, gate.clone()));
    let addr = serve(app).await;
    let mut socket = connect(
        addr,
        &json!({"model": "reader", "input": FOUR, "stream": true, "response_format": "pcm"}),
    )
    .await;
    // wait for the first audio, then hang up while the second passage is being read
    let mut seen = Vec::new();
    while !contains(&seen, &[0, 0]) || seen.len() < PASSAGE_BYTES {
        let mut block = [0_u8; 4096];
        let n = socket.read(&mut block).await.unwrap();
        assert!(n > 0, "the stream ended early");
        seen.extend_from_slice(&block[..n]);
    }
    until(|| backend.passages.lock().unwrap().len() == 2).await;
    drop(socket);
    until(|| backend.cancelled.load(Ordering::SeqCst) == 1).await;
    gate.add_permits(10);
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert_eq!(backend.passages.lock().unwrap().len(), 2);
}

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
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

/// The fake backend with a record of the pieces it was asked for. With `scripted`, every piece is answered with
/// segments of two seconds that cover it, each text naming its start in the whole recording (the offsets of the
/// pieces for the settings of `CHUNKED` are 0, 26 and 52 seconds).
struct Recording {
    inner: FakeBackend,
    pieces: Mutex<Vec<TranscriptionRequest>>,
    scripted: bool,
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
    fn transcribe<'a>(
        &'a self,
        running: &'a Running,
        request: TranscriptionRequest,
    ) -> BackendFuture<'a, Transcription> {
        let index = {
            let mut pieces = self.pieces.lock().unwrap();
            pieces.push(request.clone());
            pieces.len() - 1
        };
        if !self.scripted {
            return self.inner.transcribe(running, request);
        }
        Box::pin(async move {
            let seconds = (request.wav.len() - 44) / 32000;
            let offset = [0, 26, 52][index % 3];
            Ok(Transcription {
                language: Some("german".into()),
                segments: (0..seconds / 2)
                    .map(|k| TranscribedSegment {
                        start: (2 * k) as f64,
                        end: (2 * k + 2) as f64,
                        text: format!("s{}", offset + 2 * k),
                    })
                    .collect(),
            })
        })
    }
}

fn model(key: &str) -> Model {
    Model {
        instance: Instance {
            key: key.into(),
            state: State::Unloaded,
            bytes: 10_000_000_000,
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
            bytes: 10_000_000_000,
            device: DevicePolicy::Auto,
        },
        max_context: 32768,
    }
}

const BASE: &str = r#"
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
[backend.whispercpp]
program="/usr/bin/whisper-server"
port_range=[24800,24809]
"#;
/// Pieces of 30 s with 4 s overlap.
const CHUNKED: &str = r#"
[backend.whispercpp]
program="/usr/bin/whisper-server"
port_range=[24800,24809]
chunk_secs=30
overlap_secs=4
"#;
const STT: &str = r#"
[[model]]
key="whisper"
backend="whispercpp"
kind="stt"
source={file="/var/lib/eliezer/models/ggml-model-q5_0.bin"}
vram_gb=2.0
extra={language="de"}
"#;
const SECOND: &str = r#"
[[model]]
key="whisper2"
backend="whispercpp"
kind="stt"
source={file="/var/lib/eliezer/models/ggml-other.bin"}
vram_gb=2.0
"#;
const MUSIC: &str = r#"
[[model]]
key="composer"
backend="audiocpp"
kind="music"
source={file="/var/lib/eliezer/models/music"}
vram_gb=10.0
"#;

fn setup_with(text: &str, scripted: bool) -> (Router, Arc<Recording>, App) {
    let config = Arc::new(Config::parse(text).unwrap());
    let backend = Arc::new(Recording {
        inner: FakeBackend::new(FakeConfig::default()),
        pieces: Mutex::default(),
        scripted,
    });
    let keys: Vec<String> = config.model.iter().map(|m| m.key.clone()).collect();
    let supervisor = Supervisor::spawn(
        keys.iter().map(|k| model(k)).collect(),
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
        vec![],
    );
    (managed_router(app.clone()), backend, app)
}
fn setup(text: &str) -> (Router, Arc<Recording>) {
    let (router, backend, _) = setup_with(text, false);
    (router, backend)
}

/// A silent WAV file (16 kHz, mono, 16 bit).
fn wav(seconds: usize) -> Vec<u8> {
    let data = seconds * 32000;
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&(36 + data as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&[1, 0, 1, 0]);
    wav.extend_from_slice(&16000_u32.to_le_bytes());
    wav.extend_from_slice(&32000_u32.to_le_bytes());
    wav.extend_from_slice(&[2, 0, 16, 0]);
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data as u32).to_le_bytes());
    wav.resize(44 + data, 0);
    wav
}

struct Form(Vec<(String, Option<String>, Vec<u8>)>);
impl Form {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn text(mut self, name: &str, value: &str) -> Self {
        self.0.push((name.into(), None, value.as_bytes().to_vec()));
        self
    }
    fn file(mut self, bytes: Vec<u8>) -> Self {
        self.0
            .push(("file".into(), Some("sermon.wav".into()), bytes));
        self
    }
    fn body(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, file, value) in &self.0 {
            out.extend_from_slice(
                format!("--BOUND\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes(),
            );
            if let Some(file) = file {
                out.extend_from_slice(
                    format!("; filename=\"{file}\"\r\nContent-Type: application/octet-stream")
                        .as_bytes(),
                );
            }
            out.extend_from_slice(b"\r\n\r\n");
            out.extend_from_slice(value);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"--BOUND--\r\n");
        out
    }
}

async fn post(app: &Router, form: &Form) -> (StatusCode, String, String) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", "multipart/form-data; boundary=BOUND")
        .body(Body::from(form.body()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let kind = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = to_bytes(response.into_body(), 1 << 22).await.unwrap();
    (status, kind, String::from_utf8_lossy(&bytes).into_owned())
}
fn json(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| panic!("not JSON: {text}"))
}

#[tokio::test]
async fn json_is_the_default_and_the_model_language_applies() {
    let (app, backend) = setup(&format!("{BASE}{STT}"));
    let (status, kind, body) = post(&app, &Form::new().file(wav(2))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(kind.contains("json"));
    assert_eq!(json(&body), serde_json::json!({"text": "test transcript"}));
    let pieces = backend.pieces.lock().unwrap();
    assert_eq!(pieces.len(), 1);
    assert_eq!(
        pieces[0].language.as_deref(),
        Some("de"),
        "the model's language"
    );
    assert!(
        pieces[0].wav.starts_with(b"RIFF")
            && pieces[0].prompt.is_none()
            && pieces[0].temperature.is_none()
    );
}

#[tokio::test]
async fn the_request_fields_reach_the_program() {
    let (app, backend) = setup(&format!("{BASE}{STT}"));
    let form = Form::new()
        .text("model", "whisper")
        .text("language", "en")
        .text("prompt", "Jesaja, Petrus")
        .text("temperature", "0.4")
        .text("timestamp_granularities[]", "segment")
        .text("stream", "false")
        .file(wav(2));
    let (status, _, body) = post(&app, &form).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let pieces = backend.pieces.lock().unwrap();
    assert_eq!(pieces[0].language.as_deref(), Some("en"));
    assert_eq!(pieces[0].prompt.as_deref(), Some("Jesaja, Petrus"));
    assert_eq!(pieces[0].temperature, Some(0.4));
}

#[tokio::test]
async fn every_response_format_has_its_shape() {
    let (app, _) = setup(&format!("{BASE}{STT}"));
    let ask = |format: &'static str| {
        let app = app.clone();
        async move {
            post(
                &app,
                &Form::new().text("response_format", format).file(wav(2)),
            )
            .await
        }
    };
    let (status, kind, body) = ask("text").await;
    assert_eq!(
        (status, body.as_str()),
        (StatusCode::OK, "test transcript\n")
    );
    assert!(kind.starts_with("text/plain"));
    let (_, kind, body) = ask("srt").await;
    assert_eq!(
        body,
        "1\n00:00:00,000 --> 00:00:02,000\ntest transcript\n\n"
    );
    assert!(kind.starts_with("application/x-subrip"));
    let (_, kind, body) = ask("vtt").await;
    assert_eq!(
        body,
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\ntest transcript\n\n"
    );
    assert!(kind.starts_with("text/vtt"));
    let (_, _, body) = ask("json").await;
    assert_eq!(json(&body), serde_json::json!({"text": "test transcript"}));
    let (_, kind, body) = ask("verbose_json").await;
    assert!(kind.contains("json"));
    let value = json(&body);
    assert_eq!(value["task"], "transcribe");
    assert_eq!(value["language"], "de");
    assert_eq!(value["duration"], 2.0);
    assert_eq!(value["text"], "test transcript");
    assert_eq!(
        value["segments"][0],
        serde_json::json!({"id": 0, "start": 0.0, "end": 2.0, "text": "test transcript"})
    );
}

#[tokio::test]
async fn a_long_file_is_cut_and_the_segments_are_joined_with_the_offsets() {
    let text = format!("{BASE}{STT}").replace(
        "[backend.whispercpp]\nprogram=\"/usr/bin/whisper-server\"\nport_range=[24800,24809]\n",
        CHUNKED,
    );
    let (app, backend, _) = setup_with(&text, true);
    let (status, _, body) = post(
        &app,
        &Form::new()
            .text("response_format", "verbose_json")
            .file(wav(70)),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    {
        let pieces = backend.pieces.lock().unwrap();
        // 0..30, 26..56 and 52..70 seconds: a piece is at most 30 s, the overlap is 4 s
        let seconds: Vec<_> = pieces.iter().map(|p| (p.wav.len() - 44) / 32000).collect();
        assert_eq!(seconds, [30, 30, 18]);
    }
    let value = json(&body);
    assert_eq!(value["duration"], 70.0);
    let segments = value["segments"].as_array().unwrap();
    let starts: Vec<f64> = segments
        .iter()
        .map(|s| s["start"].as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = (0..35).map(|k| f64::from(2 * k)).collect();
    assert_eq!(
        starts, expected,
        "every stretch exactly once, times relative to the whole file"
    );
    assert_eq!(segments[14]["text"], "s28");
    assert_eq!(segments[34]["end"], 70.0);
    let ids: Vec<u64> = segments.iter().map(|s| s["id"].as_u64().unwrap()).collect();
    assert_eq!(ids, (0..35).collect::<Vec<u64>>());
    // srt carries the same joined times
    let (_, _, srt) = post(
        &app,
        &Form::new().text("response_format", "srt").file(wav(70)),
    )
    .await;
    assert!(
        srt.contains("\n18\n00:00:34,000 --> 00:00:36,000\ns34\n"),
        "{srt}"
    );
}

#[tokio::test]
async fn every_wrong_request_is_refused_before_a_model_is_asked() {
    let (app, backend) = setup(&format!("{BASE}{STT}{SECOND}{MUSIC}"));
    let good = || Form::new().text("model", "whisper").file(wav(1));
    let not_audio = Form::new()
        .text("model", "whisper")
        .file(b"This is not audio at all.".to_vec());
    let cases: Vec<(&str, Form, StatusCode)> = vec![
        ("not audio", not_audio, StatusCode::BAD_REQUEST),
        (
            "no file",
            Form::new().text("model", "whisper"),
            StatusCode::BAD_REQUEST,
        ),
        (
            "unknown model",
            Form::new().text("model", "nothing").file(wav(1)),
            StatusCode::NOT_FOUND,
        ),
        (
            "not a speech model",
            Form::new().text("model", "composer").file(wav(1)),
            StatusCode::BAD_REQUEST,
        ),
        (
            "two speech models, none named",
            Form::new().file(wav(1)),
            StatusCode::BAD_REQUEST,
        ),
        (
            "response format",
            good().text("response_format", "mp3"),
            StatusCode::BAD_REQUEST,
        ),
        (
            "word timestamps",
            good().text("timestamp_granularities[]", "word"),
            StatusCode::BAD_REQUEST,
        ),
        (
            "temperature high",
            good().text("temperature", "5"),
            StatusCode::BAD_REQUEST,
        ),
        (
            "temperature text",
            good().text("temperature", "warm"),
            StatusCode::BAD_REQUEST,
        ),
        (
            "language name",
            good().text("language", "German"),
            StatusCode::BAD_REQUEST,
        ),
        (
            "language with a command",
            good().text("language", "de; reboot"),
            StatusCode::BAD_REQUEST,
        ),
    ];
    for (name, form, expected) in cases {
        let (status, kind, body) = post(&app, &form).await;
        assert_eq!(status, expected, "{name}: {body}");
        assert!(kind.contains("json"), "{name}: {kind}");
        assert!(json(&body)["error"]["message"].is_string(), "{name}");
    }
    assert!(backend.pieces.lock().unwrap().is_empty());
    // not multipart at all
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let (status, _, _) = post(&app, &good()).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn without_a_speech_model_the_answer_is_not_found() {
    let (app, backend) = setup(&format!("{BASE}{MUSIC}"));
    let (status, _, body) = post(&app, &Form::new().file(wav(1))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(backend.pieces.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_limits_come_from_the_configuration() {
    // the file limit of the model: 1 MB; a 2 MB file is too large, a short one passes
    let small = STT.replace(
        "extra={language=\"de\"}",
        "extra={language=\"de\",max_file_mb=1}",
    );
    let (app, backend) = setup(&format!("{BASE}{small}"));
    let (status, _, body) = post(&app, &Form::new().file(wav(70))).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(json(&body)["error"]["type"], "payload_too_large");
    assert!(backend.pieces.lock().unwrap().is_empty());
    assert_eq!(
        post(&app, &Form::new().file(wav(2))).await.0,
        StatusCode::OK
    );
    // the length limit of the model: 1 minute; the file is small enough but too long
    let short = STT.replace(
        "extra={language=\"de\"}",
        "extra={language=\"de\",max_audio_minutes=1}",
    );
    let (app, _) = setup(&format!("{BASE}{short}"));
    let (status, _, body) = post(&app, &Form::new().file(wav(70))).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(
        post(&app, &Form::new().file(wav(50))).await.0,
        StatusCode::OK
    );
    // the limits of the backend section apply without the model's own
    let section = BASE.replace(
        "port_range=[24800,24809]",
        "port_range=[24800,24809]\nmax_file_mb=1",
    );
    let (app, _) = setup(&format!("{section}{STT}"));
    assert_eq!(
        post(&app, &Form::new().file(wav(70))).await.0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    // the limit is that of the model chosen, not of another one
    let mixed = format!("{BASE}{small}{}", SECOND);
    let (app, _) = setup(&mixed);
    let big = Form::new().text("model", "whisper2").file(wav(70));
    assert_eq!(post(&app, &big).await.0, StatusCode::OK);
    let big = Form::new().text("model", "whisper").file(wav(70));
    assert_eq!(post(&app, &big).await.0, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn a_speech_job_takes_the_gpu_slot_like_a_music_job() {
    let (app, backend, state) = setup_with(&format!("{BASE}{STT}{MUSIC}"), false);
    // the music model is loaded and idle
    let spec = eliezer_backend::LoadSpec {
        model: "composer".into(),
        context: 32768,
        parallel: 1,
        bytes: 10_000_000_000,
        device: DevicePolicy::Auto,
    };
    drop(state.supervisor.acquire(spec).await.unwrap());
    let loaded = |states: &[Instance], key: &str| {
        states
            .iter()
            .any(|i| i.key == key && i.state == State::Loaded)
    };
    let states = state.supervisor.snapshot().await.unwrap();
    assert!(loaded(&states, "composer"));
    assert_eq!(
        post(&app, &Form::new().file(wav(1))).await.0,
        StatusCode::OK
    );
    assert_eq!(backend.pieces.lock().unwrap().len(), 1);
    let states = state.supervisor.snapshot().await.unwrap();
    assert!(loaded(&states, "whisper"));
    assert!(
        !loaded(&states, "composer"),
        "the idle music model made room"
    );
}

#[tokio::test]
async fn the_model_list_says_which_models_transcribe() {
    let (app, _) = setup(&format!("{BASE}{STT}"));
    let request = Request::builder()
        .uri("/api/v1/models")
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let list: Value = serde_json::from_slice(&bytes).unwrap();
    let entry = &list["models"][0];
    assert_eq!(entry["key"], "whisper");
    assert_eq!(entry["type"], "stt");
    assert_eq!(entry["capabilities"]["transcription"], true);
    assert!(entry["capabilities"]["music_generation"].is_null());
    assert!(entry["max_context_length"].is_null());
    // and the OpenAI list names it
    let request = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let list: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(list["data"][0]["id"], "whisper");
}

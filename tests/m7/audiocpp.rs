use eliezer_audiocpp::{
    AudiocppBackend, GENERATION_BASE, MAX_PROGRAM_MESSAGE_CHARS, MusicModel, TtsModel, arguments,
    program_message, speak_body, speech_body,
};
use eliezer_backend::{
    Backend, BackendError, ChatRequest, LoadSpec, MusicRequest, SpeechRequest, VoiceSample,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU16, AtomicU64, Ordering},
    time::Duration,
};

const FAKE: &str = env!("CARGO_BIN_EXE_fake-audio-server");

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "eliezer-audio-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A block of ports per test so that tests running side by side do not meet.
fn ports(width: u16) -> (u16, u16) {
    static NEXT: AtomicU16 = AtomicU16::new(0);
    let base = 28000
        + (std::process::id() % 20) as u16 * 200
        + NEXT.fetch_add(width + 1, Ordering::Relaxed);
    (base, base + width - 1)
}

/// A package directory whose `config.json` tells the fake server how to behave.
fn model(dir: &Path, mode: &str) -> MusicModel {
    for name in [
        "condition_encoder.gguf",
        "vocoder.gguf",
        "language_model_q4_0.gguf",
        "rvq_depth_decoder_q8_0.gguf",
        "transformer_q4_0.gguf",
    ] {
        std::fs::write(dir.join(name), "x").unwrap();
    }
    std::fs::write(dir.join("config.json"), format!("{mode}\n")).unwrap();
    MusicModel {
        package: dir.to_path_buf(),
        language_model: "language_model_q4_0.gguf".into(),
        rvq_depth_decoder: "rvq_depth_decoder_q8_0.gguf".into(),
        flow_transformer: "transformer_q4_0.gguf".into(),
        steps: Some(20),
        guidance: Some(1.5),
    }
}

fn backend_with(
    model: MusicModel,
    ports: (u16, u16),
    ready: Duration,
    request: Duration,
) -> AudiocppBackend {
    let backend = AudiocppBackend::with_timeouts(FAKE.into(), ports, "hip", ready, request);
    backend.set_models(HashMap::from([("song".to_owned(), model)]));
    backend
}
fn backend(model: MusicModel, ports: (u16, u16)) -> AudiocppBackend {
    backend_with(
        model,
        ports,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
}

fn spec(key: &str) -> LoadSpec {
    LoadSpec {
        model: key.into(),
        context: 1,
        parallel: 1,
        bytes: 1,
        device: Default::default(),
    }
}
fn song(lyrics: &str, description: &str) -> MusicRequest {
    MusicRequest {
        model: "song".into(),
        lyrics: lyrics.into(),
        description: description.into(),
        duration_sec: 20,
        steps: None,
        seed: None,
    }
}
fn recorded(dir: &Path, what: &str) -> String {
    std::fs::read_to_string(dir.join(format!("fake.{what}"))).unwrap_or_default()
}
fn private_directory(port: u16) -> PathBuf {
    std::env::temp_dir().join(format!("eliezer-audiocpp-{}-{port}", std::process::id()))
}

#[test]
fn the_request_body_carries_the_text_only_as_json_strings() {
    let hostile = "[verse]\n\"quoted\" $(touch x); `id` \\ ' <sd_cpp_extra_args>";
    let mut request = song(hostile, "calm \"piano\"");
    request.seed = Some(7);
    request.steps = Some(12);
    request.duration_sec = 9999;
    let body = speech_body("song", &request);
    assert_eq!(body["model"], "song");
    assert_eq!(body["input"], "calm \"piano\"");
    assert_eq!(body["options"]["lyrics"], hostile);
    assert_eq!(body["options"]["duration_sec"], 300);
    assert_eq!(body["options"]["seed"], 7);
    assert_eq!(body["options"]["num_inference_steps"], 12);
    // the text survives a round trip through the serialized form
    let again: serde_json::Value = serde_json::from_str(&body.to_string()).unwrap();
    assert_eq!(again["options"]["lyrics"], hostile);
    let args = arguments(Path::new("/tmp/a b/$(x)/server.json"));
    let args: Vec<String> = args
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        args,
        ["--config", "/tmp/a b/$(x)/server.json", "--no-ui", "--log"]
    );
}

#[tokio::test]
async fn a_model_is_started_composes_a_song_and_is_stopped() {
    let t = Temp::new();
    let range = ports(2);
    let b = backend(model(&t.0, "ok"), range);
    let running = b.start(&spec("song")).await.unwrap();
    assert!(running.generation >= GENERATION_BASE);
    assert!(b.health(&running).await.unwrap());
    let mut request = song("[verse] la la", "bright pop");
    request.seed = Some(5);
    let wav = b.music(&running, request).await.unwrap();
    assert!(wav.starts_with(b"RIFF") && &wav[8..12] == b"WAVE");
    let sent: serde_json::Value = serde_json::from_str(&recorded(&t.0, "request")).unwrap();
    assert_eq!(sent["input"], "bright pop");
    assert_eq!(sent["options"]["lyrics"], "[verse] la la");
    assert_eq!(sent["options"]["duration_sec"], 20);
    assert_eq!(sent["options"]["seed"], 5);
    // what the program was started with
    let args = recorded(&t.0, "args");
    let args: Vec<&str> = args.lines().collect();
    assert_eq!(args[0], "--config");
    assert_eq!(&args[2..], ["--no-ui", "--log"]);
    let config: serde_json::Value = serde_json::from_str(&recorded(&t.0, "config")).unwrap();
    let port = config["port"].as_u64().unwrap() as u16;
    assert!((range.0..=range.1).contains(&port));
    assert_eq!(config["host"], "127.0.0.1");
    assert_eq!(config["backend"], "hip");
    assert_eq!(config["lazy_load"], false);
    assert_eq!(config["ui"], false);
    assert_eq!(config["busy_timeout_ms"], 0);
    let entry = &config["models"][0];
    assert_eq!(entry["id"], "song");
    assert_eq!(entry["family"], "minimax_music3");
    assert_eq!(entry["task"], "gen");
    assert_eq!(entry["path"], t.0.display().to_string());
    let session = &entry["session_options"];
    assert_eq!(
        session["minimax_music3.language_model_gguf"],
        "language_model_q4_0.gguf"
    );
    assert_eq!(
        session["minimax_music3.rvq_depth_decoder_gguf"],
        "rvq_depth_decoder_q8_0.gguf"
    );
    assert_eq!(
        session["minimax_music3.flow_transformer_gguf"],
        "transformer_q4_0.gguf"
    );
    assert_eq!(entry["default_request_options"]["num_inference_steps"], 20);
    assert_eq!(entry["default_request_options"]["guidance_scale"], 1.5);
    // the configuration file lives in a private directory that is gone after the stop
    let directory = private_directory(port);
    assert!(directory.join("server.json").is_file());
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        std::fs::metadata(directory.join("server.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    b.stop(&running).await.unwrap();
    assert!(!directory.exists());
    assert!(!b.health(&running).await.unwrap());
    assert_eq!(
        b.music(&running, song("x", "y")).await,
        Err(BackendError::Crashed)
    );
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        std::net::TcpListener::bind(("127.0.0.1", port)).is_ok(),
        "the port is free again"
    );
}

#[tokio::test]
async fn the_child_gets_a_cleared_environment() {
    let t = Temp::new();
    assert!(std::env::var_os("CARGO_PKG_NAME").is_some());
    let b = backend(model(&t.0, "ok"), ports(2));
    let running = b.start(&spec("song")).await.unwrap();
    let env = recorded(&t.0, "env");
    assert!(!env.contains("HF_TOKEN") && !env.contains("CARGO"), "{env}");
    let allowed = [
        "PATH",
        "LD_LIBRARY_PATH",
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ];
    assert!(
        env.lines().all(|n| allowed.contains(&n)),
        "only allowed names: {env}"
    );
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn bad_requests_and_wrong_use_are_refused() {
    let t = Temp::new();
    let b = backend(model(&t.0, "ok"), ports(2));
    let running = b.start(&spec("song")).await.unwrap();
    for (lyrics, description) in [("", "x"), ("   ", "x"), ("x", ""), ("x", " \n ")] {
        assert_eq!(
            b.music(&running, song(lyrics, description)).await,
            Err(BackendError::InvalidRequest),
            "{lyrics:?} {description:?}"
        );
    }
    let chat = ChatRequest {
        model: "song".into(),
        input: "hi".into(),
        reasoning: None,
    };
    assert!(matches!(
        b.chat(&running, chat).await,
        Err(BackendError::InvalidRequest)
    ));
    b.stop(&running).await.unwrap();
    // unknown model
    assert_eq!(
        b.start(&spec("other")).await.err(),
        Some(BackendError::Unavailable)
    );
}

#[tokio::test]
async fn answers_that_are_no_song_are_mapped_to_errors() {
    for (mode, expected) in [
        ("bad-request", BackendError::InvalidRequest),
        ("error-500", BackendError::Crashed),
        ("not-wav", BackendError::Crashed),
    ] {
        let t = Temp::new();
        let b = backend(model(&t.0, mode), ports(2));
        let running = b.start(&spec("song")).await.unwrap();
        assert_eq!(
            b.music(&running, song("x", "y")).await,
            Err(expected),
            "{mode}"
        );
        // a failed request leaves the program alone
        assert!(b.health(&running).await.unwrap(), "{mode}");
        b.stop(&running).await.unwrap();
    }
}

#[tokio::test]
async fn a_program_that_dies_is_noticed() {
    let t = Temp::new();
    let b = backend(model(&t.0, "die-after-first"), ports(2));
    let running = b.start(&spec("song")).await.unwrap();
    assert!(b.music(&running, song("x", "y")).await.is_ok());
    assert_eq!(
        b.music(&running, song("x", "y")).await,
        Err(BackendError::Crashed)
    );
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert!(!b.health(&running).await.unwrap());
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn starting_fails_cleanly() {
    // ends before it listens
    let t = Temp::new();
    let b = backend(model(&t.0, "crash"), ports(2));
    assert_eq!(
        b.start(&spec("song")).await.err(),
        Some(BackendError::Crashed)
    );
    // listens but never lists the model as loaded: the wait ends
    let t = Temp::new();
    let b = backend_with(
        model(&t.0, "never-loaded"),
        ports(2),
        Duration::from_millis(1500),
        Duration::from_secs(5),
    );
    assert_eq!(
        b.start(&spec("song")).await.err(),
        Some(BackendError::Unavailable)
    );
    // a model that takes a while to load is waited for
    let t = Temp::new();
    let b = backend(model(&t.0, "slow-load 1200"), ports(2));
    let started = std::time::Instant::now();
    let running = b.start(&spec("song")).await.unwrap();
    assert!(started.elapsed() >= Duration::from_millis(1000));
    b.stop(&running).await.unwrap();
    // a missing component, and component names that leave the package
    let t = Temp::new();
    let mut broken = model(&t.0, "ok");
    std::fs::remove_file(t.0.join("vocoder.gguf")).unwrap();
    let b = backend(broken.clone(), ports(2));
    assert_eq!(
        b.start(&spec("song")).await.err(),
        Some(BackendError::ModelFile)
    );
    for name in [
        "../evil.gguf",
        "a/b.gguf",
        ".hidden.gguf",
        "",
        "a b.gguf",
        "x;y.gguf",
    ] {
        std::fs::write(t.0.join("vocoder.gguf"), "x").unwrap();
        broken.language_model = name.into();
        let b = backend(broken.clone(), ports(2));
        assert_eq!(
            b.start(&spec("song")).await.err(),
            Some(BackendError::ModelFile),
            "{name:?}"
        );
    }
    // no program: it cannot be started
    let t = Temp::new();
    let b = AudiocppBackend::with_timeouts(
        "/nonexistent/audiocpp_server".into(),
        ports(2),
        "hip",
        Duration::from_secs(2),
        Duration::from_secs(2),
    );
    b.set_models(HashMap::from([("song".to_owned(), model(&t.0, "ok"))]));
    assert_eq!(
        b.start(&spec("song")).await.err(),
        Some(BackendError::Unavailable)
    );
}

#[tokio::test]
async fn instances_use_their_own_ports_and_the_range_is_respected() {
    let t1 = Temp::new();
    let t2 = Temp::new();
    let range = ports(1);
    let backend = AudiocppBackend::with_timeouts(
        FAKE.into(),
        range,
        "hip",
        Duration::from_secs(10),
        Duration::from_secs(10),
    );
    backend.set_models(HashMap::from([
        ("one".to_owned(), model(&t1.0, "ok")),
        ("two".to_owned(), model(&t2.0, "ok")),
    ]));
    let first = backend.start(&spec("one")).await.unwrap();
    // only one port: the second model finds none
    assert_eq!(
        backend.start(&spec("two")).await.err(),
        Some(BackendError::Unavailable)
    );
    backend.stop(&first).await.unwrap();
    let second = backend.start(&spec("two")).await.unwrap();
    backend.stop(&second).await.unwrap();
}

fn tts_model(dir: &Path, mode: &str) -> TtsModel {
    std::fs::write(dir.join("chatterbox-q8_0.gguf"), "x").unwrap();
    std::fs::write(dir.join("config.json"), format!("{mode}\n")).unwrap();
    TtsModel {
        file: dir.join("chatterbox-q8_0.gguf"),
        family: "chatterbox".into(),
        options: std::collections::BTreeMap::from([
            ("exaggeration".to_owned(), 0.6),
            ("max_tokens".to_owned(), 400.0),
        ]),
    }
}
fn passage(text: &str) -> SpeechRequest {
    SpeechRequest {
        model: "reader".into(),
        text: text.into(),
        voice: None,
        language: None,
        speed: None,
        seed: None,
    }
}

#[test]
fn the_speech_body_carries_the_text_and_the_voice_reference_as_json_only() {
    let hostile = "Er sagte: \"$(touch x); `id`\" \\ ' \n";
    let mut request = passage(hostile);
    request.language = Some("de".into());
    request.speed = Some(1.25);
    request.seed = Some(42);
    request.voice = Some(VoiceSample {
        path: PathBuf::from("/var/lib/eliezer/models/voices/mama.wav"),
        transcript: Some("Es war einmal.".into()),
    });
    let body = speak_body("reader", &request).unwrap();
    assert_eq!(body["model"], "reader");
    assert_eq!(body["input"], hostile);
    assert_eq!(body["language"], "de");
    assert_eq!(body["speed"], 1.25);
    assert_eq!(body["seed"], 42);
    assert_eq!(body["response_format"], "wav");
    assert_eq!(body["voice_ref"]["type"], "path");
    assert_eq!(
        body["voice_ref"]["path"],
        "/var/lib/eliezer/models/voices/mama.wav"
    );
    assert_eq!(body["reference_text"], "Es war einmal.");
    // nothing optional is sent when it is not set
    let plain = speak_body("reader", &passage("Hallo.")).unwrap();
    for name in ["language", "speed", "seed", "voice_ref", "reference_text"] {
        assert!(plain.get(name).is_none(), "{name}");
    }
}

#[tokio::test]
async fn a_speech_model_is_started_with_its_file_family_and_options_and_reads_a_passage() {
    let t = Temp::new();
    let range = ports(2);
    let b = AudiocppBackend::with_timeouts(
        FAKE.into(),
        range,
        "hip",
        Duration::from_secs(10),
        Duration::from_secs(10),
    );
    b.set_tts_models(HashMap::from([(
        "reader".to_owned(),
        tts_model(&t.0, "ok"),
    )]));
    let running = b.start(&spec("reader")).await.unwrap();
    let mut request = passage("Es war einmal.");
    request.voice = Some(VoiceSample {
        path: t.0.join("mama.wav"),
        transcript: None,
    });
    let wav = b.speak(&running, request).await.unwrap();
    assert!(wav.starts_with(b"RIFF") && &wav[8..12] == b"WAVE");
    let sent: serde_json::Value = serde_json::from_str(&recorded(&t.0, "request")).unwrap();
    assert_eq!(sent["input"], "Es war einmal.");
    assert_eq!(
        sent["voice_ref"]["path"],
        t.0.join("mama.wav").display().to_string()
    );
    let config: serde_json::Value = serde_json::from_str(&recorded(&t.0, "config")).unwrap();
    let entry = &config["models"][0];
    assert_eq!(entry["id"], "reader");
    assert_eq!(entry["family"], "chatterbox");
    assert_eq!(entry["task"], "tts");
    assert_eq!(entry["mode"], "offline");
    assert_eq!(
        entry["path"],
        t.0.join("chatterbox-q8_0.gguf").display().to_string()
    );
    assert_eq!(entry["default_request_options"]["exaggeration"], 0.6);
    // a whole number is sent as an integer
    assert_eq!(entry["default_request_options"]["max_tokens"], 400);
    assert!(entry["default_request_options"]["max_tokens"].is_i64());
    // a speech model does not compose
    assert_eq!(
        b.music(&running, song("la", "pop")).await,
        Err(BackendError::InvalidRequest)
    );
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn a_speech_model_without_its_file_and_empty_text_are_refused() {
    let t = Temp::new();
    let b = AudiocppBackend::with_timeouts(
        FAKE.into(),
        ports(2),
        "hip",
        Duration::from_secs(10),
        Duration::from_secs(10),
    );
    let mut model = tts_model(&t.0, "ok");
    model.file = t.0.join("missing.gguf");
    b.set_tts_models(HashMap::from([("reader".to_owned(), model)]));
    assert_eq!(
        b.start(&spec("reader")).await.err(),
        Some(BackendError::ModelFile)
    );
    // an unknown model
    assert_eq!(
        b.start(&spec("nobody")).await.err(),
        Some(BackendError::Unavailable)
    );
    b.set_tts_models(HashMap::from([(
        "reader".to_owned(),
        tts_model(&t.0, "ok"),
    )]));
    let running = b.start(&spec("reader")).await.unwrap();
    assert_eq!(
        b.speak(&running, passage("   ")).await,
        Err(BackendError::InvalidRequest)
    );
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn answers_that_are_no_reading_are_mapped_to_errors() {
    for (mode, expected) in [
        ("error-500", BackendError::Crashed),
        ("bad-request", BackendError::InvalidRequest),
        ("not-wav", BackendError::Crashed),
    ] {
        let t = Temp::new();
        let b = AudiocppBackend::with_timeouts(
            FAKE.into(),
            ports(2),
            "hip",
            Duration::from_secs(10),
            Duration::from_secs(10),
        );
        b.set_tts_models(HashMap::from([(
            "reader".to_owned(),
            tts_model(&t.0, mode),
        )]));
        let running = b.start(&spec("reader")).await.unwrap();
        assert_eq!(
            b.speak(&running, passage("Hallo.")).await,
            Err(expected),
            "{mode}"
        );
        b.stop(&running).await.unwrap();
    }
}

#[test]
fn the_message_of_the_program_is_cleaned_and_cut_for_the_log() {
    let body = br#"{"error":{"message":"Chatterbox prepare requires speaker reference audio","type":"server_error"}}"#;
    assert_eq!(
        program_message(body).as_deref(),
        Some("Chatterbox prepare requires speaker reference audio")
    );
    // control characters (a forged log line) are removed, text around them stays
    let hostile = serde_json::json!({"error": {"message": "bad\n\u{1b}[31mred\r\tx\0 end"}});
    assert_eq!(
        program_message(hostile.to_string().as_bytes()).as_deref(),
        Some("bad[31mredx end")
    );
    // cut to the named maximum
    let long =
        serde_json::json!({"error": {"message": "ä".repeat(MAX_PROGRAM_MESSAGE_CHARS + 50)}});
    let cut = program_message(long.to_string().as_bytes()).unwrap();
    assert_eq!(cut.chars().count(), MAX_PROGRAM_MESSAGE_CHARS + 3);
    assert!(cut.ends_with("..."));
    let exact = serde_json::json!({"error": {"message": "x".repeat(MAX_PROGRAM_MESSAGE_CHARS)}});
    assert_eq!(
        program_message(exact.to_string().as_bytes()).unwrap().len(),
        MAX_PROGRAM_MESSAGE_CHARS
    );
    // a plain string error, and bodies without a message
    assert_eq!(
        program_message(br#"{"error":"boom"}"#).as_deref(),
        Some("boom")
    );
    for none in [
        &b""[..],
        b"not json",
        b"{}",
        br#"{"error":{"message":"  \n "}}"#,
        br#"{"error":{"message":5}}"#,
    ] {
        assert_eq!(program_message(none), None);
    }
}

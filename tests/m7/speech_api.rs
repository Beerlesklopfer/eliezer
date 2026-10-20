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
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

/// The fake backend with a record of the songs it was asked for.
struct Recording {
    inner: FakeBackend,
    songs: Mutex<Vec<MusicRequest>>,
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
"#;
const MUSIC: &str = r#"
[[model]]
key="composer"
backend="audiocpp"
kind="music"
source={file="/var/lib/eliezer/models/music"}
vram_gb=10.0
size_gb=8.4
extra={duration_sec=45}
"#;
const SECOND: &str = r#"
[[model]]
key="composer2"
backend="audiocpp"
kind="music"
source={file="/var/lib/eliezer/models/music2"}
vram_gb=10.0
"#;

fn setup(text: &str) -> (Router, Arc<Recording>) {
    let config = Arc::new(Config::parse(text).unwrap());
    let backend = Arc::new(Recording {
        inner: FakeBackend::new(FakeConfig::default()),
        songs: Mutex::default(),
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
    (managed_router(app), backend)
}

async fn post(app: &Router, body: Value) -> (StatusCode, String, Vec<u8>) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let kind = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, kind, bytes.to_vec())
}

#[tokio::test]
async fn lyrics_and_a_description_come_back_as_a_wav_file() {
    let (app, backend) = setup(&format!("{BASE}{MUSIC}"));
    let (status, kind, bytes) = post(
        &app,
        json!({
            "model": "composer",
            "input": "[verse]\nHello world",
            "instructions": "A calm acoustic song",
            "response_format": "wav",
            "seed": 7,
            "duration_sec": 60,
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(kind, "audio/wav");
    assert!(bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE");
    {
        let songs = backend.songs.lock().unwrap();
        assert_eq!(songs[0].lyrics, "[verse]\nHello world");
        assert_eq!(songs[0].description, "A calm acoustic song");
        assert_eq!(songs[0].duration_sec, 60);
        assert_eq!(songs[0].seed, Some(7));
    }
    // the model's own length applies without one in the request, and the only music model needs no name
    let (status, _, _) = post(&app, json!({"input": "la", "instructions": "x"})).await;
    assert_eq!(status, StatusCode::OK);
    let songs = backend.songs.lock().unwrap();
    assert_eq!(songs[1].duration_sec, 45);
    assert_eq!(songs[1].seed, None);
}

#[tokio::test]
async fn every_wrong_request_is_refused_before_a_model_is_asked() {
    let (app, backend) = setup(&format!("{BASE}{MUSIC}{SECOND}"));
    let good = json!({"model": "composer", "input": "la", "instructions": "x"});
    let mut cases: Vec<(&str, Value, StatusCode)> = vec![
        (
            "no lyrics",
            json!({"model": "composer", "instructions": "x"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "empty lyrics",
            json!({"model": "composer", "input": "  ", "instructions": "x"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "no description",
            json!({"model": "composer", "input": "la"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "lyrics not text",
            json!({"model": "composer", "input": 5, "instructions": "x"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "unknown model",
            json!({"model": "nothing", "input": "la", "instructions": "x"}),
            StatusCode::NOT_FOUND,
        ),
        (
            "two music models, none named",
            json!({"input": "la", "instructions": "x"}),
            StatusCode::BAD_REQUEST,
        ),
    ];
    for (field, value) in [
        ("response_format", json!("mp3")),
        ("duration_sec", json!(0)),
        ("duration_sec", json!(301)),
        ("duration_sec", json!("30")),
        ("duration_sec", json!(2.5)),
        ("seed", json!("abc")),
    ] {
        let mut body = good.clone();
        body[field] = value;
        cases.push((field, body, StatusCode::BAD_REQUEST));
    }
    let mut long = good.clone();
    long["input"] = json!("a".repeat(20_001));
    cases.push(("lyrics too long", long, StatusCode::BAD_REQUEST));
    for (name, body, expected) in cases {
        let (status, kind, bytes) = post(&app, body).await;
        assert_eq!(
            status,
            expected,
            "{name}: {}",
            String::from_utf8_lossy(&bytes)
        );
        assert!(kind.contains("json"), "{name}: {kind}");
    }
    assert!(backend.songs.lock().unwrap().is_empty());
    // not JSON at all
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/speech")
        .header("content-type", "text/plain")
        .body(Body::from("hello"))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let (status, _, _) = post(&app, good).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_model_that_makes_no_music_is_refused() {
    let text = format!(
        "{BASE}{MUSIC}{}",
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
    let (app, backend) = setup(&text);
    let (status, _, bytes) = post(
        &app,
        json!({"model": "writer", "input": "la", "instructions": "x"}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert!(backend.songs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_model_list_says_which_models_make_music() {
    let (app, _) = setup(&format!("{BASE}{MUSIC}"));
    let request = Request::builder()
        .uri("/api/v1/models")
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let list: Value = serde_json::from_slice(&bytes).unwrap();
    let entry = &list["models"][0];
    assert_eq!(entry["key"], "composer");
    assert_eq!(entry["type"], "music");
    assert_eq!(entry["capabilities"]["music_generation"], true);
    assert!(entry["capabilities"]["image_generation"].is_null());
    assert!(entry["max_context_length"].is_null());
    assert_eq!(entry["size_bytes"], 8_400_000_000_u64);
}

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use eliezer_api::{App, helper, managed_router};
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

/// The fake backend with a record of what the service asked of it.
struct Recording {
    inner: FakeBackend,
    starts: Mutex<Vec<(String, DevicePolicy)>>,
    chats: Mutex<Vec<String>>,
    images: Mutex<Vec<String>>,
    fail_writer: bool,
}
impl Backend for Recording {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        self.starts
            .lock()
            .unwrap()
            .push((spec.model.clone(), spec.device));
        if self.fail_writer && spec.model == "writer" {
            return Box::pin(async { Err(BackendError::Unavailable) });
        }
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
        self.chats.lock().unwrap().push(request.input.clone());
        self.inner.chat(running, request)
    }
    fn image<'a>(
        &'a self,
        running: &'a Running,
        request: ImageRequest,
    ) -> BackendFuture<'a, Vec<String>> {
        self.images.lock().unwrap().push(request.prompt.clone());
        self.inner.image(running, request)
    }
}

fn model(key: &str, image: bool, bytes: u64) -> Model {
    Model {
        instance: Instance {
            key: key.into(),
            state: State::Unloaded,
            bytes,
            priority: 0,
            pinned: false,
            image,
            in_flight: 0,
            last_used_ms: 0,
        },
        default_spec: LoadSpec {
            model: key.into(),
            context: if image { 32768 } else { 4096 },
            parallel: 1,
            bytes,
            device: DevicePolicy::Auto,
        },
        max_context: if image { 32768 } else { 4096 },
    }
}

const CONFIG: &str = r#"
[server]
listen="127.0.0.1:1234"
[gpu]
vendor="none"
measure="none"
vram_budget_gb=24.0
reserve_gb=2.0
[cpu]
ram_budget_gb=40.0
reserve_gb=4.0
[backend.llamacpp]
enabled=true
[backend.sdcpp]
program="/usr/bin/sd-server"
port_range=[24600,24609]
[[model]]
key="writer"
backend="llamacpp"
kind="llm"
source={file="/var/lib/eliezer/models/w.gguf"}
size_gb=5.0
default_context=4096
max_context=4096
arch={layers=2,kv_heads=2,head_dim=64}
[[model]]
key="pic"
backend="sdcpp"
kind="image"
source={file="/var/lib/eliezer/models/p.gguf"}
vram_gb=9.0
extra={prompt_helper="writer"}
[[model]]
key="plain"
backend="sdcpp"
kind="image"
source={file="/var/lib/eliezer/models/q.gguf"}
vram_gb=9.0
"#;

fn setup(reply: &str, fail_writer: bool) -> (Router, Arc<Recording>) {
    let config = Arc::new(Config::parse(CONFIG).unwrap());
    let backend = Arc::new(Recording {
        inner: FakeBackend::new(FakeConfig {
            chunks: vec![
                Chunk::Text(reply.into()),
                Chunk::Done {
                    finish_reason: "stop".into(),
                    input_tokens: 1,
                    output_tokens: 1,
                },
            ],
            ..Default::default()
        }),
        starts: Mutex::default(),
        chats: Mutex::default(),
        images: Mutex::default(),
        fail_writer,
    });
    let supervisor = Supervisor::spawn(
        vec![
            model("writer", false, 5_000_000_000),
            model("pic", true, 9_000_000_000),
            model("plain", true, 9_000_000_000),
        ],
        backend.clone(),
        Arc::new(TestClock),
        Settings {
            budget: 22_000_000_000,
            ram_budget: 36_000_000_000,
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

async fn draw(app: &Router, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[test]
fn the_description_is_cleaned_before_it_is_used() {
    assert_eq!(
        helper::clean("A red fox in snow."),
        Some("A red fox in snow.".into())
    );
    assert_eq!(
        helper::clean("Here is the prompt:\n\"A red fox\nin deep snow.\"\n"),
        Some("A red fox in deep snow.".into())
    );
    assert_eq!(helper::clean("**Prompt:** `a cube`"), Some("a cube".into()));
    assert_eq!(
        helper::clean("Prompt: a blue sphere"),
        Some("a blue sphere".into())
    );
    assert_eq!(helper::clean("  \n \n"), None);
    assert_eq!(helper::clean("\"\""), None);
    let long = helper::clean(&"word ".repeat(1000)).unwrap();
    assert!(long.chars().count() <= 1200);
    // a single short line that ends with a colon is a description, not a label
    assert_eq!(helper::clean("Roses:"), Some("Roses:".into()));
}

#[tokio::test]
async fn a_short_wish_becomes_a_description_written_on_the_cpu() {
    let (app, backend) = setup(
        "\"A tall sunflower in a field, golden hour, sharp focus.\"",
        false,
    );
    let (status, body) = draw(&app, json!({"model": "pic", "prompt": "Eine Sonnenblume"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["prompt_enhanced"], true);
    assert_eq!(
        body["data"][0]["revised_prompt"],
        "A tall sunflower in a field, golden hour, sharp focus."
    );
    // the picture is drawn from the description, not from the wish
    assert_eq!(
        backend.images.lock().unwrap().as_slice(),
        ["A tall sunflower in a field, golden hour, sharp focus."]
    );
    // the helper got the instruction and the wish, and ran on the CPU
    let chat: Value = serde_json::from_str(&backend.chats.lock().unwrap()[0]).unwrap();
    assert_eq!(chat["messages"][0]["role"], "system");
    assert!(
        chat["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("image generation model")
    );
    assert_eq!(chat["messages"][1]["content"], "Eine Sonnenblume");
    let starts = backend.starts.lock().unwrap().clone();
    assert!(
        starts.contains(&("writer".to_owned(), DevicePolicy::Cpu)),
        "{starts:?}"
    );
    assert!(
        starts
            .iter()
            .any(|(m, d)| m == "pic" && *d == DevicePolicy::Gpu),
        "{starts:?}"
    );
}

#[tokio::test]
async fn the_helper_can_be_switched_off_per_request_and_is_refused_where_there_is_none() {
    let (app, backend) = setup("A description.", false);
    let (status, body) = draw(
        &app,
        json!({"model": "pic", "prompt": "a fox", "enhance_prompt": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["prompt_enhanced"], false);
    assert!(body["data"][0].get("revised_prompt").is_none());
    assert_eq!(backend.images.lock().unwrap().as_slice(), ["a fox"]);
    assert!(
        backend.chats.lock().unwrap().is_empty(),
        "the helper was not asked"
    );
    // a model without a helper draws the prompt as it is
    let (status, body) = draw(&app, json!({"model": "plain", "prompt": "a cat"})).await;
    assert_eq!(
        (status, &body["prompt_enhanced"]),
        (StatusCode::OK, &json!(false))
    );
    // asking for what is not there is an error, a wrong type too
    let (status, _) = draw(
        &app,
        json!({"model": "plain", "prompt": "a cat", "enhance_prompt": true}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = draw(
        &app,
        json!({"model": "pic", "prompt": "a cat", "enhance_prompt": "yes"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn a_helper_that_fails_does_not_stop_the_picture() {
    let (app, backend) = setup("unused", true);
    let (status, body) = draw(&app, json!({"model": "pic", "prompt": "a lighthouse"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["prompt_enhanced"], false);
    assert_eq!(backend.images.lock().unwrap().as_slice(), ["a lighthouse"]);
    // a reply that is empty after cleaning counts as a failure too
    let (app, backend) = setup("\"\"", false);
    let (status, body) = draw(&app, json!({"model": "pic", "prompt": "a bridge"})).await;
    assert_eq!(
        (status, &body["prompt_enhanced"]),
        (StatusCode::OK, &json!(false))
    );
    assert_eq!(backend.images.lock().unwrap().as_slice(), ["a bridge"]);
}

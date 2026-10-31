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

/// Prompt, number of pictures, size and count of one edit.
type Edit = (String, usize, Option<String>, u32);

/// The fake backend with a record of the pictures it was asked to edit.
struct Recording {
    inner: FakeBackend,
    edits: Mutex<Vec<Edit>>,
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
    fn image<'a>(
        &'a self,
        running: &'a Running,
        request: ImageRequest,
    ) -> BackendFuture<'a, Vec<String>> {
        self.edits.lock().unwrap().push((
            request.prompt.clone(),
            request.references.len(),
            request.size.clone(),
            request.count,
        ));
        self.inner.image(running, request)
    }
}

fn model(key: &str) -> Model {
    Model {
        instance: Instance {
            key: key.into(),
            state: State::Unloaded,
            bytes: 9_000_000_000,
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
            bytes: 9_000_000_000,
            device: DevicePolicy::Auto,
        },
        max_context: 32768,
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
[backend.sdcpp]
program="/usr/bin/sd-server"
port_range=[24600,24609]
[[model]]
key="editor"
backend="sdcpp"
kind="image"
source={file="/var/lib/eliezer/models/e.gguf"}
vram_gb=9.0
extra={image_edit=true}
[[model]]
key="painter"
backend="sdcpp"
kind="image"
source={file="/var/lib/eliezer/models/p.gguf"}
vram_gb=9.0
"#;

fn setup() -> (Router, Arc<Recording>) {
    let config = Arc::new(Config::parse(CONFIG).unwrap());
    let backend = Arc::new(Recording {
        inner: FakeBackend::new(FakeConfig::default()),
        edits: Mutex::default(),
    });
    let supervisor = Supervisor::spawn(
        vec![model("editor"), model("painter")],
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

const PNG: &[u8] = b"\x89PNG\r\n\x1a\n-pixels-";
const JPEG: &[u8] = b"\xFF\xD8\xFF\xE0-pixels-";
const WEBP: &[u8] = b"RIFF\x04\x00\x00\x00WEBPVP8 ";

enum Part<'a> {
    Text(&'a str, &'a str),
    File(&'a str, &'a [u8]),
}

async fn post(app: &Router, parts: &[Part<'_>]) -> (StatusCode, Value) {
    let boundary = "XBOUNDARYX";
    let mut body = Vec::new();
    for part in parts {
        match part {
            Part::Text(name, value) => body.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                    .as_bytes(),
            ),
            Part::File(name, data) => {
                body.extend_from_slice(
                    format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"f\"\r\nContent-Type: application/octet-stream\r\n\r\n")
                        .as_bytes(),
                );
                body.extend_from_slice(data);
                body.extend_from_slice(b"\r\n");
            }
        }
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/images/edits")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn pictures_and_a_prompt_come_back_as_new_pictures() {
    let (app, backend) = setup();
    let (status, body) = post(
        &app,
        &[
            Part::Text("model", "editor"),
            Part::Text("prompt", "make it blue"),
            Part::Text("n", "2"),
            Part::Text("size", "512x768"),
            Part::Text("quality", "high"),
            Part::Text("background", "auto"),
            Part::Text("output_format", "jpeg"),
            Part::File("image[]", PNG),
            Part::File("image[]", JPEG),
            Part::File("image[]", WEBP),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert!(body["data"][0]["b64_json"].as_str().unwrap().len() > 20);
    assert!(body["created"].as_u64().unwrap() > 0);
    assert_eq!(
        backend.edits.lock().unwrap()[0],
        ("make it blue".to_owned(), 3, Some("512x768".to_owned()), 2)
    );
    // the single-file field of the OpenAI API, and the only editing model needs no name
    let (status, body) = post(
        &app,
        &[Part::Text("prompt", "again"), Part::File("image", PNG)],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(backend.edits.lock().unwrap()[1].1, 1);
}

#[tokio::test]
async fn every_wrong_request_is_refused_before_a_model_is_asked() {
    let (app, backend) = setup();
    let ok = [
        Part::Text("model", "editor"),
        Part::Text("prompt", "x"),
        Part::File("image", PNG),
    ];
    let cases: Vec<(&str, Vec<Part<'_>>, StatusCode)> = vec![
        (
            "no picture",
            vec![Part::Text("model", "editor"), Part::Text("prompt", "x")],
            StatusCode::BAD_REQUEST,
        ),
        (
            "no prompt",
            vec![Part::Text("model", "editor"), Part::File("image", PNG)],
            StatusCode::BAD_REQUEST,
        ),
        (
            "not a picture",
            vec![
                Part::Text("model", "editor"),
                Part::Text("prompt", "x"),
                Part::File("image", b"#!/bin/sh\nrm -rf /"),
            ],
            StatusCode::BAD_REQUEST,
        ),
        (
            "five pictures",
            vec![
                Part::Text("model", "editor"),
                Part::Text("prompt", "x"),
                Part::File("image[]", PNG),
                Part::File("image[]", PNG),
                Part::File("image[]", PNG),
                Part::File("image[]", PNG),
                Part::File("image[]", PNG),
            ],
            StatusCode::BAD_REQUEST,
        ),
        (
            "a model that does not edit",
            vec![
                Part::Text("model", "painter"),
                Part::Text("prompt", "x"),
                Part::File("image", PNG),
            ],
            StatusCode::BAD_REQUEST,
        ),
        (
            "an unknown model",
            vec![
                Part::Text("model", "nothing"),
                Part::Text("prompt", "x"),
                Part::File("image", PNG),
            ],
            StatusCode::NOT_FOUND,
        ),
        (
            "a mask",
            vec![
                Part::Text("model", "editor"),
                Part::Text("prompt", "x"),
                Part::File("image", PNG),
                Part::File("mask", PNG),
            ],
            StatusCode::BAD_REQUEST,
        ),
        (
            "bad size",
            vec![
                Part::Text("model", "editor"),
                Part::Text("prompt", "x"),
                Part::Text("size", "huge"),
                Part::File("image", PNG),
            ],
            StatusCode::BAD_REQUEST,
        ),
        (
            "bad count",
            vec![
                Part::Text("model", "editor"),
                Part::Text("prompt", "x"),
                Part::Text("n", "9"),
                Part::File("image", PNG),
            ],
            StatusCode::BAD_REQUEST,
        ),
    ];
    for (name, parts, expected) in cases {
        let (status, body) = post(&app, &parts).await;
        assert_eq!(status, expected, "{name}: {body}");
        assert!(
            body["error"].is_object() || body["error"].is_string(),
            "{name}: {body}"
        );
    }
    let big = vec![0x89_u8; 21 << 20];
    let mut huge = b"\x89PNG\r\n\x1a\n".to_vec();
    huge.extend_from_slice(&big);
    let (status, body) = post(
        &app,
        &[
            Part::Text("model", "editor"),
            Part::Text("prompt", "x"),
            Part::File("image", &huge),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert!(backend.edits.lock().unwrap().is_empty());
    // not multipart at all
    let request = Request::builder()
        .method("POST")
        .uri("/v1/images/edits")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // the good one still works afterwards
    let (status, _) = post(&app, &ok).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn the_model_list_says_which_models_edit() {
    let (app, _) = setup();
    let request = Request::builder()
        .uri("/api/v1/models")
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let list: Value = serde_json::from_slice(&bytes).unwrap();
    let find = |key: &str| {
        list["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["key"] == key)
            .unwrap()
            .clone()
    };
    assert_eq!(find("editor")["capabilities"]["image_edit"], true);
    assert_eq!(find("editor")["capabilities"]["image_generation"], true);
    assert!(find("painter")["capabilities"]["image_edit"].is_null());
}

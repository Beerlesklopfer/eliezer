use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use eliezer_api::{App, managed_router};
use eliezer_backend::{supervisor::*, *};
use eliezer_core::{
    config::Config,
    scheduler::{Clock, Instance, State},
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;
struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}
fn app(fake: FakeConfig, keys: Vec<String>) -> (Router, Supervisor) {
    let config = Arc::new(
        Config::parse(
            r#"
[server]
listen="127.0.0.1:1234"
[gpu]
vendor="none"
measure="none"
vram_budget_gb=8.0
reserve_gb=1.0
[backend.mistralrs]
enabled=true
[[model]]
key="a"
backend="mistralrs"
source={hf="test/model"}
kind="llm"
size_gb=1.0
arch={layers=2,kv_heads=2,head_dim=64}
reasoning={options=["none","low","medium","high"],default="none"}
"#,
        )
        .unwrap(),
    );
    let m = &config.model[0];
    let bytes =
        eliezer_core::memory::estimate(m, m.default_context, m.parallel, None, None).unwrap();
    let backend: Arc<dyn Backend> = Arc::new(FakeBackend::new(fake));
    let supervisor = Supervisor::spawn(
        vec![Model {
            instance: Instance {
                key: "a".into(),
                state: State::Unloaded,
                bytes,
                priority: 0,
                pinned: false,
                image: false,
                in_flight: 0,
                last_used_ms: 0,
            },
            default_spec: LoadSpec {
                model: "a".into(),
                context: m.default_context,
                parallel: 1,
                bytes,
                device: eliezer_core::config::DevicePolicy::Auto,
            },
            max_context: m.max_context,
        }],
        backend.clone(),
        Arc::new(TestClock),
        Settings {
            budget: 7_000_000_000,
            ram_budget: 0,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    );
    (
        managed_router(App::new(
            config,
            supervisor.clone(),
            backend,
            Arc::new(eliezer_gpu::sources(
                eliezer_core::config::GpuVendor::None,
                eliezer_core::config::ProbeMethod::None,
                0,
                8_000_000_000,
                "/nonexistent".into(),
            )),
            keys,
        )),
        supervisor,
    )
}
async fn call(app: Router, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder()
        .uri(path)
        .method(if body.is_some() { "POST" } else { "GET" })
        .header("content-type", "application/json")
        .body(
            body.map(|v| Body::from(v.to_string()))
                .unwrap_or_else(Body::empty),
        )
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
#[tokio::test]
async fn management_and_both_chat_protocols() {
    let (app, s) = app(Default::default(), vec![]);
    let (status, models) = call(app.clone(), "/api/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["models"][0]["key"], "a");
    assert!(
        models["models"][0]["loaded_instances"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let (status, load) = call(
        app.clone(),
        "/api/v1/models/load",
        Some(json!({"model":"a"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(load["status"], "loaded");
    let (_, oa) = call(
        app.clone(),
        "/v1/chat/completions",
        Some(json!({"model":"a","messages":[{"role":"user","content":"Hello"}]})),
    )
    .await;
    let (_, anth) = call(
        app.clone(),
        "/v1/messages",
        Some(json!({"model":"a","messages":[{"role":"user","content":"Hello"}],"max_tokens":10})),
    )
    .await;
    assert_eq!(
        oa["choices"][0]["message"]["content"],
        anth["content"][0]["text"]
    );
    let (status, _) = call(
        app.clone(),
        "/api/v1/models/unload",
        Some(json!({"instance_id":"a"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, bad) = call(
        app.clone(),
        "/v1/messages",
        Some(json!({"model":"unknown","messages":[],"max_tokens":1})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(bad["error"]["type"], "not_found_error");
    let (status, _) = call(
        app,
        "/v1/chat/completions",
        Some(json!({"model":"a","messages":[],"reasoning_effort":"xhigh"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    s.shutdown().await;
}
#[tokio::test]
async fn health_is_public_and_auth_is_required() {
    // Random runtime value instead of a key in test data.
    let key = format!("{:?}", std::time::SystemTime::now());
    let (app, s) = app(Default::default(), vec![key.clone()]);
    assert_eq!(
        call(app.clone(), "/api/v1/models", None).await.0,
        StatusCode::UNAUTHORIZED
    );
    let request = Request::builder()
        .uri("/api/v1/models")
        .header("authorization", format!("Bearer {key}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        app.clone().oneshot(request).await.unwrap().status(),
        StatusCode::OK
    );
    let request = Request::builder()
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.oneshot(request).await.unwrap().status(), StatusCode::OK);
    s.shutdown().await;
}
#[tokio::test]
async fn first_sse_frame_arrives_before_backend_finishes() {
    use futures_core::Stream;
    let (app, s) = app(
        FakeConfig {
            chunk_delay: Duration::from_millis(100),
            ..Default::default()
        },
        vec![],
    );
    for path in ["/v1/chat/completions", "/v1/messages"] {
        let request=Request::builder().uri(path).method("POST").header("content-type","application/json")
            .body(Body::from(json!({"model":"a","messages":[],"max_tokens":10,"stream":true,"stream_options":{"include_usage":true}}).to_string())).unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.into_body().into_data_stream();
        let start = std::time::Instant::now();
        let first = std::future::poll_fn(|cx| std::pin::Pin::new(&mut stream).poll_next(cx))
            .await
            .unwrap()
            .unwrap();
        assert!(start.elapsed() < Duration::from_millis(190));
        assert!(first.starts_with(b"data:") || first.starts_with(b"event:"));
        let mut output = String::from_utf8(first.to_vec()).unwrap();
        while let Some(bytes) =
            std::future::poll_fn(|cx| std::pin::Pin::new(&mut stream).poll_next(cx)).await
        {
            output.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
        }
        if path.ends_with("messages") {
            assert!(output.contains("message_stop"));
        } else {
            assert!(output.contains("[DONE]"));
            assert!(output.contains("prompt_tokens"));
        }
    }
    s.shutdown().await;
}

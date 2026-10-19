use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use eliezer_api::{App, managed_router};
use eliezer_backend::{supervisor::*, *};
use eliezer_core::{config::Config, scheduler::Clock};
use eliezer_hub::{Detail, FakeHub, HubError, RepoFile, Summary};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

fn config(search: &str) -> Arc<Config> {
    Arc::new(
        Config::parse(&format!(
            r#"
[server]
listen="127.0.0.1:1234"
[gpu]
vendor="none"
measure="none"
vram_budget_gb=12.0
reserve_gb=2.0
[cpu]
ram_budget_gb=40.0
reserve_gb=4.0
[huggingface]
{search}
[backend.mistralrs]
enabled=true
"#
        ))
        .unwrap(),
    )
}

fn file(name: &str, bytes: u64) -> RepoFile {
    RepoFile {
        name: name.into(),
        bytes,
        sha256: None,
    }
}

fn hub(fail: Option<HubError>) -> FakeHub {
    FakeHub {
        summaries: vec![
            Summary {
                repo: "org/Qwen-GGUF".into(),
                downloads: 100,
                likes: 3,
                task: Some("text-generation".into()),
                license: Some("apache-2.0".into()),
                gguf: true,
            },
            Summary {
                repo: "org/Other-GGUF".into(),
                downloads: 5,
                likes: 0,
                task: Some("text-generation".into()),
                license: Some("mit".into()),
                gguf: true,
            },
        ],
        details: vec![Detail {
            repo: "org/Qwen-GGUF".into(),
            sha: "abc".into(),
            task: Some("text-generation".into()),
            license: Some("apache-2.0".into()),
            gated: false,
            downloads: 100,
            architecture: Some("qwen3moe".into()),
            files: vec![
                file("q-Q4_K_M.gguf", 4_000_000_000),
                file("q-F16.gguf", 20_000_000_000),
                file("mmproj-q-F16.gguf", 500_000_000),
            ],
        }],
        fail,
        ..FakeHub::default()
    }
}

fn app(search: &str, hub: Option<FakeHub>) -> Router {
    let supervisor = Supervisor::spawn(
        vec![],
        Arc::new(FakeBackend::new(Default::default())),
        Arc::new(TestClock),
        Settings {
            budget: 10_000_000_000,
            ram_budget: 0,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    );
    let backend: Arc<dyn Backend> = Arc::new(FakeBackend::new(Default::default()));
    let mut app = App::new(
        config(search),
        supervisor,
        backend,
        Arc::new(eliezer_gpu::sources(
            eliezer_core::config::GpuVendor::None,
            eliezer_core::config::ProbeMethod::None,
            0,
            8_000_000_000,
            "/nonexistent".into(),
        )),
        vec![],
    );
    if let Some(hub) = hub {
        app = app.with_hub(Arc::new(hub));
    }
    managed_router(app)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn search_lists_candidates_without_opening_them() {
    let app = app("", Some(hub(None)));
    let (status, body) = get(
        &app,
        "/api/v1/hub/search?q=qwen&task=llm&license=apache-2.0",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["repo"], "org/Qwen-GGUF");
    assert_eq!(data[0]["license"], "apache-2.0");
    assert!(data[0]["detail"].is_null());
    // 12 GB card minus 2 GB reserve; 40 GB RAM minus 4 GB reserve
    assert_eq!(body["budgets"]["gpu_bytes"], 10_000_000_000_u64);
    assert_eq!(body["budgets"]["ram_bytes"], 36_000_000_000_u64);
}

#[tokio::test]
async fn search_with_details_judges_support_and_fit() {
    let app = app("", Some(hub(None)));
    let (_, body) = get(&app, "/api/v1/hub/search?q=qwen&details=true").await;
    let detail = &body["data"][0]["detail"];
    assert_eq!(detail["support"]["level"], "verified");
    assert_eq!(detail["support"]["backend"], "llamacpp");
    let variants = detail["variants"].as_array().unwrap();
    let fit = |name: &str| {
        variants
            .iter()
            .find(|v| v["file"] == name)
            .unwrap_or_else(|| panic!("{name}"))["fit"]
            .clone()
    };
    // 4 GB + 20 % + 0.5 GB projector = 5.3 GB on the GPU; 20 GB + 20 % + 0.5 GB only in RAM
    assert_eq!(fit("q-Q4_K_M.gguf"), "gpu");
    assert_eq!(fit("q-F16.gguf"), "cpu");
    assert_eq!(detail["projector"]["file"], "mmproj-q-F16.gguf");
    assert_eq!(detail["sha"], "abc");
}

#[tokio::test]
async fn one_repository_can_be_inspected() {
    let app = app("", Some(hub(None)));
    let (status, body) = get(&app, "/api/v1/hub/models/org/Qwen-GGUF").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["architecture"], "qwen3moe");
    assert_eq!(body["variants"].as_array().unwrap().len(), 2);
    let (status, body) = get(&app, "/api/v1/hub/models/org/Missing").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found");
}

#[tokio::test]
async fn bad_input_is_refused_before_anything_is_asked() {
    let app = app("", Some(hub(None)));
    for uri in [
        "/api/v1/hub/search",
        "/api/v1/hub/search?q=x&task=music",
        "/api/v1/hub/search?q=x&format=exe",
        "/api/v1/hub/search?q=x&limit=0",
        "/api/v1/hub/search?q=x&limit=21",
        "/api/v1/hub/search?q=x&limit=abc",
        "/api/v1/hub/search?q=x&details=yes",
        "/api/v1/hub/search?q=x&license=a%2Fb",
        "/api/v1/hub/search?q=a%00b",
        "/api/v1/hub/models/org/..%2Fx",
    ] {
        let (status, body) = get(&app, uri).await;
        assert!(
            status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
            "{uri}: {status} {body}"
        );
    }
    let long = "x".repeat(101);
    let (status, _) = get(&app, &format!("/api/v1/hub/search?q={long}")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn the_search_can_be_switched_off_and_failures_are_named() {
    let off = app("search = false", Some(hub(None)));
    let (status, body) = get(&off, "/api/v1/hub/search?q=x").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["type"], "hub_disabled");
    let none = app("", None);
    assert_eq!(
        get(&none, "/api/v1/hub/models/org/Qwen-GGUF").await.0,
        StatusCode::FORBIDDEN
    );
    for (error, status, code) in [
        (
            HubError::Unreachable,
            StatusCode::BAD_GATEWAY,
            "hub_unreachable",
        ),
        (
            HubError::BadResponse,
            StatusCode::BAD_GATEWAY,
            "hub_bad_response",
        ),
        (
            HubError::Busy,
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded",
        ),
    ] {
        let app = app("", Some(hub(Some(error))));
        let (got, body) = get(&app, "/api/v1/hub/search?q=x").await;
        assert_eq!(got, status);
        assert_eq!(body["error"]["type"], code);
    }
}

//! Contract test (specification 11): the model list of LM Studio 0.4 and that of Eliezer
//! go through the same checks. Missing fields are allowed, wrong types are not.
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use eliezer_api::{App, managed_router};
use eliezer_backend::{supervisor::*, *};
use eliezer_core::{
    config::Config,
    scheduler::{Clock, Instance, State},
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

/// `null` counts as missing: LM Studio returns it for `params_string` of embedding models, for example.
fn check_optional(value: &Value, key: &str, ok: fn(&Value) -> bool, what: &str, path: &str) {
    if let Some(v) = value.get(key).filter(|v| !v.is_null()) {
        assert!(ok(v), "{path}.{key} must be {what}, but is {v}");
    }
}

/// Fields and types from specification 5.2, as MultiGPT reads them.
fn check_models(body: &Value) -> usize {
    let models = body["models"].as_array().expect("`models` must be a list");
    for (i, m) in models.iter().enumerate() {
        let p = format!("models[{i}]");
        let kind = m["type"].as_str().expect("`type` is mandatory");
        assert!(
            ["llm", "embedding", "image"].contains(&kind),
            "{p}.type unknown: {kind}"
        );
        assert!(
            m["key"].as_str().is_some_and(|k| !k.is_empty()),
            "{p}.key is mandatory"
        );
        for key in ["display_name", "architecture", "params_string"] {
            check_optional(m, key, Value::is_string, "Text", &p);
        }
        for key in ["size_bytes", "max_context_length"] {
            check_optional(m, key, Value::is_u64, "integer", &p);
        }
        if let Some(q) = m.get("quantization") {
            check_optional(
                q,
                "bits_per_weight",
                Value::is_number,
                "Zahl",
                &format!("{p}.quantization"),
            );
        }
        if let Some(c) = m.get("capabilities") {
            let cp = format!("{p}.capabilities");
            check_optional(c, "vision", Value::is_boolean, "Wahrheitswert", &cp);
            if let Some(r) = c.get("reasoning") {
                let rp = format!("{cp}.reasoning");
                check_optional(r, "default", Value::is_string, "Text", &rp);
                check_optional(
                    r,
                    "allowed_options",
                    |v| v.as_array().is_some_and(|a| a.iter().all(Value::is_string)),
                    "list of strings",
                    &rp,
                );
            }
        }
        for (j, inst) in m["loaded_instances"]
            .as_array()
            .expect("loaded_instances is mandatory")
            .iter()
            .enumerate()
        {
            let ip = format!("{p}.loaded_instances[{j}]");
            assert!(inst["id"].is_string(), "{ip}.id is mandatory");
            if let Some(c) = inst.get("config") {
                for key in ["context_length", "parallel"] {
                    check_optional(c, key, Value::is_u64, "integer", &format!("{ip}.config"));
                }
            }
        }
    }
    models.len()
}

#[test]
fn lmstudio_reference_passes() {
    let body: Value = serde_json::from_str(include_str!("lmstudio_models.json")).unwrap();
    assert!(check_models(&body) > 0);
}

#[test]
fn checks_reject_wrong_types() {
    let bad =
        json!({"models":[{"type":"llm","key":"a","size_bytes":"large","loaded_instances":[]}]});
    assert!(std::panic::catch_unwind(|| check_models(&bad)).is_err());
}

#[tokio::test]
async fn eliezer_passes_same_checks() {
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
    let backend: Arc<dyn Backend> = Arc::new(FakeBackend::new(Default::default()));
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
            },
            max_context: m.max_context,
        }],
        backend.clone(),
        Arc::new(TestClock),
        Settings {
            budget: 7_000_000_000,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    );
    let router = managed_router(App::new(
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
        vec![],
    ));
    let get = |router: axum::Router, path: &'static str, body: Option<Value>| async move {
        let request = Request::builder()
            .uri(path)
            .method(if body.is_some() { "POST" } else { "GET" })
            .header("content-type", "application/json")
            .body(body.map_or_else(Body::empty, |v| Body::from(v.to_string())))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        serde_json::from_slice::<Value>(&to_bytes(response.into_body(), 1 << 20).await.unwrap())
            .unwrap()
    };
    let before = get(router.clone(), "/api/v1/models", None).await;
    assert_eq!(check_models(&before), 1);
    get(
        router.clone(),
        "/api/v1/models/load",
        Some(json!({"model":"a","context_length":4096})),
    )
    .await;
    let after = get(router, "/api/v1/models", None).await;
    assert_eq!(check_models(&after), 1);
    let instances = after["models"][0]["loaded_instances"].as_array().unwrap();
    assert_eq!(instances.len(), 1, "an instance must appear after loading");
    assert_eq!(instances[0]["id"], "a");
    supervisor.shutdown().await;
}

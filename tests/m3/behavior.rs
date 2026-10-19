//! Integration tests from specification 11 at HTTP level, with fake-backend, without network and GPU.
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{HeaderMap, Request, StatusCode},
};
use eliezer_api::{App, managed_router};
use eliezer_backend::{supervisor::*, *};
use eliezer_core::{
    config::{Config, GpuVendor, ProbeMethod},
    scheduler::{Clock, Instance, State},
};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tower::ServiceExt;

#[derive(Default)]
struct TestClock(AtomicU64);
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}
struct Env {
    router: Router,
    sup: Supervisor,
    clock: Arc<TestClock>,
    fake: Arc<FakeBackend>,
}
/// Models: (key, VRAM in GB, pinned), all language models. The budget is 7 GB.
fn env(models: &[(&str, f64, bool)], fake: FakeConfig, keys: &[&str]) -> Env {
    let all: Vec<_> = models.iter().map(|(k, g, p)| (*k, *g, *p, "llm")).collect();
    env_kinds(&all, fake, keys)
}
/// Like `env`, with the model kind as fourth element ("llm", "embedding" or "image").
fn env_kinds(models: &[(&str, f64, bool, &str)], fake: FakeConfig, keys: &[&str]) -> Env {
    let mut toml = String::from(
        "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\nmeasure=\"none\"\nvram_budget_gb=8.0\nreserve_gb=1.0\n[backend.mistralrs]\nenabled=true\n[backend.sdcpp]\nprogram=\"/usr/local/bin/sd-server\"\nport_range=[24600,24609]\n",
    );
    for (key, gb, pinned, kind) in models {
        let reasoning = if *kind == "llm" {
            "reasoning={options=[\"none\",\"low\",\"medium\",\"high\"],default=\"none\"}\n"
        } else {
            ""
        };
        let backend = if *kind == "image" {
            "sdcpp"
        } else {
            "mistralrs"
        };
        toml += &format!(
            "[[model]]\nkey=\"{key}\"\nbackend=\"{backend}\"\nsource={{hf=\"t/{key}\"}}\nkind=\"{kind}\"\nvram_gb={gb}\npinned={pinned}\n{reasoning}"
        );
    }
    let config = Arc::new(Config::parse(&toml).unwrap());
    let fake = Arc::new(FakeBackend::new(fake));
    let clock = Arc::new(TestClock::default());
    let list = config
        .model
        .iter()
        .map(|m| {
            let bytes = (m.vram_gb.unwrap() * 1e9) as u64;
            Model {
                instance: Instance {
                    key: m.key.clone(),
                    state: State::Unloaded,
                    bytes,
                    priority: 0,
                    pinned: m.pinned,
                    image: m.kind == eliezer_core::config::ModelKind::Image,
                    in_flight: 0,
                    last_used_ms: 0,
                },
                default_spec: LoadSpec {
                    model: m.key.clone(),
                    context: m.default_context,
                    parallel: 1,
                    bytes,
                },
                max_context: m.max_context,
            }
        })
        .collect();
    let sup = Supervisor::spawn(
        list,
        fake.clone(),
        clock.clone(),
        Settings {
            budget: 7_000_000_000,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    );
    let probe = Arc::new(eliezer_gpu::sources(
        GpuVendor::None,
        ProbeMethod::None,
        0,
        8_000_000_000,
        "/nonexistent".into(),
    ));
    let router = managed_router(App::new(
        config,
        sup.clone(),
        fake.clone(),
        probe,
        keys.iter().map(|k| k.to_string()).collect(),
    ));
    Env {
        router,
        sup,
        clock,
        fake,
    }
}
fn request(path: &str, body: Option<Value>, headers: &[(&str, &str)]) -> Request<Body> {
    let mut b = Request::builder()
        .uri(path)
        .method(if body.is_some() { "POST" } else { "GET" })
        .header("content-type", "application/json");
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    b.body(body.map_or_else(Body::empty, |v| Body::from(v.to_string())))
        .unwrap()
}
async fn send(
    router: &Router,
    path: &str,
    body: Option<Value>,
    headers: &[(&str, &str)],
) -> (StatusCode, HeaderMap, String) {
    let response = router
        .clone()
        .oneshot(request(path, body, headers))
        .await
        .unwrap();
    let (status, map) = (response.status(), response.headers().clone());
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, map, String::from_utf8(bytes.to_vec()).unwrap())
}
async fn call(router: &Router, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let (status, _, text) = send(router, path, body, &[]).await;
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}
fn chat(model: &str) -> Value {
    json!({"model":model,"messages":[{"role":"user","content":"Hello"}]})
}
async fn state_of(e: &Env, key: &str) -> State {
    e.sup
        .snapshot()
        .await
        .unwrap()
        .iter()
        .find(|i| i.key == key)
        .unwrap()
        .state
}
async fn wait_state(e: &Env, key: &str, state: State) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while state_of(e, key).await != state {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{key} did not reach {state:?}"));
}

#[tokio::test]
async fn load_reports_config_and_reload_with_other_context() {
    let e = env(&[("a", 1.0, false)], Default::default(), &[]);
    let (status, load) = call(
        &e.router,
        "/api/v1/models/load",
        Some(json!({"model":"a","context_length":4096,"parallel":2,"echo_load_config":true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(load["status"], "loaded");
    assert_eq!(load["instance_id"], "a");
    assert!(load["load_time_seconds"].is_number());
    assert_eq!(
        load["load_config"],
        json!({"context_length":4096,"parallel":2})
    );
    let (_, models) = call(&e.router, "/api/v1/models", None).await;
    assert_eq!(
        models["models"][0]["loaded_instances"][0]["config"],
        json!({"context_length":4096,"parallel":2}),
        "the model list must show the configuration actually loaded"
    );
    let starts = e.fake.starts();
    let (status, load) = call(
        &e.router,
        "/api/v1/models/load",
        Some(json!({"model":"a","context_length":8192})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(load["load_config"]["context_length"], 8192);
    assert_eq!(
        e.fake.starts(),
        starts + 1,
        "a different context length reloads"
    );
    let (_, models) = call(&e.router, "/api/v1/models", None).await;
    assert_eq!(
        models["models"][0]["loaded_instances"][0]["config"]["context_length"],
        8192
    );
    let (status, _) = call(
        &e.router,
        "/api/v1/models/load",
        Some(json!({"model":"a","context_length":1_000_000})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "context above max_context");
    e.sup.shutdown().await;
}

#[tokio::test]
async fn insufficient_memory_names_needed_and_free() {
    let e = env(&[("big", 9.0, false)], Default::default(), &[]);
    let (status, body) = call(
        &e.router,
        "/api/v1/models/load",
        Some(json!({"model":"big"})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["type"], "insufficient_memory");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("needs 9.0") && message.contains("free 7.0"),
        "message must state \"needs X GB, free Y GB\": {message}"
    );
    e.sup.shutdown().await;
}

#[tokio::test]
async fn eviction_and_pinning() {
    let e = env(
        &[("a", 4.0, false), ("b", 4.0, false), ("p", 4.0, true)],
        Default::default(),
        &[],
    );
    let load = |k: &'static str| call(&e.router, "/api/v1/models/load", Some(json!({"model":k})));
    assert_eq!(load("a").await.0, StatusCode::OK);
    assert_eq!(load("b").await.0, StatusCode::OK);
    assert_eq!(state_of(&e, "a").await, State::Unloaded, "a was evicted");
    assert_eq!(state_of(&e, "b").await, State::Loaded);
    assert_eq!(load("p").await.0, StatusCode::OK);
    assert_eq!(state_of(&e, "b").await, State::Unloaded);
    let (status, body) = load("a").await;
    assert_eq!(status, StatusCode::CONFLICT, "pinned model stays");
    assert_eq!(body["error"]["type"], "insufficient_memory");
    assert_eq!(state_of(&e, "p").await, State::Loaded);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn chat_to_unloaded_model_loads_and_answers() {
    let e = env(&[("a", 1.0, false)], Default::default(), &[]);
    assert_eq!(state_of(&e, "a").await, State::Unloaded);
    let (status, body) = call(&e.router, "/v1/chat/completions", Some(chat("a"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "test reply");
    assert_eq!(state_of(&e, "a").await, State::Loaded);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn idle_unload_follows_the_test_clock() {
    let e = env(&[("a", 1.0, false)], Default::default(), &[]);
    call(&e.router, "/api/v1/models/load", Some(json!({"model":"a"}))).await;
    e.clock.0.store(999, Ordering::Relaxed);
    e.sup.tick().await.unwrap();
    assert_eq!(state_of(&e, "a").await, State::Loaded, "not idle yet");
    e.clock.0.store(1001, Ordering::Relaxed);
    e.sup.tick().await.unwrap();
    wait_state(&e, "a", State::Unloaded).await;
    e.sup.shutdown().await;
}

#[tokio::test]
async fn wait_timeout_is_503_for_openai_and_529_for_anthropic() {
    // a (5 GB) is busy, b (5 GB) does not fit beside it and must wait.
    let e = env(
        &[("a", 5.0, false), ("b", 5.0, false)],
        FakeConfig {
            chunk_delay: Duration::from_secs(30),
            ..Default::default()
        },
        &[],
    );
    let held = e
        .router
        .clone()
        .oneshot(request(
            "/v1/chat/completions",
            Some(json!({"model":"a","messages":[],"stream":true})),
            &[],
        ))
        .await
        .unwrap();
    assert_eq!(held.status(), StatusCode::OK);
    for (path, body, expect, kind) in [
        ("/v1/chat/completions", chat("b"), 503, "overloaded"),
        (
            "/v1/messages",
            json!({"model":"b","messages":[],"max_tokens":5}),
            529,
            "overloaded_error",
        ),
    ] {
        let router = e.router.clone();
        let waiting = tokio::spawn(async move { call(&router, path, Some(body)).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        e.clock.0.fetch_add(200, Ordering::Relaxed);
        e.sup.tick().await.unwrap();
        let (status, body) = tokio::time::timeout(Duration::from_secs(2), waiting)
            .await
            .expect("wait-time overflow must respond")
            .unwrap();
        assert_eq!(status.as_u16(), expect);
        assert_eq!(body["error"]["type"], kind);
    }
    drop(held);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn backend_crash_is_502_and_next_request_reloads() {
    let e = env(
        &[("a", 1.0, false)],
        FakeConfig {
            crash_after: Some(0),
            ..Default::default()
        },
        &[],
    );
    let (status, body) = call(&e.router, "/v1/chat/completions", Some(chat("a"))).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["type"], "backend_error");
    assert_eq!(e.fake.starts(), 1);
    wait_state(&e, "a", State::Failed).await;
    e.clock.0.fetch_add(31, Ordering::Relaxed);
    let (status, _) = call(&e.router, "/v1/chat/completions", Some(chat("a"))).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(e.fake.starts(), 2, "the next request reloads");
    e.sup.shutdown().await;
}

#[tokio::test]
async fn reasoning_effort_default_and_validation() {
    let e = env(&[("a", 1.0, false)], Default::default(), &[]);
    for ok in ["none", "low", "medium", "high"] {
        let mut body = chat("a");
        body["reasoning_effort"] = json!(ok);
        assert_eq!(
            call(&e.router, "/v1/chat/completions", Some(body)).await.0,
            StatusCode::OK
        );
    }
    assert_eq!(
        call(&e.router, "/v1/chat/completions", Some(chat("a")))
            .await
            .0,
        StatusCode::OK
    );
    for bad in ["xhigh", "minimal", "bogus", ""] {
        let mut body = chat("a");
        body["reasoning_effort"] = json!(bad);
        let (status, err) = call(&e.router, "/v1/chat/completions", Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
        assert!(err["error"]["message"].is_string());
    }
    e.sup.shutdown().await;
}

#[tokio::test]
async fn usage_block_only_with_include_usage() {
    let e = env(&[("a", 1.0, false)], Default::default(), &[]);
    let stream = |usage: bool| {
        let mut body = chat("a");
        body["stream"] = json!(true);
        if usage {
            body["stream_options"] = json!({"include_usage":true});
        }
        let router = e.router.clone();
        async move { send(&router, "/v1/chat/completions", Some(body), &[]).await }
    };
    let (_, headers, with) = stream(true).await;
    assert_eq!(headers["content-type"], "text/event-stream");
    let last: Value = serde_json::from_str(
        with.lines()
            .rfind(|l| l.starts_with("data: {"))
            .unwrap()
            .trim_start_matches("data: "),
    )
    .unwrap();
    assert_eq!(last["usage"]["prompt_tokens"], 1);
    assert_eq!(last["usage"]["completion_tokens"], 1);
    assert!(last["eliezer"]["ttft_ms"].is_number() && last["eliezer"]["tokens"] == 1);
    assert!(with.trim_end().ends_with("data: [DONE]"));
    let (_, _, without) = stream(false).await;
    assert!(
        !without.contains("\"usage\""),
        "no usage block without include_usage"
    );
    e.sup.shutdown().await;
}

#[tokio::test]
async fn anthropic_authentication_models_and_errors() {
    let e = env(&[("a", 1.0, false)], Default::default(), &["secret-1"]);
    let body = json!({"model":"a","messages":[{"role":"user","content":"Hi"}],"max_tokens":5});
    let (status, _, text) = send(&e.router, "/v1/messages", Some(body.clone()), &[]).await;
    let err: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(err["type"], "error");
    assert_eq!(err["error"]["type"], "authentication_error");
    assert!(!text.contains("secret-1"));
    let (status, _, _) = send(
        &e.router,
        "/v1/messages",
        Some(body.clone()),
        &[("x-api-key", "falsch")],
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // x-api-key and Bearer, with and without anthropic-version
    for headers in [
        vec![("x-api-key", "secret-1")],
        vec![("authorization", "Bearer secret-1")],
        vec![
            ("x-api-key", "secret-1"),
            ("anthropic-version", "2023-06-01"),
        ],
    ] {
        let (status, _, text) = send(&e.router, "/v1/messages", Some(body.clone()), &headers).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let msg: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            (msg["type"].as_str(), msg["role"].as_str()),
            (Some("message"), Some("assistant"))
        );
        assert_eq!(msg["content"][0]["text"], "test reply");
        assert_eq!(msg["stop_reason"], "end_turn");
        assert_eq!(msg["usage"]["input_tokens"], 1);
    }
    let auth = [
        ("x-api-key", "secret-1"),
        ("anthropic-version", "2023-06-01"),
    ];
    let (_, _, text) = send(&e.router, "/v1/models", None, &auth).await;
    let list: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(list["has_more"], false);
    assert_eq!(list["data"][0]["type"], "model");
    assert_eq!(list["data"][0]["id"], "a");
    let (_, _, text) = send(
        &e.router,
        "/v1/models",
        None,
        &[("authorization", "Bearer secret-1")],
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap()["object"],
        "list"
    );
    // Error cases in Anthropic format
    let cases = [
        (
            json!({"model":"a","messages":[],"max_tokens":5,"messages":[{"role":"user","content":[{"type":"document","source":{}}]}]}),
            400,
            "invalid_request_error",
        ),
        (
            json!({"model":"a","messages":[{"role":"user","content":"x"}]}),
            400,
            "invalid_request_error",
        ),
        (
            json!({"model":"gibtsnicht","messages":[],"max_tokens":5}),
            404,
            "not_found_error",
        ),
    ];
    for (body, code, kind) in cases {
        let (status, _, text) = send(&e.router, "/v1/messages", Some(body), &auth).await;
        let err: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            (
                status.as_u16(),
                err["type"].as_str(),
                err["error"]["type"].as_str()
            ),
            (code, Some("error"), Some(kind)),
            "{text}"
        );
        assert!(err["error"]["message"].is_string());
    }
    // unknown fields are ignored
    let mut extra = body.clone();
    extra["new_field"] = json!({"x":1});
    assert_eq!(
        send(&e.router, "/v1/messages", Some(extra), &auth).await.0,
        StatusCode::OK
    );
    // count_tokens: approximation characters / 3.5 with a marker
    let (status, headers, text) = send(
        &e.router,
        "/v1/messages/count_tokens",
        Some(json!({"model":"a","messages":[{"role":"user","content":"Hello world"}]})),
        &auth,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert_eq!(headers["x-eliezer-estimate"], "true");
    assert!(
        serde_json::from_str::<Value>(&text).unwrap()["input_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    e.sup.shutdown().await;
}

#[tokio::test]
async fn anthropic_stream_event_order_matches_expectation_file() {
    let chunks = vec![
        Chunk::Thinking("I am thinking".into()),
        Chunk::Text("Hello".into()),
        Chunk::Text(" world".into()),
        Chunk::Tool {
            index: 0,
            id: Some("toolu_1".into()),
            name: Some("weather".into()),
            arguments: "{\"place\":".into(),
        },
        Chunk::Tool {
            index: 0,
            id: None,
            name: None,
            arguments: "\"Cologne\"}".into(),
        },
        Chunk::Done {
            finish_reason: "tool_calls".into(),
            input_tokens: 7,
            output_tokens: 9,
        },
    ];
    let e = env(
        &[("a", 1.0, false)],
        FakeConfig {
            chunks,
            ..Default::default()
        },
        &[],
    );
    let (status, headers, text) = send(
        &e.router,
        "/v1/messages",
        Some(json!({"model":"a","messages":[{"role":"user","content":"Hi"}],"max_tokens":50,"stream":true})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "text/event-stream");
    let mut order = Vec::new();
    let mut partial = String::new();
    let mut last_delta = Value::Null;
    for frame in text.split("\n\n").filter(|f| !f.is_empty()) {
        let name = frame
            .lines()
            .find_map(|l| l.strip_prefix("event: "))
            .unwrap();
        let data: Value = serde_json::from_str(
            frame
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(data["type"], name, "event name and type match");
        let detail = match name {
            "content_block_start" => format!(
                "{} {}",
                data["index"],
                data["content_block"]["type"].as_str().unwrap()
            ),
            "content_block_delta" => {
                if data["delta"]["type"] == "input_json_delta" {
                    partial.push_str(data["delta"]["partial_json"].as_str().unwrap());
                }
                format!(
                    "{} {}",
                    data["index"],
                    data["delta"]["type"].as_str().unwrap()
                )
            }
            "content_block_stop" => data["index"].to_string(),
            _ => String::new(),
        };
        if name == "message_delta" {
            last_delta = data;
        }
        order.push(format!("{name} {detail}").trim().to_owned());
    }
    let expected = include_str!("expected/anthropic_stream.txt");
    assert_eq!(
        order.join("\n"),
        expected.trim(),
        "event sequence differs from the expectation file"
    );
    assert_eq!(
        serde_json::from_str::<Value>(&partial).unwrap(),
        json!({"place":"Cologne"})
    );
    assert_eq!(last_delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(last_delta["usage"]["output_tokens"], 9);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn completions_endpoint_plain_and_streaming() {
    let e = env(&[("a", 1.0, false)], Default::default(), &[]);
    let (status, body) = call(
        &e.router,
        "/v1/completions",
        Some(json!({"model":"a","prompt":"Once upon"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "test reply");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 2);
    let (status, body) = call(
        &e.router,
        "/v1/completions",
        Some(json!({"model":"a","prompt":["only one"]})),
    )
    .await;
    assert_eq!(
        (status, body["object"].as_str()),
        (StatusCode::OK, Some("text_completion"))
    );
    for bad in [json!(["a", "b"]), json!(5), Value::Null] {
        let (status, err) = call(
            &e.router,
            "/v1/completions",
            Some(json!({"model":"a","prompt":bad})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(err["error"]["message"].is_string());
    }
    let (_, headers, text) = send(
        &e.router,
        "/v1/completions",
        Some(json!({"model":"a","prompt":"Hi","stream":true,"stream_options":{"include_usage":true}})),
        &[],
    )
    .await;
    assert_eq!(headers["content-type"], "text/event-stream");
    let frames: Vec<Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: {"))
        .map(|l| serde_json::from_str(&format!("{{{l}")).unwrap())
        .collect();
    assert!(frames.iter().all(|f| f["object"] == "text_completion"));
    assert_eq!(frames[0]["choices"][0]["text"], "test reply");
    assert_eq!(frames[1]["choices"][0]["finish_reason"], "stop");
    assert_eq!(frames.last().unwrap()["usage"]["completion_tokens"], 1);
    assert!(text.trim_end().ends_with("data: [DONE]"));
    e.sup.shutdown().await;
}

#[tokio::test]
async fn embeddings_endpoint() {
    let e = env_kinds(
        &[
            ("emb", 1.0, false, "embedding"),
            ("chat", 1.0, false, "llm"),
        ],
        Default::default(),
        &[],
    );
    let (status, body) = call(
        &e.router,
        "/v1/embeddings",
        Some(json!({"model":"emb","input":["hello world","again"]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "emb");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(
        (data[0]["index"].clone(), data[1]["index"].clone()),
        (json!(0), json!(1))
    );
    assert_eq!(data[0]["embedding"].as_array().unwrap().len(), 8);
    assert_ne!(data[0]["embedding"], data[1]["embedding"]);
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    let (_, single) = call(
        &e.router,
        "/v1/embeddings",
        Some(json!({"model":"emb","input":"hello world"})),
    )
    .await;
    assert_eq!(
        single["data"][0]["embedding"], data[0]["embedding"],
        "same text, same vector"
    );
    for (body, expect) in [
        (json!({"model":"chat","input":"x"}), 400),
        (json!({"model":"nope","input":"x"}), 404),
        (json!({"model":"emb","input":[]}), 400),
        (json!({"model":"emb","input":[1,2]}), 400),
        (json!({"model":"emb"}), 400),
    ] {
        let (status, err) = call(&e.router, "/v1/embeddings", Some(body)).await;
        assert_eq!(status.as_u16(), expect, "{err}");
        assert!(err["error"]["message"].is_string());
    }
    // chat on an embedding model is refused as well
    let (status, _) = call(&e.router, "/v1/chat/completions", Some(chat("emb"))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn image_generation_endpoint() {
    let e = env_kinds(
        &[("img", 4.0, false, "image"), ("chat", 1.0, false, "llm")],
        Default::default(),
        &[],
    );
    let (status, body) = call(
        &e.router,
        "/v1/images/generations",
        Some(json!({"prompt":"a cat","n":2,"size":"512x512","response_format":"b64_json"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["created"].as_u64().unwrap() > 1_700_000_000);
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert!(
        data[0]["b64_json"]
            .as_str()
            .unwrap()
            .starts_with("iVBORw0KGgo")
    );
    // with the only image model configured, `model` may be omitted; naming it works too
    let (status, _) = call(
        &e.router,
        "/v1/images/generations",
        Some(json!({"model":"img","prompt":"a dog"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    for (body, expect) in [
        (json!({"prompt":"x","n":5}), 400),
        (json!({"prompt":"x","n":0}), 400),
        (json!({"prompt":"x","response_format":"url"}), 400),
        (json!({"prompt":"x","size":"big"}), 400),
        (json!({"prompt":"  "}), 400),
        (json!({"model":"chat","prompt":"x"}), 400),
        (json!({"model":"nope","prompt":"x"}), 404),
    ] {
        let (status, err) = call(&e.router, "/v1/images/generations", Some(body)).await;
        assert_eq!(status.as_u16(), expect, "{err}");
        assert!(err["error"]["message"].is_string());
    }
    e.sup.shutdown().await;
    // two image models: `model` becomes mandatory
    let e = env_kinds(
        &[("i1", 1.0, false, "image"), ("i2", 1.0, false, "image")],
        Default::default(),
        &[],
    );
    let (status, _) = call(
        &e.router,
        "/v1/images/generations",
        Some(json!({"prompt":"x"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn chat_and_image_models_take_turns() {
    // The chat model (4 GB) and the image model (5 GB) do not fit together in 7 GB.
    let e = env_kinds(
        &[("chat", 4.0, false, "llm"), ("img", 5.0, false, "image")],
        Default::default(),
        &[],
    );
    let ask_chat = || call(&e.router, "/v1/chat/completions", Some(chat("chat")));
    let ask_image = || {
        call(
            &e.router,
            "/v1/images/generations",
            Some(json!({"prompt":"x"})),
        )
    };
    for _ in 0..2 {
        assert_eq!(ask_chat().await.0, StatusCode::OK);
        assert_eq!(state_of(&e, "chat").await, State::Loaded);
        assert_eq!(ask_image().await.0, StatusCode::OK);
        assert_eq!(state_of(&e, "img").await, State::Loaded);
        assert_eq!(
            state_of(&e, "chat").await,
            State::Unloaded,
            "image evicts chat"
        );
    }
    // the chat model reloads on its own with the next request
    assert_eq!(ask_chat().await.0, StatusCode::OK);
    assert_eq!(state_of(&e, "img").await, State::Unloaded);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn status_shows_queue_and_cancelled_waiters_leave_it() {
    let e = env(
        &[("a", 5.0, false), ("b", 5.0, false)],
        FakeConfig {
            chunk_delay: Duration::from_secs(30),
            ..Default::default()
        },
        &[],
    );
    let (_, before) = call(&e.router, "/api/v1/status", None).await;
    assert_eq!(before["queue"], json!([]));
    assert_eq!(
        before["models"][0]["last_used_ms_ago"],
        Value::Null,
        "never used"
    );
    assert_eq!(before["models"][0]["queue_len"], 0);
    let held = e
        .router
        .clone()
        .oneshot(request(
            "/v1/chat/completions",
            Some(json!({"model":"a","messages":[],"stream":true})),
            &[],
        ))
        .await
        .unwrap();
    e.clock.0.store(500, Ordering::Relaxed);
    let router = e.router.clone();
    let waiting =
        tokio::spawn(async move { call(&router, "/v1/chat/completions", Some(chat("b"))).await });
    let status = loop {
        let (_, status) = call(&e.router, "/api/v1/status", None).await;
        if status["queue"].as_array().is_some_and(|q| !q.is_empty()) {
            break status;
        }
        tokio::task::yield_now().await;
    };
    e.clock.0.store(2500, Ordering::Relaxed);
    let (_, status2) = call(&e.router, "/api/v1/status", None).await;
    assert_eq!(status["queue"][0]["model"], "b");
    assert!(status["queue"][0]["waiting_secs"].is_number());
    assert_eq!(status2["queue"][0]["waiting_secs"], 2.0);
    let model = |s: &Value, k: &str| {
        s["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["key"] == k)
            .unwrap()
            .clone()
    };
    assert_eq!(model(&status2, "b")["queue_len"], 1);
    assert_eq!(model(&status2, "a")["queue_len"], 0);
    assert_eq!(model(&status2, "a")["in_flight"], 1);
    assert_eq!(model(&status2, "a")["last_used_ms_ago"], 2500);
    // the client goes away: the waiting request leaves the queue
    waiting.abort();
    let _ = waiting.await;
    e.sup.tick().await.unwrap();
    let (_, after) = call(&e.router, "/api/v1/status", None).await;
    assert_eq!(
        after["queue"],
        json!([]),
        "abandoned request must not stay queued"
    );
    drop(held);
    e.sup.shutdown().await;
}

#[tokio::test]
async fn reconfigure_adds_removes_and_unloads_idle_models() {
    let e = env(
        &[("a", 1.0, false), ("b", 1.0, false)],
        Default::default(),
        &[],
    );
    call(&e.router, "/api/v1/models/load", Some(json!({"model":"a"}))).await;
    let model = |key: &str, gb: f64| {
        let bytes = (gb * 1e9) as u64;
        Model {
            instance: Instance {
                key: key.into(),
                state: State::Unloaded,
                bytes,
                priority: 0,
                pinned: false,
                image: false,
                in_flight: 0,
                last_used_ms: 0,
            },
            default_spec: LoadSpec {
                model: key.into(),
                context: 1024,
                parallel: 1,
                bytes,
            },
            max_context: 4096,
        }
    };
    let settings = Settings {
        budget: 7_000_000_000,
        idle_ms: 1000,
        wait_ms: 100,
        queue_max: 4,
        retry_ms: 30,
    };
    // "a" is removed while loaded, "b" stays, "c" is new
    e.sup
        .reconfigure(vec![model("b", 1.0), model("c", 1.0)], settings)
        .await
        .unwrap();
    e.sup.tick().await.unwrap();
    let keys = |e: &Env| {
        let sup = e.sup.clone();
        async move {
            sup.snapshot()
                .await
                .unwrap()
                .into_iter()
                .map(|i| i.key)
                .collect::<Vec<_>>()
        }
    };
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            e.sup.tick().await.unwrap();
            if !keys(&e).await.contains(&"a".to_string()) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("removed idle model must be unloaded and dropped");
    assert_eq!(keys(&e).await, vec!["b".to_string(), "c".to_string()]);
    let spec = |k: &str| LoadSpec {
        model: k.into(),
        context: 1024,
        parallel: 1,
        bytes: 1_000_000_000,
    };
    assert!(matches!(
        e.sup.load(spec("a")).await,
        Err(Error::UnknownModel)
    ));
    assert!(e.sup.load(spec("c")).await.is_ok());
    e.sup.shutdown().await;
}

#[tokio::test]
async fn every_response_names_the_server() {
    // Clients tell Eliezer from other programs on the port (LM Studio) by this header.
    let e = env(&[("a", 1.0, false)], Default::default(), &["k"]);
    let good = [("authorization", "Bearer k")];
    for (path, headers, status) in [
        ("/healthz", &[][..], 200),
        ("/api/v1/models", &good[..], 200),
        ("/api/v1/models", &[][..], 401),
        ("/no/such/path", &good[..], 404),
        ("/v1/messages", &[("x-api-key", "wrong")][..], 401),
    ] {
        let (code, response_headers, _) = send(&e.router, path, None, headers).await;
        assert_eq!(code.as_u16(), status, "{path}");
        let server = response_headers["server"].to_str().unwrap();
        assert!(server.starts_with("eliezer/"), "{path}: {server}");
    }
    e.sup.shutdown().await;
}

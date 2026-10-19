//! HTTP interfaces of the model supervisor.
pub mod anthropic;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use eliezer_backend::{
    Backend, ChatRequest, Chunk,
    supervisor::{Error, Lease, Supervisor},
};
use eliezer_core::{
    config::{Config, ModelKind},
    memory::estimate,
};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    pin::Pin,
    sync::{Arc, RwLock},
    task::{Context, Poll},
};
use tokio::sync::mpsc;

/// A value that can be replaced while requests are being served (configuration reload).
pub struct Shared<T>(RwLock<Arc<T>>);
impl<T> Shared<T> {
    pub fn new(value: Arc<T>) -> Self {
        Self(RwLock::new(value))
    }
    pub fn get(&self) -> Arc<T> {
        match self.0.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
    pub fn set(&self, value: Arc<T>) {
        match self.0.write() {
            Ok(mut guard) => *guard = value,
            Err(poisoned) => *poisoned.into_inner() = value,
        }
    }
}
#[derive(Clone)]
pub struct App {
    pub config: Arc<Shared<Config>>,
    pub supervisor: Supervisor,
    pub backend: Arc<dyn Backend>,
    pub probe: Arc<eliezer_gpu::Fallback>,
    pub keys: Arc<Shared<Vec<String>>>,
}
impl App {
    pub fn new(
        config: Arc<Config>,
        supervisor: Supervisor,
        backend: Arc<dyn Backend>,
        probe: Arc<eliezer_gpu::Fallback>,
        keys: Vec<String>,
    ) -> Self {
        Self {
            config: Arc::new(Shared::new(config)),
            supervisor,
            backend,
            probe,
            keys: Arc::new(Shared::new(Arc::new(keys))),
        }
    }
    /// Swaps in a reloaded configuration and key list; requests in flight keep the old ones.
    pub fn replace(&self, config: Arc<Config>, keys: Vec<String>) {
        self.config.set(config);
        self.keys.set(Arc::new(keys));
    }
}
pub fn router() -> Router {
    Router::new().route("/healthz", get(|| async { "ok" }))
}
pub fn managed_router(app: App) -> Router {
    router().merge(
        Router::new()
            .route("/api/v1/models", get(models))
            .route("/api/v1/models/load", post(load))
            .route("/api/v1/models/unload", post(unload))
            .route("/api/v1/status", get(status))
            .route("/v1/models", get(list))
            .route("/v1/chat/completions", post(chat))
            .route("/v1/completions", post(completions))
            .route("/v1/embeddings", post(embeddings))
            .route("/v1/images/generations", post(images))
            .route("/v1/messages", post(messages))
            .route("/v1/messages/count_tokens", post(count_tokens))
            .fallback(|| async {
                failure(
                    StatusCode::NOT_FOUND,
                    "not_found",
                    "Endpoint not found",
                    false,
                )
            })
            .layer(middleware::from_fn_with_state(app.clone(), authorize))
            .with_state(app),
    )
}
fn failure(status: StatusCode, code: &str, message: &str, anthropic: bool) -> Response {
    if anthropic {
        let kind = match status.as_u16() {
            400 => "invalid_request_error",
            401 => "authentication_error",
            404 => "not_found_error",
            429 => "rate_limit_error",
            503 | 529 => "overloaded_error",
            _ => "api_error",
        };
        (
            status,
            Json(json!({"type":"error","error":{"type":kind,"message":message}})),
        )
            .into_response()
    } else {
        (
            status,
            Json(json!({"error":{"type":code,"message":message}})),
        )
            .into_response()
    }
}
fn error(error: Error, anth: bool) -> Response {
    let (status, code, message) = match error {
        Error::UnknownModel => (404, "not_found", "Model is not allowed".to_owned()),
        Error::InvalidLoad => (400, "invalid_request", "Invalid load parameters".to_owned()),
        Error::InsufficientMemory { needed, free } => {
            let gb = |bytes: u64| format!("{:.1}", bytes as f64 / 1e9);
            (
                409,
                "insufficient_memory",
                format!(
                    "Not enough memory: needs {} GB, free {} GB",
                    gb(needed),
                    gb(free)
                ),
            )
        }
        Error::QueueFull | Error::Timeout => (
            if anth { 529 } else { 503 },
            "overloaded",
            "Queue is full or wait time exceeded".to_owned(),
        ),
        Error::Busy => (409, "busy", "Model is still in use".to_owned()),
        Error::Backend => (502, "backend_error", "Backend has failed".to_owned()),
        Error::InvalidRequest => (
            400,
            "invalid_request",
            "The request is invalid or does not fit the model's context window".to_owned(),
        ),
        Error::Stopped => (503, "unavailable", "Service is shutting down".to_owned()),
    };
    failure(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        code,
        &message,
        anth,
    )
}
async fn authorize(State(app): State<App>, request: Request, next: Next) -> Response {
    let anth = request.uri().path().starts_with("/v1/messages")
        || request.headers().contains_key("anthropic-version");
    let bearer = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    let api = request
        .headers()
        .get("x-api-key")
        .and_then(|h| h.to_str().ok());
    let keys = app.keys.get();
    if !keys.is_empty()
        && !keys
            .iter()
            .any(|key| bearer == Some(key.as_str()) || (anth && api == Some(key.as_str())))
    {
        return failure(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Authentication required",
            anth,
        );
    }
    next.run(request).await
}
fn json_body(
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
    anth: bool,
) -> Result<Value, Box<Response>> {
    body.map(|Json(v)| v).map_err(|_| {
        Box::new(failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Invalid JSON request body",
            anth,
        ))
    })
}
async fn models(State(app): State<App>) -> Response {
    let states = match app.supervisor.snapshot().await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let specs = match app.supervisor.loaded().await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let config = app.config.get();
    Json(json!({"models":config.model.iter().map(|m| {
        let loaded = states.iter().find(|i| i.key == m.key && i.state == eliezer_core::scheduler::State::Loaded);
        let mut value = json!({"type":match m.kind { eliezer_core::config::ModelKind::Llm => "llm", eliezer_core::config::ModelKind::Embedding => "embedding", _ => "image" },
            "key":m.key,"display_name":m.key,"max_context_length":m.max_context,"capabilities":{"vision":m.vision},"loaded_instances":[]});
        if let Some(reasoning) = &m.reasoning { value["capabilities"]["reasoning"] = json!({"allowed_options":reasoning.options,"default":reasoning.default}); }
        if let Some(size) = m.size_gb.and_then(|gb| eliezer_core::memory::gb_bytes(gb).ok()) { value["size_bytes"] = json!(size); }
        if let (Some(_), Some(spec)) = (loaded, specs.iter().find(|s| s.model == m.key)) { value["loaded_instances"] = json!([{"id":m.key,"config":{"context_length":spec.context,"parallel":spec.parallel}}]); }
        value
    }).collect::<Vec<_>>()})).into_response()
}
async fn list(State(app): State<App>, headers: axum::http::HeaderMap) -> Response {
    let anth = headers.contains_key("anthropic-version");
    let config = app.config.get();
    let data: Vec<_> = config.model.iter().map(|m| if anth { json!({"type":"model","id":m.key,"display_name":m.key,"created_at":"2026-10-19T00:00:00Z"}) }
        else { json!({"id":m.key,"object":"model","owned_by":"eliezer","created":0}) }).collect();
    Json(if anth {
        json!({"data":data,"has_more":false})
    } else {
        json!({"object":"list","data":data})
    })
    .into_response()
}
fn spec(app: &App, body: &Value) -> Result<eliezer_backend::LoadSpec, Error> {
    let key = body["model"].as_str().ok_or(Error::InvalidLoad)?;
    let config = app.config.get();
    let model = config
        .model
        .iter()
        .find(|m| m.key == key)
        .ok_or(Error::UnknownModel)?;
    let number = |name: &str, default| -> Result<u32, Error> {
        match body.get(name) {
            None => Ok(default),
            Some(value) => value
                .as_u64()
                .and_then(|n| n.try_into().ok())
                .ok_or(Error::InvalidLoad),
        }
    };
    let context = number("context_length", model.default_context)?;
    let parallel = number("parallel", model.parallel)?;
    let file_size = model
        .source
        .file
        .as_ref()
        .and_then(|path| std::fs::metadata(path).ok())
        .map(|m| m.len());
    let bytes =
        estimate(model, context, parallel, file_size, None).map_err(|_| Error::InvalidLoad)?;
    Ok(eliezer_backend::LoadSpec {
        model: key.into(),
        context,
        parallel,
        bytes,
    })
}
async fn load(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match json_body(body, false) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let spec = match spec(&app, &body) {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let start = std::time::Instant::now();
    match app.supervisor.load(spec).await {
        Ok(spec) => Json(json!({"type":"llm","instance_id":spec.model,"load_time_seconds":start.elapsed().as_secs_f64(),"status":"loaded","load_config":{"context_length":spec.context,"parallel":spec.parallel}})).into_response(),
        Err(e) => error(e,false),
    }
}
async fn unload(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match json_body(body, false) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let Some(key) = body["instance_id"].as_str() else {
        return error(Error::InvalidLoad, false);
    };
    match tokio::time::timeout(
        std::time::Duration::from_secs(app.config.get().server.request_timeout_secs),
        app.supervisor.unload(key.into()),
    )
    .await
    {
        Ok(Ok(())) => Json(json!({"instance_id":key})).into_response(),
        Ok(Err(e)) => error(e, false),
        Err(_) => error(Error::Busy, false),
    }
}
async fn status(State(app): State<App>) -> Response {
    use eliezer_core::scheduler::State as S;
    let info = match app.supervisor.status().await {
        Ok(info) => info,
        Err(e) => return error(e, false),
    };
    let used = info
        .instances
        .iter()
        .filter(|i| matches!(i.state, S::Loaded | S::Loading | S::Unloading))
        .fold(0_u64, |sum, i| sum.saturating_add(i.bytes));
    let budget = eliezer_core::memory::gb_bytes(app.config.get().gpu.vram_budget_gb).unwrap_or(0);
    // The measurement may start processes (nvidia-smi), so do not block in the async context.
    let probe = app.probe.clone();
    let (memory, probe_name) = tokio::task::spawn_blocking(move || {
        let memory = eliezer_gpu::GpuProbe::memory(&*probe).ok();
        (memory, probe.active_name())
    })
    .await
    .unwrap_or((None, "none".to_owned()));
    // Without a real measurement (method "none") the supervisor's bookkeeping counts.
    let (total, used) = match memory {
        Some(m) if probe_name != "none" => (m.total_bytes, m.used_bytes),
        _ => (budget, used),
    };
    let models: Vec<Value> = info
        .instances
        .iter()
        .map(|i| {
            let in_use = matches!(i.state, S::Loaded | S::Loading | S::Unloading);
            json!({
                "key": i.key,
                "state": format!("{:?}", i.state).to_lowercase(),
                "vram_bytes": i.bytes,
                "last_used_ms_ago": in_use.then(|| info.now_ms.saturating_sub(i.last_used_ms)),
                "in_flight": i.in_flight,
                "queue_len": info.queue.iter().filter(|(model, _)| *model == i.key).count(),
            })
        })
        .collect();
    let queue: Vec<Value> = info
        .queue
        .iter()
        .map(|(model, enqueued)| {
            json!({"model": model, "waiting_secs": info.now_ms.saturating_sub(*enqueued) as f64 / 1000.0})
        })
        .collect();
    Json(json!({
        "gpu": {"probe": probe_name, "total_bytes": total, "used_bytes": used,
                "free_bytes": total.saturating_sub(used), "budget_bytes": budget},
        "models": models,
        "queue": queue,
    }))
    .into_response()
}
async fn completions(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let mut body = match json_body(body, false) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let prompt = match &body["prompt"] {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) if items.len() == 1 => items[0].as_str().map(str::to_owned),
        _ => None,
    };
    let Some(prompt) = prompt else {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "prompt must be a string or a list with exactly one string",
            false,
        );
    };
    body["messages"] = json!([{"role": "user", "content": prompt}]);
    complete(app, body, false, true).await
}
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
async fn embeddings(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match json_body(body, false) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let config = app.config.get();
    let Some(model) = config
        .model
        .iter()
        .find(|m| Some(m.key.as_str()) == body["model"].as_str())
    else {
        return error(Error::UnknownModel, false);
    };
    if model.kind != ModelKind::Embedding {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Model is not an embedding model",
            false,
        );
    }
    let input: Option<Vec<String>> = match &body["input"] {
        Value::String(text) => Some(vec![text.clone()]),
        Value::Array(items) if !items.is_empty() => items
            .iter()
            .map(|item| item.as_str().map(str::to_owned))
            .collect(),
        _ => None,
    };
    let Some(input) = input else {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "input must be a string or a non-empty list of strings",
            false,
        );
    };
    let spec = match spec(&app, &json!({"model": model.key})) {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let lease = match app.supervisor.acquire(spec.clone()).await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let request = eliezer_backend::EmbedRequest {
        model: spec.model.clone(),
        input,
    };
    match app
        .supervisor
        .embed(&lease, request, app.backend.as_ref())
        .await
    {
        Ok(result) => Json(json!({
            "object": "list",
            "model": spec.model,
            "data": result.vectors.iter().enumerate()
                .map(|(index, vector)| json!({"object": "embedding", "index": index, "embedding": vector}))
                .collect::<Vec<_>>(),
            "usage": {"prompt_tokens": result.input_tokens, "total_tokens": result.input_tokens},
        }))
        .into_response(),
        Err(e) => error(e, false),
    }
}
fn valid_size(size: &str) -> bool {
    size.split_once('x').is_some_and(|(w, h)| {
        [w, h]
            .iter()
            .all(|part| (2..=4).contains(&part.len()) && part.bytes().all(|b| b.is_ascii_digit()))
    })
}
async fn images(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match json_body(body, false) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let bad = |message: &str| failure(StatusCode::BAD_REQUEST, "invalid_request", message, false);
    let config = app.config.get();
    let model = match body["model"].as_str() {
        Some(key) => match config.model.iter().find(|m| m.key == key) {
            Some(model) => model,
            None => return error(Error::UnknownModel, false),
        },
        None => {
            let mut images = config.model.iter().filter(|m| m.kind == ModelKind::Image);
            match (images.next(), images.next()) {
                (Some(only), None) => only,
                _ => return bad("model is required when there is not exactly one image model"),
            }
        }
    };
    if model.kind != ModelKind::Image {
        return bad("Model is not an image model");
    }
    let Some(prompt) = body["prompt"].as_str().filter(|p| !p.trim().is_empty()) else {
        return bad("prompt must be a non-empty string");
    };
    let count = match body.get("n") {
        None => 1,
        Some(n) => match n.as_u64().filter(|n| (1..=4).contains(n)) {
            Some(n) => n as u32,
            None => return bad("n must be between 1 and 4"),
        },
    };
    if body.get("response_format").is_some_and(|f| f != "b64_json") {
        return bad("Only response_format b64_json is supported");
    }
    let size = match body.get("size") {
        None => None,
        Some(Value::String(size)) if valid_size(size) => Some(size.clone()),
        Some(_) => return bad("size must look like 512x512"),
    };
    let spec = match spec(&app, &json!({"model": model.key})) {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let lease = match app.supervisor.acquire(spec.clone()).await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let request = eliezer_backend::ImageRequest {
        model: spec.model,
        prompt: prompt.to_owned(),
        count,
        size,
    };
    match app
        .supervisor
        .image(&lease, request, app.backend.as_ref())
        .await
    {
        Ok(images) => Json(json!({
            "created": unix_now(),
            "data": images.iter().map(|b64| json!({"b64_json": b64})).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => error(e, false),
    }
}
async fn chat(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match json_body(body, false) {
        Ok(v) => complete(app, v, false, false).await,
        Err(e) => *e,
    }
}
async fn messages(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match json_body(body, true) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let config = app.config.get();
    let Some(model) = config
        .model
        .iter()
        .find(|m| Some(m.key.as_str()) == body["model"].as_str())
    else {
        return error(Error::UnknownModel, true);
    };
    let options = model
        .reasoning
        .as_ref()
        .map(|r| r.options.as_slice())
        .unwrap_or(&[]);
    match anthropic::request(&body, options) {
        Ok(v) => complete(app, v, true, false).await,
        Err(message) => failure(StatusCode::BAD_REQUEST, "invalid_request", message, true),
    }
}
async fn count_tokens(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match json_body(body, true) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let config = app.config.get();
    let Some(model) = config
        .model
        .iter()
        .find(|m| Some(m.key.as_str()) == body["model"].as_str())
    else {
        return error(Error::UnknownModel, true);
    };
    // count_tokens, unlike messages, does not need max_tokens.
    let mut input = body;
    input["max_tokens"] = json!(1);
    let translated = match anthropic::request(
        &input,
        model
            .reasoning
            .as_ref()
            .map(|r| r.options.as_slice())
            .unwrap_or(&[]),
    ) {
        Ok(v) => v,
        Err(message) => return failure(StatusCode::BAD_REQUEST, "invalid_request", message, true),
    };
    let chars = translated["messages"].to_string().chars().count() as u64;
    (
        [("x-eliezer-estimate", "true")],
        Json(json!({"input_tokens":(chars * 2).div_ceil(7)})),
    )
        .into_response()
}
async fn complete(app: App, mut body: Value, anth: bool, completion: bool) -> Response {
    let spec = match spec(&app, &body) {
        Ok(v) => v,
        Err(e) => return error(e, anth),
    };
    let config = app.config.get();
    let Some(model) = config.model.iter().find(|m| m.key == spec.model) else {
        return error(Error::UnknownModel, anth);
    };
    if model.kind != ModelKind::Llm {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Model does not support chat",
            anth,
        );
    }
    if !body["messages"].is_array() {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Nachrichten fehlen",
            anth,
        );
    }
    if let Some(reasoning) = &model.reasoning {
        let effort = body["reasoning_effort"]
            .as_str()
            .unwrap_or(&reasoning.default);
        if !reasoning.options.iter().any(|s| s == effort) {
            return failure(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Reasoning option is not supported by the model",
                anth,
            );
        }
        body["reasoning_effort"] = json!(effort);
    } else if body.get("reasoning_effort").is_some() {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Model does not support reasoning options",
            anth,
        );
    }
    let lease = match app.supervisor.acquire(spec.clone()).await {
        Ok(v) => v,
        Err(e) => return error(e, anth),
    };
    let request = ChatRequest {
        model: spec.model.clone(),
        input: body.to_string(),
        reasoning: body["reasoning_effort"].as_str().map(str::to_owned),
    };
    let mut chunks = match app
        .supervisor
        .chat(&lease, request, app.backend.as_ref())
        .await
    {
        Ok(v) => v,
        Err(e) => return error(e, anth),
    };
    if body["stream"] == true {
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(stream_response(
            app,
            lease,
            chunks,
            tx,
            spec.model,
            StreamOptions {
                anth,
                usage: body["stream_options"]["include_usage"] == true,
                completion,
            },
        ));
        return (
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            Body::from_stream(FrameStream(rx)),
        )
            .into_response();
    }
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tools = std::collections::BTreeMap::new();
    let mut done = None;
    while let Some(chunk) = chunks.recv().await {
        match chunk {
            Ok(Chunk::Text(value)) => text.push_str(&value),
            Ok(Chunk::Thinking(value)) => thinking.push_str(&value),
            Ok(Chunk::Tool {
                index,
                id,
                name,
                arguments,
            }) => {
                let entry = tools
                    .entry(index)
                    .or_insert_with(|| (String::new(), String::new(), String::new()));
                if let Some(id) = id {
                    entry.0 = id;
                }
                if let Some(name) = name {
                    entry.1 = name;
                }
                entry.2.push_str(&arguments);
            }
            Ok(Chunk::Done {
                finish_reason,
                input_tokens,
                output_tokens,
            }) => {
                done = Some((finish_reason, input_tokens, output_tokens));
                break;
            }
            Err(_) => {
                app.supervisor.report_crash(spec.model).await;
                return error(Error::Backend, anth);
            }
        }
    }
    let Some(done) = done else {
        return error(Error::Backend, anth);
    };
    let (finish, input, output) = done.clone();
    if completion {
        return Json(json!({"id":"cmpl_eliezer","object":"text_completion","created":unix_now(),"model":spec.model,
            "choices":[{"index":0,"text":text,"finish_reason":done.0}],
            "usage":{"prompt_tokens":done.1,"completion_tokens":done.2,"total_tokens":done.1+done.2}}))
        .into_response();
    }
    let mut message = json!({"role":"assistant","content":text});
    if !thinking.is_empty() {
        message["reasoning_content"] = json!(thinking);
    }
    if !tools.is_empty() {
        message["tool_calls"]=json!(tools.into_values().map(|(id,name,args)|json!({"id":id,"type":"function","function":{"name":name,"arguments":args}})).collect::<Vec<_>>());
    }
    let result = json!({"id":"msg_eliezer","object":"chat.completion","created":0,"model":spec.model,"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":{"prompt_tokens":input,"completion_tokens":output,"total_tokens":input+output}});
    if anth {
        match anthropic::response(&result) {
            Ok(v) => Json(v).into_response(),
            Err(_) => error(Error::Backend, true),
        }
    } else {
        Json(result).into_response()
    }
}
struct FrameStream(mpsc::Receiver<String>);
impl futures_core::Stream for FrameStream {
    type Item = Result<Bytes, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.poll_recv(cx).map(|v| v.map(|s| Ok(Bytes::from(s))))
    }
}
async fn send(tx: &mpsc::Sender<String>, event: Option<&str>, data: Value) -> bool {
    let prefix = event.map(|e| format!("event: {e}\n")).unwrap_or_default();
    tx.send(format!("{prefix}data: {data}\n\n")).await.is_ok()
}
/// Which dialect a stream speaks and whether it ends with a usage block.
struct StreamOptions {
    anth: bool,
    usage: bool,
    completion: bool,
}
async fn stream_response(
    app: App,
    lease: Lease,
    mut chunks: mpsc::Receiver<Result<Chunk, eliezer_backend::BackendError>>,
    tx: mpsc::Sender<String>,
    model: String,
    options: StreamOptions,
) {
    let StreamOptions {
        anth,
        usage,
        completion,
    } = options;
    let _lease = lease;
    let (id, object) = if completion {
        ("cmpl_eliezer", "text_completion")
    } else {
        ("chatcmpl_eliezer", "chat.completion.chunk")
    };
    let start = std::time::Instant::now();
    let mut first = None;
    let mut block: Option<(usize, String)> = None;
    let mut next_index = 0;
    if anth && !send(&tx,Some("message_start"),json!({"type":"message_start","message":{"id":"msg_eliezer","type":"message","role":"assistant","model":model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}})).await{return;}
    if anth && !send(&tx, Some("ping"), json!({"type":"ping"})).await {
        return;
    }
    loop {
        let chunk = tokio::select! {_ = tx.closed()=>return,chunk=chunks.recv()=>chunk};
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = match chunk {
            Ok(v) => v,
            Err(_) => {
                app.supervisor.report_crash(model.clone()).await;
                let _=send(&tx,if anth {Some("error")}else{None},json!({"type":"error","error":{"type":"api_error","message":"Backend has failed"}})).await;
                return;
            }
        };
        first.get_or_insert_with(|| start.elapsed().as_millis());
        if let Chunk::Done {
            finish_reason,
            input_tokens,
            output_tokens,
        } = chunk
        {
            if anth {
                if let Some((index, _)) = block.take()
                    && !send(
                        &tx,
                        Some("content_block_stop"),
                        json!({"type":"content_block_stop","index":index}),
                    )
                    .await
                {
                    return;
                }
                if !send(&tx,Some("message_delta"),json!({"type":"message_delta","delta":{"stop_reason":anthropic::stop_reason(&finish_reason),"stop_sequence":null},"usage":{"output_tokens":output_tokens}})).await{return;}
                let _ = send(&tx, Some("message_stop"), json!({"type":"message_stop"})).await;
            } else {
                if !send(&tx,None,json!({"id":id,"object":object,"created":0,"model":model,"choices":[if completion {json!({"index":0,"text":"","finish_reason":finish_reason})} else {json!({"index":0,"delta":{},"finish_reason":finish_reason})}]})).await{return;}
                if usage && !send(&tx,None,json!({"id":id,"object":object,"created":0,"model":model,"choices":[],"usage":{"prompt_tokens":input_tokens,"completion_tokens":output_tokens,"total_tokens":input_tokens+output_tokens},"eliezer":{"ttft_ms":first.unwrap_or(0),"gen_ms":start.elapsed().as_millis(),"tokens":output_tokens}})).await{return;}
                let _ = tx.send("data: [DONE]\n\n".into()).await;
            }
            return;
        }
        if !anth {
            if completion {
                // Plain completions carry text only; reasoning and tool calls have no place in them.
                let Chunk::Text(text) = chunk else {
                    continue;
                };
                if !send(&tx,None,json!({"id":id,"object":object,"created":0,"model":model,"choices":[{"index":0,"text":text,"finish_reason":null}]})).await{return;}
                continue;
            }
            let delta = match chunk {
                Chunk::Text(s) => json!({"content":s}),
                Chunk::Thinking(s) => json!({"reasoning_content":s}),
                Chunk::Tool {
                    index,
                    id,
                    name,
                    arguments,
                } => {
                    json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]})
                }
                _ => return,
            };
            if !send(&tx,None,json!({"id":id,"object":object,"created":0,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":null}]})).await{return;}
        } else {
            let (kind, initial, delta) = match chunk {
                Chunk::Text(s) => (
                    "text".to_owned(),
                    json!({"type":"text","text":""}),
                    json!({"type":"text_delta","text":s}),
                ),
                Chunk::Thinking(s) => (
                    "thinking".into(),
                    json!({"type":"thinking","thinking":""}),
                    json!({"type":"thinking_delta","thinking":s}),
                ),
                Chunk::Tool {
                    index,
                    id,
                    name,
                    arguments,
                } => (
                    format!("tool_{index}"),
                    json!({"type":"tool_use","id":id,"name":name,"input":{}}),
                    json!({"type":"input_json_delta","partial_json":arguments}),
                ),
                _ => return,
            };
            if block.as_ref().is_none_or(|(_, old)| old != &kind) {
                if let Some((index, _)) = block.take()
                    && !send(
                        &tx,
                        Some("content_block_stop"),
                        json!({"type":"content_block_stop","index":index}),
                    )
                    .await
                {
                    return;
                }
                let index = next_index;
                next_index += 1;
                if !send(
                    &tx,
                    Some("content_block_start"),
                    json!({"type":"content_block_start","index":index,"content_block":initial}),
                )
                .await
                {
                    return;
                }
                block = Some((index, kind));
            }
            if let Some((index, _)) = &block
                && !send(
                    &tx,
                    Some("content_block_delta"),
                    json!({"type":"content_block_delta","index":index,"delta":delta}),
                )
                .await
            {
                return;
            }
        }
    }
    app.supervisor.report_crash(model).await;
    let _=send(&tx,if anth{Some("error")}else{None},json!({"type":"error","error":{"type":"api_error","message":"Backend response is incomplete"}})).await;
}

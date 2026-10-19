//! HTTP-Schnittstellen des Modellverwalters.
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
use eliezer_core::{config::Config, memory::estimate};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct App {
    pub config: Arc<Config>,
    pub supervisor: Supervisor,
    pub backend: Arc<dyn Backend>,
    pub keys: Arc<Vec<String>>,
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
            .route("/v1/messages", post(messages))
            .route("/v1/messages/count_tokens", post(count_tokens))
            .fallback(|| async {
                failure(
                    StatusCode::NOT_FOUND,
                    "not_found",
                    "Endpunkt nicht gefunden",
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
        Error::UnknownModel => (404, "not_found", "Modell ist nicht freigegeben"),
        Error::InvalidLoad => (400, "invalid_request", "Ungültige Ladeparameter"),
        Error::InsufficientMemory => (
            409,
            "insufficient_memory",
            "Nicht genug Speicher für das Modell",
        ),
        Error::QueueFull | Error::Timeout => (
            if anth { 529 } else { 503 },
            "overloaded",
            "Warteschlange ist voll oder Wartezeit überschritten",
        ),
        Error::Busy => (409, "busy", "Modell wird noch verwendet"),
        Error::Backend => (502, "backend_error", "Backend ist ausgefallen"),
        Error::Stopped => (503, "unavailable", "Dienst wird beendet"),
    };
    failure(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        code,
        message,
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
    if !app.keys.is_empty()
        && !app
            .keys
            .iter()
            .any(|key| bearer == Some(key.as_str()) || (anth && api == Some(key.as_str())))
    {
        return failure(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Anmeldung erforderlich",
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
            "Ungültiger JSON-Anfragekörper",
            anth,
        ))
    })
}
async fn models(State(app): State<App>) -> Response {
    let states = match app.supervisor.snapshot().await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    Json(json!({"models":app.config.model.iter().map(|m| {
        let loaded = states.iter().find(|i| i.key == m.key && i.state == eliezer_core::scheduler::State::Loaded);
        let mut value = json!({"type":match m.kind { eliezer_core::config::ModelKind::Llm => "llm", eliezer_core::config::ModelKind::Embedding => "embedding", _ => "image" },
            "key":m.key,"display_name":m.key,"max_context_length":m.max_context,"capabilities":{"vision":m.vision},"loaded_instances":[]});
        if let Some(reasoning) = &m.reasoning { value["capabilities"]["reasoning"] = json!({"allowed_options":reasoning.options,"default":reasoning.default}); }
        if let Some(size) = m.size_gb.and_then(|gb| eliezer_core::memory::gb_bytes(gb).ok()) { value["size_bytes"] = json!(size); }
        if loaded.is_some() { value["loaded_instances"] = json!([{"id":m.key,"config":{"context_length":m.default_context,"parallel":m.parallel}}]); }
        value
    }).collect::<Vec<_>>()})).into_response()
}
async fn list(State(app): State<App>, headers: axum::http::HeaderMap) -> Response {
    let anth = headers.contains_key("anthropic-version");
    let data: Vec<_> = app.config.model.iter().map(|m| if anth { json!({"type":"model","id":m.key,"display_name":m.key,"created_at":"2026-10-19T00:00:00Z"}) }
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
    let model = app
        .config
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
        std::time::Duration::from_secs(app.config.server.request_timeout_secs),
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
    match app.supervisor.snapshot().await {
        Ok(instances) => {
            let used = instances
                .iter()
                .filter(|i| {
                    matches!(
                        i.state,
                        eliezer_core::scheduler::State::Loaded
                            | eliezer_core::scheduler::State::Loading
                            | eliezer_core::scheduler::State::Unloading
                    )
                })
                .fold(0_u64, |s, i| s.saturating_add(i.bytes));
            let total = eliezer_core::memory::gb_bytes(app.config.gpu.vram_budget_gb).unwrap_or(0);
            Json(json!({"gpu":{"probe":"none","name":"Buchführung","total_bytes":total,"used_bytes":used,"free_bytes":total.saturating_sub(used),"budget_bytes":total},
                "models":instances.iter().map(|i|json!({"key":i.key,"state":format!("{:?}",i.state).to_lowercase(),"vram_bytes":i.bytes,"in_flight":i.in_flight})).collect::<Vec<_>>(),"queue":[]})).into_response()
        }
        Err(e) => error(e, false),
    }
}
async fn chat(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match json_body(body, false) {
        Ok(v) => complete(app, v, false).await,
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
    let Some(model) = app
        .config
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
        Ok(v) => complete(app, v, true).await,
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
    let Some(model) = app
        .config
        .model
        .iter()
        .find(|m| Some(m.key.as_str()) == body["model"].as_str())
    else {
        return error(Error::UnknownModel, true);
    };
    // count_tokens benötigt anders als messages kein max_tokens.
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
async fn complete(app: App, mut body: Value, anth: bool) -> Response {
    let spec = match spec(&app, &body) {
        Ok(v) => v,
        Err(e) => return error(e, anth),
    };
    let Some(model) = app.config.model.iter().find(|m| m.key == spec.model) else {
        return error(Error::UnknownModel, anth);
    };
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
                "Denkoption wird vom Modell nicht unterstützt",
                anth,
            );
        }
        body["reasoning_effort"] = json!(effort);
    } else if body.get("reasoning_effort").is_some() {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Modell unterstützt keine Denkoptionen",
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
            anth,
            body["stream_options"]["include_usage"] == true,
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
    let Some((finish, input, output)) = done else {
        return error(Error::Backend, anth);
    };
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
async fn stream_response(
    app: App,
    lease: Lease,
    mut chunks: mpsc::Receiver<Result<Chunk, eliezer_backend::BackendError>>,
    tx: mpsc::Sender<String>,
    model: String,
    anth: bool,
    usage: bool,
) {
    let _lease = lease;
    let start = std::time::Instant::now();
    let mut first = None;
    let mut block: Option<(usize, String)> = None;
    let mut next_index = 0;
    if anth && !send(&tx,Some("message_start"),json!({"type":"message_start","message":{"id":"msg_eliezer","type":"message","role":"assistant","model":model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}})).await{return;}
    loop {
        let chunk = tokio::select! {_ = tx.closed()=>return,chunk=chunks.recv()=>chunk};
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = match chunk {
            Ok(v) => v,
            Err(_) => {
                app.supervisor.report_crash(model.clone()).await;
                let _=send(&tx,if anth {Some("error")}else{None},json!({"type":"error","error":{"type":"api_error","message":"Backend ist ausgefallen"}})).await;
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
                if !send(&tx,None,json!({"id":"chatcmpl_eliezer","object":"chat.completion.chunk","created":0,"model":model,"choices":[{"index":0,"delta":{},"finish_reason":finish_reason}]})).await{return;}
                if usage && !send(&tx,None,json!({"id":"chatcmpl_eliezer","object":"chat.completion.chunk","created":0,"model":model,"choices":[],"usage":{"prompt_tokens":input_tokens,"completion_tokens":output_tokens,"total_tokens":input_tokens+output_tokens},"eliezer":{"ttft_ms":first.unwrap_or(0),"gen_ms":start.elapsed().as_millis(),"tokens":output_tokens}})).await{return;}
                let _ = tx.send("data: [DONE]\n\n".into()).await;
            }
            return;
        }
        if !anth {
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
            if !send(&tx,None,json!({"id":"chatcmpl_eliezer","object":"chat.completion.chunk","created":0,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":null}]})).await{return;}
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
    let _=send(&tx,if anth{Some("error")}else{None},json!({"type":"error","error":{"type":"api_error","message":"Backend-Antwort ist unvollständig"}})).await;
}

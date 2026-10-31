//! `POST /v1/audio/speech`: the endpoint chooses by the kind of the model. A model with `kind = "music"` composes a song
//! from lyrics and a description: the lyrics are `input`, the music description is `instructions`, the answer is the
//! WAV file (`audio/wav`). A model with `kind = "tts"` reads `input` aloud in the voice named in `voice` (see
//! `tts.rs`).

use super::{App, Error, error, failure, spec, valid_music_length};
use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use eliezer_core::config::ModelKind;
use serde_json::{Value, json};

const MAX_TEXT_CHARS: usize = 20_000;
/// The length drawn when neither the request nor the model names one.
const DEFAULT_SECONDS: u32 = 30;

fn bad(message: &str) -> Response {
    failure(StatusCode::BAD_REQUEST, "invalid_request", message, false)
}

pub async fn speech(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            return failure(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "The request is too big",
                false,
            );
        }
        Err(_) => return bad("Invalid JSON request body"),
    };
    let config = app.config.get();
    let model = match body["model"].as_str() {
        Some(key) => match config.model.iter().find(|m| m.key == key) {
            Some(model) => model,
            None => return error(Error::UnknownModel, false),
        },
        None => {
            let mut found = config
                .model
                .iter()
                .filter(|m| matches!(m.kind, ModelKind::Music | ModelKind::Tts));
            match (found.next(), found.next()) {
                (Some(only), None) => only,
                _ => {
                    return bad(
                        "model is required when there is not exactly one music or speech synthesis model",
                    );
                }
            }
        }
    };
    match model.kind {
        ModelKind::Music => {}
        ModelKind::Tts => return super::tts::speak(&app, &config, model, &body).await,
        _ => return bad("Model is not a music or speech synthesis model"),
    }
    // A text field: the text, or a message why it is wrong.
    let text = |name: &str, meaning: &str| -> Result<String, String> {
        match body.get(name) {
            Some(Value::String(text)) if !text.trim().is_empty() => Ok(text.clone()),
            None | Some(Value::Null) => Err(format!("{name} ({meaning}) is required")),
            Some(_) => Err(format!("{name} must be a non-empty string")),
        }
    };
    let lyrics = match text("input", "the lyrics") {
        Ok(lyrics) => lyrics,
        Err(message) => return bad(&message),
    };
    let description = match text("instructions", "the description of the music") {
        Ok(description) => description,
        Err(message) => return bad(&message),
    };
    if lyrics.chars().count() > MAX_TEXT_CHARS || description.chars().count() > MAX_TEXT_CHARS {
        return bad("input and instructions are limited to 20000 characters each");
    }
    if body
        .get("response_format")
        .is_some_and(|f| !f.is_null() && f != "wav")
    {
        return bad("Only response_format wav is supported");
    }
    let duration_sec = match body.get("duration_sec") {
        None | Some(Value::Null) => model
            .extra
            .as_ref()
            .and_then(|e| e.duration_sec)
            .unwrap_or(DEFAULT_SECONDS),
        Some(value) => match value.as_u64().and_then(|n| u32::try_from(n).ok()) {
            Some(n) if valid_music_length(n) => n,
            _ => return bad("duration_sec must be a whole number between 1 and 300"),
        },
    };
    let seed = match body.get("seed") {
        None | Some(Value::Null) => None,
        Some(seed) => match seed.as_i64() {
            Some(seed) => Some(seed),
            None => return bad("seed must be a whole number"),
        },
    };
    let spec = match spec(&app, &json!({"model": model.key})) {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let lease = match app.supervisor.acquire(spec.clone()).await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let request = eliezer_backend::MusicRequest {
        model: spec.model,
        lyrics,
        description,
        duration_sec,
        steps: None,
        seed,
    };
    match app
        .supervisor
        .music(&lease, request, app.backend.as_ref())
        .await
    {
        Ok(wav) => ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response(),
        Err(e) => error(e, false),
    }
}

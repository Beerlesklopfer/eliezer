//! `POST /v1/audio/speech` for models with `kind = "tts"`: the text is read aloud in the voice of a sample from the voice
//! directory. The request is JSON: `model`, `input`, `voice` (a name from `GET /v1/audio/voices`, or empty for the
//! `default_voice` of the model, else the model's own speaker), `response_format` (`wav` only), and optionally `speed`,
//! `language` and `seed`. A long text is cut into passages at sentence boundaries, the passages are read one after the
//! other by the same loaded model, and the audio is joined with the pauses of the configuration, so a chapter works.
//! The answer is one WAV file (`audio/wav`).

use super::{App, Error, error, failure, spec, voices};
use axum::{
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use eliezer_backend::SpeechRequest;
use eliezer_core::config::{Config, ModelConfig};
use eliezer_speech::{
    passages::passages,
    pcm::{read_wav, write_wav},
};
use serde_json::{Value, json};

fn bad(message: &str) -> Response {
    failure(StatusCode::BAD_REQUEST, "invalid_request", message, false)
}

/// The limits and pauses of a model: its own, else those of `[backend.audiocpp]`.
struct Limits {
    input: usize,
    passage: usize,
    pause_ms: u32,
    paragraph_pause_ms: u32,
}

fn limits(config: &Config, model: &ModelConfig) -> Option<Limits> {
    let audio = config.backend.audiocpp.as_ref()?;
    let extra = model.extra.as_ref();
    Some(Limits {
        input: extra
            .and_then(|e| e.max_input_chars)
            .unwrap_or(audio.max_input_chars) as usize,
        passage: extra
            .and_then(|e| e.max_passage_chars)
            .unwrap_or(audio.max_passage_chars) as usize,
        pause_ms: extra.and_then(|e| e.pause_ms).unwrap_or(audio.pause_ms),
        paragraph_pause_ms: extra
            .and_then(|e| e.paragraph_pause_ms)
            .unwrap_or(audio.paragraph_pause_ms),
    })
}

pub async fn speak(app: &App, config: &Config, model: &ModelConfig, body: &Value) -> Response {
    let Some(limits) = limits(config, model) else {
        return failure(
            StatusCode::NOT_FOUND,
            "not_found",
            "No audio backend is configured",
            false,
        );
    };
    let extra = model.extra.as_ref();
    let input = match body.get("input") {
        Some(Value::String(text)) if !text.trim().is_empty() => text.as_str(),
        None | Some(Value::Null) => return bad("input (the text) is required"),
        Some(_) => return bad("input must be a non-empty string"),
    };
    if input.chars().count() > limits.input {
        return failure(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "input is longer than the limit of the model",
            false,
        );
    }
    if body
        .get("response_format")
        .is_some_and(|f| !f.is_null() && f != "wav")
    {
        return bad("Only response_format wav is supported");
    }
    let speed = match body.get("speed") {
        None | Some(Value::Null) => extra.and_then(|e| e.speed),
        Some(value) => match value.as_f64() {
            Some(s) if (0.25..=4.0).contains(&s) => Some(s),
            _ => return bad("speed must be a number from 0.25 to 4"),
        },
    };
    let language = match body.get("language") {
        None | Some(Value::Null) => extra.and_then(|e| e.language.clone()),
        Some(Value::String(l)) if l.is_empty() => extra.and_then(|e| e.language.clone()),
        Some(Value::String(l))
            if (2..=3).contains(&l.len()) && l.bytes().all(|b| b.is_ascii_lowercase()) =>
        {
            Some(l.clone())
        }
        Some(_) => return bad("language must be a language code such as de"),
    };
    let seed = match body.get("seed") {
        None | Some(Value::Null) => None,
        Some(value) => match value.as_u64() {
            Some(seed) if seed <= u64::from(u32::MAX) => Some(seed),
            _ => return bad("seed must be a whole number from 0 to 4294967295"),
        },
    };
    // The voice: a name from the directory, never a path. An empty name is the default of the model.
    let requested = match body.get("voice") {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) if name.is_empty() => None,
        Some(Value::String(name)) => Some(name.as_str()),
        Some(_) => return bad("voice must be a string"),
    };
    let voice = match requested.or(extra.and_then(|e| e.default_voice.as_deref())) {
        None => None,
        Some(name) => match voices::find(config, name) {
            Some(sample) => Some(sample),
            None => return bad("Unknown voice (see GET /v1/audio/voices)"),
        },
    };
    let list = passages(input, limits.passage);
    if list.is_empty() {
        return bad("input has no text to read");
    }
    let spec = match spec(app, &json!({"model": model.key})) {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let lease = match app.supervisor.acquire(spec.clone()).await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let mut parts = Vec::with_capacity(list.len());
    for passage in list {
        let request = SpeechRequest {
            model: spec.model.clone(),
            text: passage.text,
            voice: voice.clone(),
            language: language.clone(),
            speed,
            seed,
        };
        let wav = match app
            .supervisor
            .speak(&lease, request, app.backend.as_ref())
            .await
        {
            Ok(wav) => wav,
            Err(e) => return error(e, false),
        };
        let Ok(pcm) = read_wav(&wav) else {
            tracing::warn!(model = %model.key, "The speech program answered without usable audio");
            return error(Error::Backend, false);
        };
        let pause = if passage.ends_paragraph {
            limits.paragraph_pause_ms
        } else {
            limits.pause_ms
        };
        parts.push((pcm, pause));
    }
    match eliezer_speech::pcm::join(parts) {
        Ok(joined) => ([(header::CONTENT_TYPE, "audio/wav")], write_wav(&joined)).into_response(),
        Err(_) => {
            tracing::warn!(model = %model.key, "The passages differ in sample rate or channels");
            error(Error::Backend, false)
        }
    }
}

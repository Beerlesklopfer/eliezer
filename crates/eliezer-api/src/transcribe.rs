//! `POST /v1/audio/transcriptions`: speech recognition, OpenAI shape. The upload is decoded in the service to 16 kHz mono
//! (any of MP3, M4A/AAC, Ogg Vorbis, FLAC, WAV), cut into overlapping pieces of the configured length, the pieces go to the
//! speech program one after the other, and the segments are joined with the offset of their piece, so every time in
//! the answer is relative to the whole file. The limits (file size, length) and the piece length come from the configuration.

use super::{App, Error, error, failure, spec};
use axum::{
    Json,
    extract::{Multipart, State, multipart::MultipartRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use eliezer_backend::TranscriptionRequest;
use eliezer_core::config::{ModelConfig, ModelKind};
use eliezer_speech::{DecodeError, Part};
use serde_json::{Value, json};

fn refuse(status: StatusCode, code: &str, message: &str) -> Response {
    failure(status, code, message, false)
}
fn bad(message: &str) -> Response {
    refuse(StatusCode::BAD_REQUEST, "invalid_request", message)
}
fn too_large(message: &str) -> Response {
    refuse(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", message)
}
fn read_error(error: &axum::extract::multipart::MultipartError) -> Response {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        too_large("The request is too big")
    } else {
        bad("The multipart request body is invalid")
    }
}

/// The limits of a model: its own, else those of `[backend.whispercpp]`.
fn limits(app: &App, model: &ModelConfig) -> (u64, u64) {
    let backend = app.config.get();
    let defaults = backend.backend.whispercpp.as_ref();
    let extra = model.extra.as_ref();
    let file_mb = extra
        .and_then(|e| e.max_file_mb)
        .or(defaults.map(|d| d.max_file_mb))
        .unwrap_or(0);
    let minutes = extra
        .and_then(|e| e.max_audio_minutes)
        .or(defaults.map(|d| d.max_audio_minutes))
        .unwrap_or(0);
    (u64::from(file_mb) << 20, u64::from(minutes) * 60)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

pub async fn transcriptions(
    State(app): State<App>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let Ok(mut multipart) = multipart else {
        return bad("The request must be multipart/form-data");
    };
    let config = app.config.get();
    // The file is read in pieces and stops at the largest limit of any speech model; the limit of the chosen model
    // (which can come in a field after the file) is checked afterwards.
    let ceiling: Option<u64> = config
        .model
        .iter()
        .filter(|m| m.kind == ModelKind::Stt)
        .map(|m| limits(&app, m).0)
        .max();
    let (mut model, mut language, mut prompt, mut format, mut temperature) =
        (None, None, None, None, None);
    let mut granularities: Vec<String> = Vec::new();
    let mut file: Option<Vec<u8>> = None;
    loop {
        let mut field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => return read_error(&e),
        };
        let name = field.name().unwrap_or_default().to_owned();
        match name.as_str() {
            "file" => {
                let mut bytes = Vec::new();
                loop {
                    match field.chunk().await {
                        Ok(Some(chunk)) => {
                            // Without a speech model nothing is kept: the answer will be "not found" anyway.
                            let Some(ceiling) = ceiling else { continue };
                            if bytes.len() as u64 + chunk.len() as u64 > ceiling {
                                return too_large("The file is larger than the limit of the model");
                            }
                            bytes.extend_from_slice(&chunk);
                        }
                        Ok(None) => break,
                        Err(e) => return read_error(&e),
                    }
                }
                file = Some(bytes);
            }
            "model"
            | "language"
            | "prompt"
            | "response_format"
            | "temperature"
            | "timestamp_granularities[]"
            | "timestamp_granularities" => match field.text().await {
                Ok(text) => match name.as_str() {
                    "model" => model = Some(text),
                    "language" => language = Some(text),
                    "prompt" => prompt = Some(text),
                    "response_format" => format = Some(text),
                    "temperature" => temperature = Some(text),
                    _ => granularities.push(text),
                },
                Err(e) => return read_error(&e),
            },
            // stream and the like are accepted and ignored.
            _ => {
                if let Err(e) = field.bytes().await {
                    return read_error(&e);
                }
            }
        }
    }
    let found = match model.as_deref().filter(|m| !m.is_empty()) {
        Some(key) => match config.model.iter().find(|m| m.key == key) {
            Some(found) => found,
            None => return error(Error::UnknownModel, false),
        },
        None => {
            let mut speech = config.model.iter().filter(|m| m.kind == ModelKind::Stt);
            match (speech.next(), speech.next()) {
                (Some(only), None) => only,
                (None, _) => {
                    return refuse(
                        StatusCode::NOT_FOUND,
                        "not_found",
                        "No speech recognition model is configured",
                    );
                }
                _ => return bad("model is required when there is more than one speech model"),
            }
        }
    };
    if found.kind != ModelKind::Stt {
        return bad("Model is not a speech recognition model");
    }
    let Some(file) = file else {
        return bad("file (the audio) is required");
    };
    let format = match format.as_deref().filter(|f| !f.is_empty()) {
        None | Some("json") => Format::Json,
        Some("text") => Format::Text,
        Some("verbose_json") => Format::VerboseJson,
        Some("srt") => Format::Srt,
        Some("vtt") => Format::Vtt,
        Some(_) => return bad("response_format is json, text, verbose_json, srt or vtt"),
    };
    if granularities.iter().any(|g| g != "segment") {
        return bad("Only timestamp_granularities segment is supported");
    }
    let temperature = match temperature.as_deref().filter(|t| !t.is_empty()) {
        None => None,
        Some(text) => match text.trim().parse::<f64>() {
            Ok(t) if t.is_finite() && (0.0..=1.0).contains(&t) => Some(t),
            _ => return bad("temperature must be a number from 0 to 1"),
        },
    };
    let default_language = found
        .extra
        .as_ref()
        .and_then(|e| e.language.clone())
        .unwrap_or_else(|| "auto".to_owned());
    let language = match language.filter(|l| !l.is_empty()) {
        None => default_language,
        Some(l)
            if l == "auto"
                || ((2..=3).contains(&l.len()) && l.bytes().all(|b| b.is_ascii_lowercase())) =>
        {
            l
        }
        Some(_) => return bad("language must be auto or a language code such as de"),
    };
    let (max_bytes, max_seconds) = limits(&app, found);
    if file.len() as u64 > max_bytes {
        return too_large("The file is larger than the limit of the model");
    }
    let Some(settings) = config.backend.whispercpp.as_ref() else {
        return refuse(
            StatusCode::NOT_FOUND,
            "not_found",
            "No speech recognition backend is configured",
        );
    };
    let (chunk_secs, overlap_secs) = (settings.chunk_secs, settings.overlap_secs);
    let samples = match tokio::task::spawn_blocking(move || {
        eliezer_speech::decode(file, max_seconds)
    })
    .await
    {
        Ok(Ok(samples)) => samples,
        Ok(Err(DecodeError::NotAudio)) => {
            return bad("The file is not audio this service can read");
        }
        Ok(Err(DecodeError::TooLong)) => {
            return too_large("The recording is longer than the limit of the model");
        }
        Err(_) => return error(Error::Backend, false),
    };
    let duration = samples.len() as f64 / f64::from(eliezer_speech::SAMPLE_RATE);
    let spec = match spec(&app, &json!({"model": found.key})) {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let lease = match app.supervisor.acquire(spec.clone()).await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let mut parts = Vec::new();
    let mut reported: Option<String> = None;
    for piece in eliezer_speech::plan(samples.len(), chunk_secs, overlap_secs) {
        let offset = piece.start as f64 / f64::from(eliezer_speech::SAMPLE_RATE);
        let request = TranscriptionRequest {
            model: spec.model.clone(),
            wav: eliezer_speech::wav(&samples[piece]),
            language: Some(language.clone()),
            prompt: prompt.clone().filter(|p| !p.trim().is_empty()),
            temperature,
        };
        match app
            .supervisor
            .transcribe(&lease, request, app.backend.as_ref())
            .await
        {
            Ok(done) => {
                reported = reported.or_else(|| done.language.clone());
                parts.push(Part {
                    offset,
                    transcription: done,
                });
            }
            Err(e) => return error(e, false),
        }
    }
    let segments = eliezer_speech::join(parts, f64::from(overlap_secs));
    let text = eliezer_speech::text(&segments);
    match format {
        Format::Json => Json(json!({"text": text})).into_response(),
        Format::Text => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            format!("{text}\n"),
        )
            .into_response(),
        Format::Srt => (
            [(header::CONTENT_TYPE, "application/x-subrip; charset=utf-8")],
            eliezer_speech::srt(&segments),
        )
            .into_response(),
        Format::Vtt => (
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            eliezer_speech::vtt(&segments),
        )
            .into_response(),
        Format::VerboseJson => {
            let shown = if language == "auto" {
                reported.unwrap_or(language)
            } else {
                language
            };
            let segments: Vec<Value> = segments
                .iter()
                .enumerate()
                .map(|(id, s)| json!({"id": id, "start": s.start, "end": s.end, "text": s.text}))
                .collect();
            Json(json!({
                "task": "transcribe",
                "language": shown,
                "duration": duration,
                "text": text,
                "segments": segments,
            }))
            .into_response()
        }
    }
}

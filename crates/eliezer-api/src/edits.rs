//! `POST /v1/images/edits`: a picture (or up to four) and a prompt go in as `multipart/form-data`, new pictures
//! come out (as for `/v1/images/generations`). Only models with `extra.image_edit = true` accept it. The files are
//! checked by their first bytes and size here; the program that draws them decodes them.

use super::{App, Error, error, failure, spec, unix_now, valid_size};
use axum::{
    Json,
    extract::{Multipart, State, multipart::MultipartRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use eliezer_backend::image_type;
use eliezer_core::config::ModelKind;
use serde_json::json;

const MAX_IMAGES: usize = 4;
const MAX_IMAGE_BYTES: usize = 20 << 20;
/// All files, the text fields and the framing.
pub const MAX_BODY: usize = MAX_IMAGES * MAX_IMAGE_BYTES + (1 << 20);

fn refuse(status: StatusCode, code: &str, message: &str) -> Response {
    failure(status, code, message, false)
}
fn bad(message: &str) -> Response {
    refuse(StatusCode::BAD_REQUEST, "invalid_request", message)
}
fn too_large() -> Response {
    refuse(
        StatusCode::PAYLOAD_TOO_LARGE,
        "payload_too_large",
        "A picture is larger than 20 MB or the request is too big",
    )
}
fn read_error(error: &axum::extract::multipart::MultipartError) -> Response {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        too_large()
    } else {
        bad("The multipart request body is invalid")
    }
}

pub async fn edits(
    State(app): State<App>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let Ok(mut multipart) = multipart else {
        return bad("The request must be multipart/form-data");
    };
    let (mut model, mut prompt, mut count, mut size, mut seed) = (None, None, None, None, None);
    let mut images: Vec<Vec<u8>> = Vec::new();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => return read_error(&e),
        };
        let name = field.name().unwrap_or_default().to_owned();
        match name.as_str() {
            "image" | "image[]" => {
                if images.len() == MAX_IMAGES {
                    return bad("At most 4 pictures can be edited at once");
                }
                match field.bytes().await {
                    Ok(bytes) if bytes.len() > MAX_IMAGE_BYTES => return too_large(),
                    Ok(bytes) => images.push(bytes.to_vec()),
                    Err(e) => return read_error(&e),
                }
            }
            "mask" => return bad("A mask is not supported"),
            "model" | "prompt" | "n" | "size" | "seed" => match field.text().await {
                Ok(text) => match name.as_str() {
                    "model" => model = Some(text),
                    "prompt" => prompt = Some(text),
                    "n" => count = Some(text),
                    "size" => size = Some(text),
                    _ => seed = Some(text),
                },
                Err(e) => return read_error(&e),
            },
            // quality, background, output_format and the like are accepted and ignored.
            _ => {
                if let Err(e) = field.bytes().await {
                    return read_error(&e);
                }
            }
        }
    }
    let config = app.config.get();
    let found = match model.as_deref() {
        Some(key) => match config.model.iter().find(|m| m.key == key) {
            Some(found) => found,
            None => return error(Error::UnknownModel, false),
        },
        None => {
            let mut editors = config.model.iter().filter(|m| {
                m.kind == ModelKind::Image
                    && m.extra.as_ref().and_then(|e| e.image_edit) == Some(true)
            });
            match (editors.next(), editors.next()) {
                (Some(only), None) => only,
                _ => {
                    return bad(
                        "model is required when there is not exactly one model that edits pictures",
                    );
                }
            }
        }
    };
    if found.kind != ModelKind::Image {
        return bad("Model is not an image model");
    }
    if found.extra.as_ref().and_then(|e| e.image_edit) != Some(true) {
        return bad("This model cannot edit pictures (extra.image_edit is not set)");
    }
    let Some(prompt) = prompt.filter(|p| !p.trim().is_empty()) else {
        return bad("prompt must be a non-empty string");
    };
    if images.is_empty() {
        return bad("At least one picture is required (image or image[])");
    }
    if images.iter().any(|bytes| image_type(bytes).is_none()) {
        return bad("Pictures must be PNG, JPEG or WebP files");
    }
    let count = match count {
        None => 1,
        Some(n) => match n.trim().parse::<u32>().ok().filter(|n| (1..=4).contains(n)) {
            Some(n) => n,
            None => return bad("n must be between 1 and 4"),
        },
    };
    let size = match size.filter(|s| !s.is_empty()) {
        None => None,
        Some(size) if valid_size(&size) => Some(size),
        Some(_) => return bad("size must look like 512x512"),
    };
    let seed = match seed.filter(|s| !s.is_empty()) {
        None => None,
        Some(seed) => match seed.trim().parse::<i64>() {
            Ok(seed) => Some(seed),
            Err(_) => return bad("seed must be a whole number"),
        },
    };
    let spec = match spec(&app, &json!({"model": found.key})) {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let lease = match app.supervisor.acquire(spec.clone()).await {
        Ok(v) => v,
        Err(e) => return error(e, false),
    };
    let request = eliezer_backend::ImageRequest {
        model: spec.model,
        prompt,
        count,
        size,
        seed,
        references: images,
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

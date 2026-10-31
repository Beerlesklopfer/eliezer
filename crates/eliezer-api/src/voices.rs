//! The voices of speech synthesis: reference recordings that a model clones. A voice is the file `<name>.wav` (mono,
//! 16 bit, at `backend.audiocpp.voice_sample_rate`) and optionally `<name>.txt` (the words spoken in it) in the voice
//! directory of the configuration. The directory is the administrator's: the service reads it, and the upload and
//! delete endpoints write to it (with the API key like the other modifying endpoints), but the samples are never
//! served, logged or listed with a path. A name from a request is checked against the names found in the directory
//! before any path is built from it.

use super::{App, failure};
use axum::{
    Json,
    extract::{Multipart, Path, State, multipart::MultipartRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use eliezer_backend::VoiceSample;
use eliezer_core::config::{AudiocppConfig, Config, valid_voice_name};
use serde_json::json;
use std::{
    fs,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path as FsPath, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// The longest transcript (bytes) that is read, uploaded or sent to the program.
const MAX_TRANSCRIPT_BYTES: usize = 8192;
/// The longest text field of the upload form (bytes).
const MAX_FIELD_BYTES: usize = 16 * 1024;

fn refuse(status: StatusCode, code: &str, message: &str) -> Response {
    failure(status, code, message, false)
}
fn bad(message: &str) -> Response {
    refuse(StatusCode::BAD_REQUEST, "invalid_request", message)
}
/// How much of a refused upload is read and discarded so that the refusal reaches the client.
const DRAIN_MAX_BYTES: u64 = 1 << 30;

fn too_large(message: &str) -> Response {
    refuse(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", message)
}
fn not_configured() -> Response {
    refuse(
        StatusCode::NOT_FOUND,
        "not_found",
        "No audio backend is configured",
    )
}
fn read_error(error: &axum::extract::multipart::MultipartError) -> Response {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        too_large("The request is too big")
    } else {
        bad("The multipart request body is invalid")
    }
}
fn io_failure(what: &str, error: &std::io::Error) -> Response {
    tracing::error!(kind = ?error.kind(), "{what}");
    refuse(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        "The voice directory cannot be used",
    )
}

/// The audio settings of the configuration, when the backend is configured.
fn audio(config: &Config) -> Option<&AudiocppConfig> {
    config.backend.audiocpp.as_ref()
}

/// The names of the voices in the directory: regular files `<name>.wav` with a valid name (symbolic links and other
/// files are ignored), sorted. A directory that does not exist has no voices.
pub fn names(dir: &FsPath, max_chars: u32) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let file = e.file_name().into_string().ok()?;
            let stem = file.strip_suffix(".wav")?;
            valid_voice_name(stem, max_chars).then(|| stem.to_owned())
        })
        .collect();
    names.sort();
    names
}

/// The sample of a voice by name, or `None` when the name is not valid or not in the directory. The path is built
/// only after the name was found in the listing.
pub fn find(config: &Config, name: &str) -> Option<VoiceSample> {
    let audio = audio(config)?;
    if !valid_voice_name(name, audio.voice_name_max_chars) {
        return None;
    }
    let dir = config.voices_dir();
    let known = names(&dir, audio.voice_name_max_chars);
    let name = known.iter().find(|n| n.as_str() == name)?;
    let transcript = fs::File::open(dir.join(format!("{name}.txt")))
        .ok()
        .and_then(|file| {
            use std::io::Read;
            let mut text = Vec::new();
            file.take(MAX_TRANSCRIPT_BYTES as u64)
                .read_to_end(&mut text)
                .ok()?;
            let text = String::from_utf8(text).ok()?;
            let text = text.trim();
            (!text.is_empty()).then(|| text.to_owned())
        });
    Some(VoiceSample {
        path: dir.join(format!("{name}.wav")),
        transcript,
    })
}

/// `GET /v1/audio/voices`: the names of the voices, nothing else.
pub async fn list(State(app): State<App>) -> Response {
    let config = app.config.get();
    let Some(audio) = audio(&config) else {
        return not_configured();
    };
    let voices = names(&config.voices_dir(), audio.voice_name_max_chars);
    Json(json!({"voices": voices})).into_response()
}

/// A text field of the form, at most `cap` bytes.
async fn text_field(
    mut field: axum::extract::multipart::Field<'_>,
    cap: usize,
) -> Result<String, Box<Response>> {
    let mut bytes = Vec::new();
    loop {
        match field.chunk().await {
            Ok(Some(chunk)) => {
                if bytes.len() + chunk.len() > cap {
                    return Err(Box::new(too_large("A field of the form is too long")));
                }
                bytes.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(e) => return Err(Box::new(read_error(&e))),
        }
    }
    String::from_utf8(bytes).map_err(|_| Box::new(bad("A field of the form is not text")))
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to a new file of the directory (mode 0600) that is not a voice yet and returns its path.
fn write_temporary(dir: &FsPath, bytes: &[u8]) -> std::io::Result<PathBuf> {
    use std::io::Write;
    let path = dir.join(format!(
        ".upload-{}-{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut out = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        out.write_all(bytes)?;
        out.sync_all()
    })();
    match result {
        Ok(()) => Ok(path),
        Err(error) => {
            let _ = fs::remove_file(&path);
            Err(error)
        }
    }
}

/// `POST /v1/audio/voices` (multipart): `name`, `file` (any audio the service can decode), optional `transcript` and
/// `replace=true`. The file is stored as mono 16 bit WAV at the configured sample rate.
pub async fn upload(
    State(app): State<App>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let Ok(mut multipart) = multipart else {
        return bad("The request must be multipart/form-data");
    };
    let config = app.config.get();
    let Some(settings) = audio(&config) else {
        return not_configured();
    };
    let max_bytes = u64::from(settings.voice_max_file_mb) << 20;
    let (mut name, mut transcript, mut replace) = (None, None, false);
    let mut file: Option<Vec<u8>> = None;
    loop {
        let mut field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => return read_error(&e),
        };
        match field.name().unwrap_or_default() {
            "file" => {
                let mut bytes = Vec::new();
                loop {
                    match field.chunk().await {
                        Ok(Some(chunk)) => {
                            if bytes.len() as u64 + chunk.len() as u64 > max_bytes {
                                // Read the rest of the file and throw it away (up to a bound) before answering:
                                // a connection that is closed with unread data is reset, and the reset can
                                // destroy the answer on its way, so the client would see a broken pipe instead of
                                // this message.
                                let mut skipped = (bytes.len() + chunk.len()) as u64;
                                while skipped < DRAIN_MAX_BYTES {
                                    match field.chunk().await {
                                        Ok(Some(more)) => skipped += more.len() as u64,
                                        _ => break,
                                    }
                                }
                                return too_large("The file is larger than the limit");
                            }
                            bytes.extend_from_slice(&chunk);
                        }
                        Ok(None) => break,
                        Err(e) => return read_error(&e),
                    }
                }
                file = Some(bytes);
            }
            "name" => match text_field(field, MAX_FIELD_BYTES).await {
                Ok(text) => name = Some(text),
                Err(response) => return *response,
            },
            "transcript" => match text_field(field, MAX_TRANSCRIPT_BYTES).await {
                Ok(text) => transcript = Some(text),
                Err(response) => return *response,
            },
            "replace" => match text_field(field, 16).await {
                Ok(text) => {
                    replace = match text.trim() {
                        "true" => true,
                        "" | "false" => false,
                        _ => return bad("replace is true or false"),
                    }
                }
                Err(response) => return *response,
            },
            _ => {
                if let Err(e) = field.bytes().await {
                    return read_error(&e);
                }
            }
        }
    }
    let Some(name) = name else {
        return bad("name is required");
    };
    if !valid_voice_name(&name, settings.voice_name_max_chars) {
        return bad("name has letters, digits, - and _ only, and is not longer than the limit");
    }
    let Some(file) = file else {
        return bad("file (the recording) is required");
    };
    let transcript = transcript
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty());
    let (seconds, rate) = (
        u64::from(settings.voice_max_seconds),
        settings.voice_sample_rate,
    );
    let samples =
        match tokio::task::spawn_blocking(move || eliezer_speech::decode_at(file, seconds, rate))
            .await
        {
            Ok(Ok(samples)) => samples,
            Ok(Err(eliezer_speech::DecodeError::NotAudio)) => {
                return bad("The file is not audio this service can read");
            }
            Ok(Err(eliezer_speech::DecodeError::TooLong)) => {
                return too_large("The recording is longer than the limit");
            }
            Err(_) => return refuse(StatusCode::BAD_GATEWAY, "backend_error", "Decoding failed"),
        };
    let wav = eliezer_speech::wav_at(&samples, rate);
    store(&config, &name, &wav, transcript.as_deref(), replace)
}

/// Puts the voice into the directory (created with mode 0700 when it is missing). Without `replace` an existing voice
/// is never touched: the final name is made with a hard link, which fails when the name exists.
fn store(
    config: &Arc<Config>,
    name: &str,
    wav: &[u8],
    transcript: Option<&str>,
    replace: bool,
) -> Response {
    let Some(settings) = audio(config) else {
        return not_configured();
    };
    let dir = config.voices_dir();
    if let Err(e) = fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)
    {
        return io_failure("The voice directory cannot be created", &e);
    }
    let known = names(&dir, settings.voice_name_max_chars);
    let exists = known.iter().any(|n| n == name);
    if exists && !replace {
        return refuse(
            StatusCode::CONFLICT,
            "conflict",
            "A voice with this name exists; send replace=true to overwrite it",
        );
    }
    if !exists && known.len() >= settings.max_voices as usize {
        return refuse(
            StatusCode::CONFLICT,
            "voice_limit",
            "The number of voices is at the limit",
        );
    }
    let target = dir.join(format!("{name}.wav"));
    let text = dir.join(format!("{name}.txt"));
    let sample = match write_temporary(&dir, wav) {
        Ok(path) => path,
        Err(e) => return io_failure("A voice sample cannot be written", &e),
    };
    let placed = if replace {
        fs::rename(&sample, &target)
    } else {
        let linked = fs::hard_link(&sample, &target);
        let _ = fs::remove_file(&sample);
        linked
    };
    if let Err(e) = placed {
        let _ = fs::remove_file(&sample);
        return if e.kind() == std::io::ErrorKind::AlreadyExists {
            refuse(
                StatusCode::CONFLICT,
                "conflict",
                "A voice with this name exists; send replace=true to overwrite it",
            )
        } else {
            io_failure("A voice sample cannot be stored", &e)
        };
    }
    // The transcript belongs to the sample: a new sample without one drops the old one.
    match transcript {
        Some(words) => {
            let result = write_temporary(&dir, words.as_bytes()).and_then(|path| {
                let renamed = fs::rename(&path, &text);
                if renamed.is_err() {
                    let _ = fs::remove_file(&path);
                }
                renamed
            });
            if let Err(e) = result {
                return io_failure("A voice transcript cannot be stored", &e);
            }
        }
        None => {
            let _ = fs::remove_file(&text);
        }
    }
    (
        if exists {
            StatusCode::OK
        } else {
            StatusCode::CREATED
        },
        Json(json!({"name": name, "replaced": exists})),
    )
        .into_response()
}

/// `DELETE /v1/audio/voices/{name}`: removes the sample and its transcript.
pub async fn delete(State(app): State<App>, Path(name): Path<String>) -> Response {
    let config = app.config.get();
    let Some(settings) = audio(&config) else {
        return not_configured();
    };
    if !valid_voice_name(&name, settings.voice_name_max_chars) {
        return bad("Invalid voice name");
    }
    let dir = config.voices_dir();
    let known = names(&dir, settings.voice_name_max_chars);
    let Some(found) = known.iter().find(|n| **n == name) else {
        return refuse(StatusCode::NOT_FOUND, "not_found", "Unknown voice");
    };
    if let Err(e) = fs::remove_file(dir.join(format!("{found}.wav"))) {
        return io_failure("A voice sample cannot be removed", &e);
    }
    let _ = fs::remove_file(dir.join(format!("{found}.txt")));
    Json(json!({"name": found, "deleted": true})).into_response()
}

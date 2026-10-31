//! `POST /v1/audio/speech` for models with `kind = "tts"`: the text is read aloud in the voice of a sample from the voice
//! directory. The request is JSON: `model`, `input`, `voice` (a name from `GET /v1/audio/voices`, or empty for the
//! `default_voice` of the model, else the model's own speaker), `response_format` (`wav`, the default, or `pcm`), and optionally `speed`,
//! `language`, `seed` and `stream`. A long text is cut into passages at sentence boundaries, the passages are read one after the
//! other by the same loaded model, and the audio is joined with the pauses of the configuration, so a chapter works.
//! The answer is one WAV file (`audio/wav`), or with `pcm` the raw samples (signed 16 bit little endian, `audio/pcm`;
//! the rate, channels and bits are in the headers `X-Audio-Sample-Rate`, `X-Audio-Channels` and `X-Audio-Bits`).
//!
//! With `"stream": true` the answer is sent while it is made (chunked transfer encoding): the first passage as soon as
//! it is read, the others after it, and the pauses as silence in between. A streamed `wav` has an open length (the RIFF
//! and data sizes are 0xFFFFFFFF, which sox, ffmpeg and mpv read until the end of the stream). The sample rate is the
//! one of the first passage's audio; a later passage with another rate or channel count ends the stream. The status is
//! sent with the first bytes, so an error in a later passage cannot be reported: the stream ends with an aborted
//! connection (no final chunk), which a client sees as a failed transfer and not as a complete file. A client that
//! goes away stops the synthesis of the passages that follow, so the GPU is free again.

use super::{App, Error, error, failure, spec, voices};
use axum::{
    body::{Body, Bytes},
    http::{HeaderName, StatusCode, header},
    response::{IntoResponse, Response},
};
use eliezer_backend::{SpeechRequest, VoiceSample, supervisor::Lease};
use eliezer_core::config::{Config, ModelConfig};
use eliezer_speech::{
    passages::{Passage, passages},
    pcm::{Pcm, read_wav, write_wav},
};
use serde_json::{Value, json};
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::mpsc;

/// How many finished chunks wait for a slow reader. A small number keeps the synthesis from running far ahead of a
/// player (and of a client that went away without the service noticing yet).
const STREAM_BACKLOG: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Wav,
    Pcm,
}

/// The body of a streamed answer; an `Err` item aborts the connection.
struct Chunks(mpsc::Receiver<Result<Bytes, std::io::Error>>);
impl futures_core::Stream for Chunks {
    type Item = Result<Bytes, std::io::Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.poll_recv(cx)
    }
}

/// What every passage of a request shares.
struct Job {
    model: String,
    key: String,
    voice: Option<VoiceSample>,
    language: Option<String>,
    speed: Option<f64>,
    seed: Option<u64>,
}
impl Job {
    /// Reads one passage aloud.
    async fn read(&self, app: &App, lease: &Lease, text: String) -> Result<Pcm, Error> {
        let request = SpeechRequest {
            model: self.model.clone(),
            text,
            voice: self.voice.clone(),
            language: self.language.clone(),
            speed: self.speed,
            seed: self.seed,
        };
        let wav = app
            .supervisor
            .speak(lease, request, app.backend.as_ref())
            .await?;
        read_wav(&wav).map_err(|_| {
            tracing::warn!(model = %self.key, "The speech program answered without usable audio");
            Error::Backend
        })
    }
}

fn pause_ms(limits: &Limits, passage: &Passage) -> u32 {
    if passage.ends_paragraph {
        limits.paragraph_pause_ms
    } else {
        limits.pause_ms
    }
}

/// The samples as bytes (signed 16 bit little endian), followed by `pause_ms` of silence.
fn raw(pcm: &Pcm, pause_ms: u32) -> Vec<u8> {
    let frames = (u64::from(pcm.rate) * u64::from(pause_ms) / 1000) as usize;
    let silence = frames * usize::from(pcm.channels) * 2;
    let mut out = Vec::with_capacity(pcm.samples.len() * 2 + silence);
    for sample in &pcm.samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out.resize(out.len() + silence, 0);
    out
}

/// A WAV header for a stream of unknown length: RIFF and data sizes are 0xFFFFFFFF.
fn open_wav_header(rate: u32, channels: u16) -> Vec<u8> {
    let block = channels * 2;
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&u32::MAX.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&u32::MAX.to_le_bytes());
    out
}

/// The headers that tell a client how to read raw samples.
fn audio_headers(pcm: &Pcm) -> [(HeaderName, String); 3] {
    [
        (
            HeaderName::from_static("x-audio-sample-rate"),
            pcm.rate.to_string(),
        ),
        (
            HeaderName::from_static("x-audio-channels"),
            pcm.channels.to_string(),
        ),
        (HeaderName::from_static("x-audio-bits"), "16".to_owned()),
    ]
}

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
    let format = match body.get("response_format") {
        None | Some(Value::Null) => Format::Wav,
        Some(Value::String(f)) if f == "wav" => Format::Wav,
        Some(Value::String(f)) if f == "pcm" => Format::Pcm,
        Some(_) => return bad("Only response_format wav and pcm are supported"),
    };
    let stream = match body.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(stream)) => *stream,
        Some(_) => return bad("stream must be true or false"),
    };
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
    let default_voice = extra.and_then(|e| e.default_voice.as_deref());
    if requested.is_none()
        && default_voice.is_none()
        && extra.and_then(|e| e.requires_voice) == Some(true)
    {
        let empty = voices::names(
            &config.voices_dir(),
            config
                .backend
                .audiocpp
                .as_ref()
                .map_or(64, |a| a.voice_name_max_chars),
        )
        .is_empty();
        let mut message = format!(
            "The model \"{}\" needs a voice: pass \"voice\" (see GET /v1/audio/voices) or set extra.default_voice in the model's configuration.",
            model.key
        );
        if empty {
            message.push_str(
                " There are no voices yet; add one with POST /v1/audio/voices or \"eliezer voices add\".",
            );
        }
        return bad(&message);
    }
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
    let job = Job {
        model: spec.model.clone(),
        key: model.key.clone(),
        voice,
        language,
        speed,
        seed,
    };
    if stream {
        return stream_answer(app.clone(), lease, job, limits, list, format).await;
    }
    let mut parts = Vec::with_capacity(list.len());
    for passage in list {
        let pause = pause_ms(&limits, &passage);
        match job.read(app, &lease, passage.text).await {
            Ok(pcm) => parts.push((pcm, pause)),
            Err(e) => return error(e, false),
        }
    }
    match eliezer_speech::pcm::join(parts) {
        Ok(joined) => match format {
            Format::Wav => {
                ([(header::CONTENT_TYPE, "audio/wav")], write_wav(&joined)).into_response()
            }
            Format::Pcm => (
                [(header::CONTENT_TYPE, "audio/pcm")],
                audio_headers(&joined),
                raw(&joined, 0),
            )
                .into_response(),
        },
        Err(_) => {
            tracing::warn!(model = %model.key, "The passages differ in sample rate or channels");
            error(Error::Backend, false)
        }
    }
}

/// The streamed answer. The first passage is read here, so that its failure is still a proper error answer; the
/// rest is read by a task that feeds the body.
///
/// A later passage with another sample rate or channel count ends the stream with an error instead of being
/// resampled: one model has one output rate, so a change means that the speech program is misbehaving, and silently
/// converting would hide that (the non-streamed answer refuses such a mix as well, with a 502).
async fn stream_answer(
    app: App,
    lease: Lease,
    job: Job,
    limits: Limits,
    list: Vec<Passage>,
    format: Format,
) -> Response {
    let mut list = list.into_iter().peekable();
    let Some(first) = list.next() else {
        return bad("input has no text to read");
    };
    let pause = if list.peek().is_some() {
        pause_ms(&limits, &first)
    } else {
        0
    };
    let first_pcm = match job.read(&app, &lease, first.text).await {
        Ok(pcm) => pcm,
        Err(e) => return error(e, false),
    };
    let (tx, rx) = mpsc::channel(STREAM_BACKLOG);
    let mut head = if format == Format::Wav {
        open_wav_header(first_pcm.rate, first_pcm.channels)
    } else {
        Vec::new()
    };
    head.extend_from_slice(&raw(&first_pcm, pause));
    // The channel is empty, so this cannot fail.
    let _ = tx.try_send(Ok(Bytes::from(head)));
    let headers = audio_headers(&first_pcm);
    let (rate, channels) = (first_pcm.rate, first_pcm.channels);
    tokio::spawn(async move {
        // The lease lives as long as the task: the model stays loaded while the stream runs.
        let lease = lease;
        while let Some(passage) = list.next() {
            let pause = if list.peek().is_some() {
                pause_ms(&limits, &passage)
            } else {
                0
            };
            // A client that goes away drops the body and closes the channel: stop at once, also in the middle of a
            // passage, so that the GPU is free for the next job.
            let result = tokio::select! {
                () = tx.closed() => {
                    tracing::info!(model = %job.key, "The client went away; speech synthesis stopped");
                    return;
                }
                result = job.read(&app, &lease, passage.text) => result,
            };
            let failure = match result {
                Ok(pcm) if pcm.rate == rate && pcm.channels == channels => {
                    if tx.send(Ok(Bytes::from(raw(&pcm, pause)))).await.is_err() {
                        return;
                    }
                    continue;
                }
                Ok(_) => {
                    tracing::warn!(model = %job.key, "A passage differs in sample rate or channels; the stream is aborted");
                    "A passage differs in sample rate or channels"
                }
                Err(e) => {
                    tracing::warn!(model = %job.key, error = ?e, "A passage failed; the stream is aborted");
                    "A passage could not be read aloud"
                }
            };
            // The status is sent already. An error item aborts the connection without the closing chunk, so the
            // client sees a failed transfer and not a file that is complete.
            let _ = tx.send(Err(std::io::Error::other(failure))).await;
            return;
        }
    });
    let kind = match format {
        Format::Wav => "audio/wav",
        Format::Pcm => "audio/pcm",
    };
    (
        [
            (header::CONTENT_TYPE, kind),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        [(HeaderName::from_static("x-accel-buffering"), "no")],
        headers,
        Body::from_stream(Chunks(rx)),
    )
        .into_response()
}

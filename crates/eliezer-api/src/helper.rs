//! The prompt helper of an image model: a small language model that turns a short wish ("a sunflower") into the
//! detailed description image models draw well. It is a configured model (`extra.prompt_helper`), runs on the CPU
//! when a RAM budget exists (the image model keeps the GPU) and is only an aid: when it fails, the original prompt
//! is used.

use super::App;
use eliezer_backend::{ChatRequest, Chunk};
use eliezer_core::config::DevicePolicy;
use serde_json::json;
use std::time::Duration;

/// How long loading the helper and writing one prompt may take in all.
const LIMIT: Duration = Duration::from_secs(180);
/// The longest prompt that is passed on, in characters.
const MAX_PROMPT: usize = 1200;

pub const SYSTEM: &str = "You write prompts for an image generation model. Turn the user's short wish, in any language, \
into ONE detailed English image description of at most 80 words: subject, setting, lighting, mood, style and camera or \
painting medium. Keep every detail the user names (objects, colors, numbers, places) and add concrete visual details. \
Do not add text, captions or watermarks to the picture. Output only the description, no introduction and no comment.";

/// The description as it is passed on: one line without a label or quotes, not too long; `None` when nothing is left.
pub fn clean(text: &str) -> Option<String> {
    let mut lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    // "Here is the prompt:" in front of the real text.
    if lines.len() > 1 && lines[0].ends_with(':') && lines[0].len() < 60 {
        lines.remove(0);
    }
    // Bold markers are noise in a prompt.
    let joined = lines.join(" ").replace('*', "");
    let mut text = joined.trim();
    for label in ["Prompt:", "Description:", "prompt:"] {
        if let Some(rest) = text.strip_prefix(label) {
            text = rest.trim();
        }
    }
    let text = text
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | '*' | '“' | '”'))
        .trim();
    if text.is_empty() {
        return None;
    }
    Some(text.chars().take(MAX_PROMPT).collect())
}

/// Asks the helper model `key` for a description of `wish`.
pub async fn revise(app: &App, key: &str, wish: &str) -> Option<String> {
    let config = app.config.get();
    let model = config.model.iter().find(|m| m.key == key)?;
    let context = model.default_context;
    let mut spec = eliezer_backend::LoadSpec {
        model: key.to_owned(),
        context,
        parallel: model.parallel,
        bytes: eliezer_core::memory::estimate(model, context, model.parallel, None, None).ok()?,
        device: model.device,
    };
    // The image model takes the GPU; the helper writes on the CPU when there is room for it.
    if config.cpu.ram_budget_gb > 0.0 {
        spec.device = DevicePolicy::Cpu;
    }
    let work = async {
        let lease = app.supervisor.acquire(spec.clone()).await.ok()?;
        let body = json!({
            "model": key,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": wish},
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
        });
        let request = ChatRequest {
            model: key.to_owned(),
            input: body.to_string(),
            reasoning: None,
        };
        let mut chunks = app
            .supervisor
            .chat(&lease, request, app.backend.as_ref())
            .await
            .ok()?;
        let mut text = String::new();
        while let Some(chunk) = chunks.recv().await {
            match chunk {
                Ok(Chunk::Text(piece)) => text.push_str(&piece),
                Ok(Chunk::Done { .. }) => break,
                Ok(_) => {}
                Err(_) => return None,
            }
        }
        clean(&text)
    };
    match tokio::time::timeout(LIMIT, work).await {
        Ok(Some(text)) => Some(text),
        _ => {
            tracing::warn!(event = "prompt_helper_failed", model = %key, "The prompt helper gave no description; the original prompt is used");
            None
        }
    }
}

//! Images in chat requests: turns the OpenAI content parts into what llama.cpp's multimodal
//! library wants, a plain text prompt with one marker per image plus the decoded image bytes.
//!
//! Only `data:` URIs are accepted. A remote URL is never fetched: the service would become a proxy
//! for requests to any address the machine can reach.

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

/// Images per request.
pub const MAX_IMAGES: usize = 8;
/// Decoded size of one image.
pub const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
/// Decoded size of all images of a request.
pub const MAX_TOTAL_BYTES: usize = 40 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub enum VisionError {
    /// An `image_url` that is not a base64 `data:` URI.
    NotInline,
    /// The base64 text cannot be decoded.
    BadData,
    TooMany,
    TooLarge,
    /// A content part of a type other than text or image.
    UnsupportedPart,
}
impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotInline => "Images must be sent inline as base64 data: URIs, not as links",
            Self::BadData => "The image data is not valid base64",
            Self::TooMany => "Too many images in one request",
            Self::TooLarge => "An image is too large",
            Self::UnsupportedPart => "Only text and image parts are supported in message content",
        })
    }
}
impl std::error::Error for VisionError {}

/// The content parts of a message that are images.
fn is_image(part: &Value) -> bool {
    matches!(
        part["type"].as_str(),
        Some("image_url" | "input_image" | "image")
    )
}

/// True when any message carries an image part.
pub fn has_images(messages: &[Value]) -> bool {
    messages.iter().any(|m| {
        m["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(is_image))
    })
}

/// The messages with every list content flattened to a string (the marker stands where an image
/// was), and the decoded images in the order of their markers.
pub struct Extracted {
    pub messages: Vec<Value>,
    pub images: Vec<Vec<u8>>,
}

fn decode(url: &str) -> Result<Vec<u8>, VisionError> {
    let rest = url.strip_prefix("data:").ok_or(VisionError::NotInline)?;
    let (meta, data) = rest.split_once(',').ok_or(VisionError::NotInline)?;
    if !meta.ends_with(";base64") {
        return Err(VisionError::NotInline);
    }
    // A decoded image is at most three quarters of its text; refuse before decoding.
    if data.len() / 4 * 3 > MAX_IMAGE_BYTES + 3 {
        return Err(VisionError::TooLarge);
    }
    let bytes = STANDARD
        .decode(data.trim())
        .map_err(|_| VisionError::BadData)?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::TooLarge);
    }
    Ok(bytes)
}

/// The URL of an image part: `image_url.url`, a plain `image_url` string, or the base64 `source`
/// of an Anthropic-style `image` part.
fn url_of(part: &Value) -> Option<String> {
    if let Some(url) = part["image_url"]["url"]
        .as_str()
        .or(part["image_url"].as_str())
    {
        return Some(url.to_owned());
    }
    let source = &part["source"];
    if source["type"] == "base64" {
        let media = source["media_type"].as_str().unwrap_or("image/png");
        return Some(format!("data:{media};base64,{}", source["data"].as_str()?));
    }
    None
}

pub fn extract(messages: &[Value], marker: &str) -> Result<Extracted, VisionError> {
    let mut images: Vec<Vec<u8>> = Vec::new();
    let mut total = 0_usize;
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let Some(parts) = message["content"].as_array() else {
            out.push(message.clone());
            continue;
        };
        let mut text = String::new();
        for part in parts {
            if is_image(part) {
                if images.len() == MAX_IMAGES {
                    return Err(VisionError::TooMany);
                }
                let bytes = decode(&url_of(part).ok_or(VisionError::NotInline)?)?;
                total = total.saturating_add(bytes.len());
                if total > MAX_TOTAL_BYTES {
                    return Err(VisionError::TooLarge);
                }
                images.push(bytes);
                text.push_str(marker);
            } else if part["type"] == "text" {
                text.push_str(part["text"].as_str().unwrap_or_default());
            } else {
                return Err(VisionError::UnsupportedPart);
            }
        }
        let mut flat = message.clone();
        flat["content"] = json!(text);
        out.push(flat);
    }
    Ok(Extracted {
        messages: out,
        images,
    })
}

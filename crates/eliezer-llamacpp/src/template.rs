//! Renders the Jinja chat template that ships inside a GGUF file.
//!
//! The llama.cpp binding only knows a basic template without tools. Hugging Face templates
//! are written for Python Jinja2, so the environment mimics its behaviour: `trim_blocks`,
//! `lstrip_blocks`, Python string methods (pycompat), `raise_exception`, `strftime_now` and a
//! `tojson` filter with Python's separators and without HTML escaping.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::value::{Kwargs, Value};
use minijinja::{Environment, Error, ErrorKind};
use serde::Serialize;
use serde_json::{Map, Value as Json};

const TEMPLATE_NAME: &str = "chat";
/// Marks errors created by `raise_exception` so that they can be told from engine errors.
const RAISE_PREFIX: &str = "\u{1}raise:";

/// Errors while compiling or rendering a chat template.
#[derive(Debug)]
pub enum TemplateError {
    /// The template source does not compile.
    Compile(String),
    /// Rendering failed in the engine (type error, missing value, ...).
    Render(String),
    /// The template itself rejected the input through `raise_exception`.
    Raised(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(msg) => write!(f, "chat template does not compile: {msg}"),
            Self::Render(msg) => write!(f, "chat template could not be rendered: {msg}"),
            Self::Raised(msg) => write!(f, "chat template rejected the conversation: {msg}"),
        }
    }
}

impl std::error::Error for TemplateError {}

fn map_error(error: Error, compile: bool) -> TemplateError {
    if let Some(msg) = error.detail().and_then(|d| d.strip_prefix(RAISE_PREFIX)) {
        return TemplateError::Raised(msg.to_string());
    }
    if compile {
        TemplateError::Compile(error.to_string())
    } else {
        TemplateError::Render(error.to_string())
    }
}

/// Inputs of one rendering.
pub struct RenderOptions<'a> {
    /// OpenAI chat messages.
    pub messages: &'a [Json],
    /// OpenAI tools array.
    pub tools: Option<&'a Json>,
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<&'a str>,
}

/// The text of a message content: a string, or the text parts of a list.
fn content_text(content: &Json) -> String {
    match content {
        Json::String(text) => text.clone(),
        Json::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The conversation without system messages: their text goes, separated by a blank line, in front of
/// the first user message (which becomes a user message of its own when there is none). `None` when
/// there is no system message to merge.
pub fn merge_system_into_user(messages: &[Json]) -> Option<Vec<Json>> {
    let is_system = |m: &Json| matches!(m["role"].as_str(), Some("system" | "developer"));
    let system: Vec<String> = messages
        .iter()
        .filter(|m| is_system(m))
        .map(|m| content_text(&m["content"]))
        .filter(|t| !t.is_empty())
        .collect();
    if !messages.iter().any(is_system) {
        return None;
    }
    let prefix = system.join("\n\n");
    let mut out: Vec<Json> = Vec::with_capacity(messages.len());
    let mut placed = prefix.is_empty();
    for message in messages.iter().filter(|m| !is_system(m)) {
        let mut message = message.clone();
        if !placed && message["role"] == "user" {
            message["content"] = match &message["content"] {
                Json::Array(parts) => {
                    let mut parts = parts.clone();
                    parts.insert(
                        0,
                        serde_json::json!({"type": "text", "text": format!("{prefix}\n\n")}),
                    );
                    Json::Array(parts)
                }
                other => Json::String(format!("{prefix}\n\n{}", content_text(other))),
            };
            placed = true;
        }
        out.push(message);
    }
    if !placed {
        out.insert(0, serde_json::json!({"role": "user", "content": prefix}));
    }
    Some(out)
}

/// A compiled chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
    uses_tools: bool,
}

impl ChatTemplate {
    /// Compiles `source`; `bos_token` and `eos_token` are the texts of the model's special tokens.
    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<Value, Error> {
                Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("{RAISE_PREFIX}{message}"),
                ))
            },
        );
        env.add_function("strftime_now", |format: String| -> String {
            let seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            strftime_utc(&format, seconds)
        });
        env.add_filter("tojson", tojson);
        env.add_template_owned(TEMPLATE_NAME, source.to_string())
            .map_err(|e| map_error(e, true))?;
        Ok(Self {
            env,
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
            uses_tools: source.contains("tools"),
        })
    }

    /// Whether the template source refers to `tools` and can therefore render tool definitions.
    pub fn supports_tools(&self) -> bool {
        self.uses_tools
    }

    /// Like [`ChatTemplate::render`], but a template that rejects the system role (Mistral 7B v0.3 raises
    /// "Only user and assistant roles are supported!") gets the system prompt in front of the first user
    /// message instead, which is what such models were trained on. Other rejections stay errors.
    pub fn render_lenient(&self, options: &RenderOptions<'_>) -> Result<String, TemplateError> {
        let first = self.render(options);
        if !matches!(first, Err(TemplateError::Raised(_))) {
            return first;
        }
        let Some(merged) = merge_system_into_user(options.messages) else {
            return first;
        };
        self.render(&RenderOptions {
            messages: &merged,
            tools: options.tools,
            add_generation_prompt: options.add_generation_prompt,
            enable_thinking: options.enable_thinking,
            reasoning_effort: options.reasoning_effort,
        })
        .or(first)
    }

    /// Renders the conversation into the prompt text.
    pub fn render(&self, options: &RenderOptions<'_>) -> Result<String, TemplateError> {
        let mut vars = Map::new();
        vars.insert("bos_token".into(), Json::String(self.bos_token.clone()));
        vars.insert("eos_token".into(), Json::String(self.eos_token.clone()));
        vars.insert(
            "messages".into(),
            Json::Array(options.messages.iter().map(normalize_message).collect()),
        );
        if let Some(tools) = options.tools {
            vars.insert("tools".into(), tools.clone());
        }
        vars.insert(
            "add_generation_prompt".into(),
            Json::Bool(options.add_generation_prompt),
        );
        if let Some(flag) = options.enable_thinking {
            vars.insert("enable_thinking".into(), Json::Bool(flag));
        }
        if let Some(effort) = options.reasoning_effort {
            vars.insert("reasoning_effort".into(), Json::String(effort.to_string()));
        }
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| map_error(e, false))?;
        template.render(&vars).map_err(|e| map_error(e, false))
    }
}

/// Prepares one OpenAI message for the templates: text parts of an array content are joined
/// with a newline into one string (other part types, e.g. images, are dropped), a null content is
/// removed, and tool-call
/// arguments are turned from a JSON string into an object (kept as a string if they do not parse).
fn normalize_message(message: &Json) -> Json {
    let mut message = message.clone();
    let Some(object) = message.as_object_mut() else {
        return message;
    };
    // A null content (assistant message that only carries tool calls) is removed: Python templates
    // test `"x" in message.content` after `"content" in message`, which fails for null.
    if object.get("content").is_some_and(Json::is_null) {
        object.remove("content");
    }
    if let Some(Json::Array(parts)) = object.get("content") {
        let texts: Vec<&str> = parts
            .iter()
            .filter(|p| {
                p.get("type")
                    .and_then(Json::as_str)
                    .is_none_or(|t| t == "text")
            })
            .filter_map(|p| p.get("text").and_then(Json::as_str))
            .collect();
        let joined = texts.join("\n");
        object.insert("content".into(), Json::String(joined));
    }
    if let Some(Json::Array(calls)) = object.get_mut("tool_calls") {
        for call in calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let parsed = function
                .get("arguments")
                .and_then(Json::as_str)
                .and_then(|s| serde_json::from_str::<Json>(s).ok());
            if let (Some(parsed), Some(function)) = (parsed, function.as_object_mut()) {
                function.insert("arguments".into(), parsed);
            }
        }
    }
    message
}

/// `tojson` as Hugging Face defines it: Python separators (`", "` and `": "`), no ASCII
/// escaping, no HTML escaping, optional `indent`.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    // Other keyword arguments (ensure_ascii, sort_keys, ...) are accepted and ignored.
    let _ = kwargs.get::<Option<Value>>("ensure_ascii")?;
    let _ = kwargs.get::<Option<Value>>("sort_keys")?;
    let _ = kwargs.get::<Option<Value>>("separators")?;
    let mut buffer = Vec::new();
    let result = match indent {
        Some(width) => {
            let pad = vec![b' '; width];
            let formatter = serde_json::ser::PrettyFormatter::with_indent(&pad);
            let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, formatter);
            value.serialize(&mut serializer)
        }
        None => {
            let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, PyFormatter);
            value.serialize(&mut serializer)
        }
    };
    result.map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson failed: {e}")))?;
    let text = String::from_utf8(buffer)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson failed: {e}")))?;
    Ok(Value::from_safe_string(text))
}

/// Compact JSON with the spacing of Python's `json.dumps` default.
struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Minimal `strftime` for the current time in UTC. Supports `%Y %y %m %d %e %H %M %S %j %B %b
/// %A %a %F %T %%`; unknown specifiers are copied unchanged.
fn strftime_utc(format: &str, epoch_seconds: i64) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    let days = epoch_seconds.div_euclid(86_400);
    let secs = epoch_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let weekday = (days + 4).rem_euclid(7) as usize;
    let (hour, minute, second) = (secs / 3600, secs % 3600 / 60, secs % 60);
    let day_of_year = days - days_from_civil(year, 1, 1) + 1;
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&format!("{year:04}")),
            Some('y') => out.push_str(&format!("{:02}", year.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('e') => out.push_str(&format!("{day:2}")),
            Some('H') => out.push_str(&format!("{hour:02}")),
            Some('M') => out.push_str(&format!("{minute:02}")),
            Some('S') => out.push_str(&format!("{second:02}")),
            Some('j') => out.push_str(&format!("{day_of_year:03}")),
            Some('B') => out.push_str(MONTHS[(month - 1) as usize]),
            Some('b') => out.push_str(&MONTHS[(month - 1) as usize][..3]),
            Some('A') => out.push_str(DAYS[weekday]),
            Some('a') => out.push_str(&DAYS[weekday][..3]),
            Some('F') => out.push_str(&format!("{year:04}-{month:02}-{day:02}")),
            Some('T') => out.push_str(&format!("{hour:02}:{minute:02}:{second:02}")),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Inverse of [`civil_from_days`].
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::strftime_utc;

    #[test]
    fn formats_known_instant() {
        // 2024-02-29 13:05:09 UTC, a Thursday.
        let t = 1_709_211_909;
        assert_eq!(strftime_utc("%Y-%m-%d", t), "2024-02-29");
        assert_eq!(
            strftime_utc("%A %d %B %Y %T", t),
            "Thursday 29 February 2024 13:05:09"
        );
        assert_eq!(strftime_utc("%j %% %q", t), "060 % %q");
    }
}

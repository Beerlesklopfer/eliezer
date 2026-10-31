//! Parses the raw output of a model into text, thinking and tool calls.
//!
//! The formats mirror what the chat templates print for assistant turns in the history, because
//! that is what the models were trained to produce:
//!
//! - `Think` and `QwenXml`: `<think>\n...\n</think>\n\nanswer`; Qwen tool calls are
//!   `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`.
//! - `Harmony` (gpt-oss): messages `<|channel|>CHANNEL[ to=functions.NAME][ <|constrain|>json]
//!   <|message|>BODY<|end||call|return>`; channel `analysis` is thinking, `final` is the answer,
//!   `commentary` with a `functions.NAME` recipient is a tool call whose body is the JSON arguments.
//! - `Gemma4`: `<|channel>thought\n...\n<channel|>answer` and
//!   `<|tool_call>call:NAME{key:value,...}<tool_call|>`, with strings in `<|"|>...<|"|>`.
//!
//! Streaming rule: only a possible prefix of a marker is held back, marker text is never emitted.

use eliezer_backend::Chunk;
use serde_json::{Map, Value};

/// Output syntax of a model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Plain,
    Think,
    QwenXml,
    Harmony,
    Gemma4,
}

/// Picks the output format from markers in the chat template source.
pub fn detect_format(template_source: &str) -> Format {
    if template_source.contains("<|channel|>") {
        Format::Harmony
    } else if template_source.contains("<|channel>") {
        Format::Gemma4
    } else if template_source.contains("<function=") {
        Format::QwenXml
    } else if template_source.contains("<think>") {
        Format::Think
    } else {
        Format::Plain
    }
}

const HARMONY_END: [&str; 3] = ["<|end|>", "<|call|>", "<|return|>"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Thinking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    /// Answer text (all formats).
    Text,
    /// Inside a thinking block.
    Thinking,
    /// Inside a Qwen or Gemma tool call; the body stays in the buffer until the end marker.
    Tool,
    /// Harmony: between messages.
    HarmonyIdle,
    /// Harmony: header of a message, up to `<|message|>`.
    HarmonyHeader,
    /// Harmony: streaming body of an analysis or answer message.
    HarmonyBody(Kind),
    /// Harmony: body of a tool call to the named function.
    HarmonyTool(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Skip {
    None,
    /// Remove one newline (directly after an opening marker).
    One,
    /// Remove all newlines (between a closing marker and the answer).
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    OpenThink,
    OpenTool,
    /// Silently dropped (Gemma prints it where it expects the tool result).
    Drop,
}

/// Incremental parser; feed pieces with [`OutputParser::push`] and end with [`OutputParser::finish`].
#[derive(Debug)]
pub struct OutputParser {
    format: Format,
    state: State,
    buf: String,
    skip: Skip,
    tool_count: usize,
}

impl OutputParser {
    /// `thinking_prefilled`: the generation prompt ended inside an open thinking block, so the
    /// output starts as thinking and the first closing marker ends it.
    pub fn new(format: Format, thinking_prefilled: bool) -> Self {
        let state = match format {
            Format::Harmony => State::HarmonyIdle,
            Format::Plain => State::Text,
            _ if thinking_prefilled => State::Thinking,
            _ => State::Text,
        };
        Self {
            format,
            state,
            buf: String::new(),
            skip: Skip::None,
            tool_count: 0,
        }
    }

    /// Whether at least one complete tool call has been emitted.
    pub fn saw_tool_call(&self) -> bool {
        self.tool_count > 0
    }

    /// Feeds the next piece of model output and returns what is unambiguous so far.
    pub fn push(&mut self, piece: &str) -> Vec<Chunk> {
        self.buf.push_str(piece);
        self.run(false)
    }

    /// Ends the output: held-back text is flushed, an unterminated tool call is dropped.
    pub fn finish(&mut self) -> Vec<Chunk> {
        let out = self.run(true);
        self.buf.clear();
        out
    }

    fn run(&mut self, last: bool) -> Vec<Chunk> {
        let mut out = Vec::new();
        while self.step(&mut out, last) {}
        out
    }

    /// Markers that end answer text, with what they mean.
    fn text_markers(&self) -> &'static [(&'static str, Marker)] {
        match self.format {
            Format::Plain | Format::Harmony => &[],
            Format::Think => &[("<think>", Marker::OpenThink)],
            Format::QwenXml => &[
                ("<think>", Marker::OpenThink),
                ("<tool_call>", Marker::OpenTool),
            ],
            Format::Gemma4 => &[
                ("<|channel>thought", Marker::OpenThink),
                ("<|tool_call>", Marker::OpenTool),
                ("<|tool_response>", Marker::Drop),
            ],
        }
    }

    fn think_close(&self) -> &'static str {
        match self.format {
            Format::Gemma4 => "<channel|>",
            _ => "</think>",
        }
    }

    fn tool_close(&self) -> &'static str {
        match self.format {
            Format::Gemma4 => "<tool_call|>",
            _ => "</tool_call>",
        }
    }

    /// Does the buffer start with newlines that must be removed? Returns false if nothing is left.
    fn apply_skip(&mut self) -> bool {
        match self.skip {
            Skip::None => {}
            Skip::One => {
                if self.buf.is_empty() {
                    return false;
                }
                if self.buf.starts_with('\n') {
                    self.buf.remove(0);
                }
                self.skip = Skip::None;
            }
            Skip::All => {
                let trimmed = self.buf.trim_start_matches('\n').len();
                let cut = self.buf.len() - trimmed;
                self.buf.drain(..cut);
                if self.buf.is_empty() {
                    return false;
                }
                self.skip = Skip::None;
            }
        }
        !self.buf.is_empty()
    }

    fn step(&mut self, out: &mut Vec<Chunk>, last: bool) -> bool {
        match self.state.clone() {
            State::Text => self.step_text(out, last),
            State::Thinking => self.step_thinking(out, last),
            State::Tool => self.step_tool(out, last),
            State::HarmonyIdle => self.step_harmony_idle(last),
            State::HarmonyHeader => self.step_harmony_header(last),
            State::HarmonyBody(kind) => self.step_harmony_body(out, kind, last),
            State::HarmonyTool(name) => self.step_harmony_tool(out, &name, last),
        }
    }

    fn step_text(&mut self, out: &mut Vec<Chunk>, last: bool) -> bool {
        if !self.apply_skip() {
            return false;
        }
        let markers = self.text_markers();
        // Qwen templates print "\n\n" between the answer and a tool call; that is layout, not text.
        let trim_before_tool = self.format == Format::QwenXml;
        if let Some((pos, len, marker)) = find_marker(&self.buf, markers.iter().map(|m| (m.0, m.1)))
        {
            let mut before = &self.buf[..pos];
            if trim_before_tool && marker == Marker::OpenTool {
                before = before.trim_end_matches('\n');
            }
            emit(out, Kind::Text, before);
            self.buf.drain(..pos + len);
            match marker {
                Marker::OpenThink => {
                    self.state = State::Thinking;
                    self.skip = Skip::One;
                }
                Marker::OpenTool => self.state = State::Tool,
                Marker::Drop => {}
            }
            return true;
        }
        let held = if last {
            0
        } else {
            held_prefix_len(&self.buf, markers.iter().map(|m| m.0))
        };
        let mut cut = self.buf.len() - held;
        if trim_before_tool && !last {
            cut = self.buf[..cut].trim_end_matches('\n').len();
        }
        emit(out, Kind::Text, &self.buf[..cut]);
        self.buf.drain(..cut);
        false
    }

    fn step_thinking(&mut self, out: &mut Vec<Chunk>, last: bool) -> bool {
        if !self.apply_skip() {
            return false;
        }
        let close = self.think_close();
        // One newline directly before the closing marker belongs to the layout.
        let with_newline = format!("\n{close}");
        let markers = [(with_newline.as_str(), ()), (close, ())];
        if let Some((pos, len, ())) = find_marker(&self.buf, markers.iter().copied()) {
            emit(out, Kind::Thinking, &self.buf[..pos]);
            self.buf.drain(..pos + len);
            self.state = State::Text;
            // The templates print "\n\n" between "</think>" and the answer.
            self.skip = if self.format == Format::Gemma4 {
                Skip::None
            } else {
                Skip::All
            };
            return true;
        }
        let held = if last {
            0
        } else {
            held_prefix_len(&self.buf, markers.iter().map(|m| m.0))
        };
        let cut = self.buf.len() - held;
        emit(out, Kind::Thinking, &self.buf[..cut]);
        self.buf.drain(..cut);
        false
    }

    fn step_tool(&mut self, out: &mut Vec<Chunk>, last: bool) -> bool {
        let close = self.tool_close();
        if let Some(pos) = self.buf.find(close) {
            let body: String = self.buf.drain(..pos + close.len()).collect();
            let body = &body[..pos];
            let parsed = match self.format {
                Format::Gemma4 => parse_gemma_call(body),
                _ => parse_qwen_call(body),
            };
            match parsed {
                Some((name, arguments)) => self.emit_tool(out, name, arguments),
                None => tracing::warn!("Tool call with unreadable body dropped"),
            }
            self.state = State::Text;
            return true;
        }
        if last {
            tracing::warn!("Unterminated tool call dropped at the end of the output");
            self.buf.clear();
        }
        false
    }

    fn emit_tool(&mut self, out: &mut Vec<Chunk>, name: String, arguments: String) {
        out.push(Chunk::Tool {
            index: self.tool_count,
            id: Some(format!("call_{}", self.tool_count)),
            name: Some(name),
            arguments,
        });
        self.tool_count += 1;
    }

    fn step_harmony_idle(&mut self, last: bool) -> bool {
        let markers = [("<|start|>", ()), ("<|channel|>", ())];
        if let Some((pos, _, ())) = find_marker(&self.buf, markers.iter().copied()) {
            // Keep the marker: the header parser reads the whole header including it.
            self.buf.drain(..pos);
            self.state = State::HarmonyHeader;
            return true;
        }
        // Whatever lies between messages carries no content.
        let held = if last {
            0
        } else {
            held_prefix_len(&self.buf, markers.iter().map(|m| m.0))
        };
        let cut = self.buf.len() - held;
        self.buf.drain(..cut);
        false
    }

    fn step_harmony_header(&mut self, last: bool) -> bool {
        let Some(pos) = self.buf.find("<|message|>") else {
            if last {
                tracing::warn!("Unterminated message header dropped at the end of the output");
                self.buf.clear();
            }
            return false;
        };
        let header: String = self.buf.drain(..pos + "<|message|>".len()).collect();
        let header = &header[..pos];
        let channel = harmony_channel(header);
        let recipient = header
            .split(|c: char| c.is_whitespace() || c == '<')
            .find_map(|word| word.strip_prefix("to="));
        self.state = match (channel, recipient) {
            (_, Some(recipient)) => match recipient.strip_prefix("functions.") {
                Some(name) if !name.is_empty() => State::HarmonyTool(name.to_string()),
                _ => {
                    tracing::warn!("Call to the unsupported recipient {recipient} dropped");
                    State::HarmonyTool(String::new())
                }
            },
            ("analysis", None) => State::HarmonyBody(Kind::Thinking),
            _ => State::HarmonyBody(Kind::Text),
        };
        true
    }

    fn step_harmony_body(&mut self, out: &mut Vec<Chunk>, kind: Kind, last: bool) -> bool {
        let markers = HARMONY_END.map(|m| (m, ()));
        if let Some((pos, len, ())) = find_marker(&self.buf, markers.iter().copied()) {
            emit(out, kind, &self.buf[..pos]);
            self.buf.drain(..pos + len);
            self.state = State::HarmonyIdle;
            return true;
        }
        let held = if last {
            0
        } else {
            held_prefix_len(&self.buf, HARMONY_END.iter().copied())
        };
        let cut = self.buf.len() - held;
        emit(out, kind, &self.buf[..cut]);
        self.buf.drain(..cut);
        false
    }

    fn step_harmony_tool(&mut self, out: &mut Vec<Chunk>, name: &str, last: bool) -> bool {
        let markers = HARMONY_END.map(|m| (m, ()));
        if let Some((pos, len, ())) = find_marker(&self.buf, markers.iter().copied()) {
            let arguments = self.buf[..pos].trim();
            let arguments = if arguments.is_empty() {
                "{}".to_string()
            } else {
                arguments.to_string()
            };
            self.buf.drain(..pos + len);
            if !name.is_empty() {
                self.emit_tool(out, name.to_string(), arguments);
            }
            self.state = State::HarmonyIdle;
            return true;
        }
        if last {
            tracing::warn!("Unterminated tool call dropped at the end of the output");
            self.buf.clear();
        }
        false
    }
}

/// Name of the channel in a Harmony header: the word after `<|channel|>`.
fn harmony_channel(header: &str) -> &str {
    header
        .split_once("<|channel|>")
        .map(|(_, rest)| rest)
        .and_then(|rest| rest.split(|c: char| c.is_whitespace() || c == '<').next())
        .unwrap_or("")
}

fn emit(out: &mut Vec<Chunk>, kind: Kind, text: &str) {
    if text.is_empty() {
        return;
    }
    match (out.last_mut(), kind) {
        (Some(Chunk::Text(existing)), Kind::Text)
        | (Some(Chunk::Thinking(existing)), Kind::Thinking) => {
            existing.push_str(text);
        }
        (_, Kind::Text) => out.push(Chunk::Text(text.to_string())),
        (_, Kind::Thinking) => out.push(Chunk::Thinking(text.to_string())),
    }
}

/// Earliest complete marker in `text`: (position, length, payload).
fn find_marker<'a, T: Copy>(
    text: &str,
    markers: impl Iterator<Item = (&'a str, T)>,
) -> Option<(usize, usize, T)> {
    markers
        .filter_map(|(marker, payload)| text.find(marker).map(|pos| (pos, marker.len(), payload)))
        .min_by_key(|(pos, _, _)| *pos)
}

/// Length of the longest suffix of `text` that is a proper prefix of one of the markers.
fn held_prefix_len<'a>(text: &str, markers: impl Iterator<Item = &'a str>) -> usize {
    let mut held = 0;
    for marker in markers {
        let longest = marker.len().saturating_sub(1).min(text.len());
        for len in (held + 1..=longest).rev() {
            if text.is_char_boundary(text.len() - len)
                && marker.is_char_boundary(len)
                && text.ends_with(&marker[..len])
            {
                held = len;
                break;
            }
        }
    }
    held
}

/// Parses a value text of a tool parameter.
///
/// Rule for Qwen parameters: one leading and one trailing newline are removed, then the text is
/// parsed as JSON if it is valid JSON (number, bool, null, object, array or quoted string);
/// otherwise it stays a plain string.
fn qwen_value(raw: &str) -> Value {
    let text = raw.strip_prefix('\n').unwrap_or(raw);
    let text = text.strip_suffix('\n').unwrap_or(text);
    match serde_json::from_str::<Value>(text) {
        Ok(value) => value,
        Err(_) => Value::String(text.to_string()),
    }
}

/// Body between `<tool_call>` and `</tool_call>`: `<function=NAME>` and `<parameter=KEY>` blocks.
fn parse_qwen_call(body: &str) -> Option<(String, String)> {
    let start = body.find("<function=")? + "<function=".len();
    let name_end = body[start..].find('>')? + start;
    let name = body[start..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut rest = &body[name_end + 1..];
    let mut arguments = Map::new();
    while let Some(at) = rest.find("<parameter=") {
        let after = &rest[at + "<parameter=".len()..];
        let key_end = after.find('>')?;
        let key = after[..key_end].trim().to_string();
        let value_part = &after[key_end + 1..];
        let (raw, next) = match value_part.find("</parameter>") {
            Some(end) => (
                &value_part[..end],
                &value_part[end + "</parameter>".len()..],
            ),
            None => {
                // Missing closing tag: the value runs to the end of the function block.
                let end = value_part.find("</function>").unwrap_or(value_part.len());
                (&value_part[..end], &value_part[end..])
            }
        };
        arguments.insert(key, qwen_value(raw));
        rest = next;
    }
    Some((name, Value::Object(arguments).to_string()))
}

const GEMMA_QUOTE: &str = "<|\"|>";

/// Body between `<|tool_call>` and `<tool_call|>`: `call:NAME{key:value,...}`.
fn parse_gemma_call(body: &str) -> Option<(String, String)> {
    let rest = body.trim().strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = rest[..brace].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut cursor = Cursor {
        text: &rest[brace + 1..],
    };
    let arguments = cursor.object()?;
    Some((name, Value::Object(arguments).to_string()))
}

/// Reader for the Gemma value syntax: strings in `<|"|>`, bare numbers/bools/null, `{k:v}`, `[v]`.
struct Cursor<'a> {
    text: &'a str,
}

impl Cursor<'_> {
    fn eat(&mut self, prefix: &str) -> bool {
        match self.text.strip_prefix(prefix) {
            Some(rest) => {
                self.text = rest;
                true
            }
            None => false,
        }
    }

    /// After the opening brace: `key:value` pairs up to and including the closing brace.
    fn object(&mut self) -> Option<Map<String, Value>> {
        let mut map = Map::new();
        if self.eat("}") {
            return Some(map);
        }
        loop {
            let key = if self.eat(GEMMA_QUOTE) {
                let end = self.text.find(GEMMA_QUOTE)?;
                let key = self.text[..end].to_string();
                self.text = &self.text[end + GEMMA_QUOTE.len()..];
                key
            } else {
                let end = self.text.find(':')?;
                let key = self.text[..end].trim().to_string();
                self.text = &self.text[end..];
                key
            };
            if !self.eat(":") {
                return None;
            }
            map.insert(key, self.value()?);
            if self.eat(",") {
                continue;
            }
            return self.eat("}").then_some(map);
        }
    }

    fn value(&mut self) -> Option<Value> {
        if self.eat(GEMMA_QUOTE) {
            let end = self.text.find(GEMMA_QUOTE)?;
            let value = self.text[..end].to_string();
            self.text = &self.text[end + GEMMA_QUOTE.len()..];
            return Some(Value::String(value));
        }
        if self.eat("{") {
            return self.object().map(Value::Object);
        }
        if self.eat("[") {
            let mut items = Vec::new();
            if self.eat("]") {
                return Some(Value::Array(items));
            }
            loop {
                items.push(self.value()?);
                if self.eat(",") {
                    continue;
                }
                return self.eat("]").then_some(Value::Array(items));
            }
        }
        let end = self.text.find([',', '}', ']']).unwrap_or(self.text.len());
        let literal = self.text[..end].trim();
        self.text = &self.text[end..];
        Some(
            serde_json::from_str::<Value>(literal)
                .unwrap_or_else(|_| Value::String(literal.to_string())),
        )
    }
}

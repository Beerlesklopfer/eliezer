//! M4: the output parser turns raw model output into text, thinking and tool calls.
//!
//! The samples mirror how the fixture templates print assistant turns and tool calls in the
//! history, which is the format the models were trained on.

use eliezer_backend::Chunk;
use eliezer_llamacpp::output::{Format, OutputParser, detect_format};

fn text(s: &str) -> Chunk {
    Chunk::Text(s.to_string())
}

fn think(s: &str) -> Chunk {
    Chunk::Thinking(s.to_string())
}

fn tool(index: usize, name: &str, arguments: &str) -> Chunk {
    Chunk::Tool {
        index,
        id: Some(format!("call_{index}")),
        name: Some(name.to_string()),
        arguments: arguments.to_string(),
    }
}

/// Joins adjacent chunks of the same kind so that different split points compare equal.
fn merge(chunks: Vec<Chunk>) -> Vec<Chunk> {
    let mut out: Vec<Chunk> = Vec::new();
    for chunk in chunks {
        match (out.last_mut(), chunk) {
            (Some(Chunk::Text(a)), Chunk::Text(b)) => a.push_str(&b),
            (Some(Chunk::Thinking(a)), Chunk::Thinking(b)) => a.push_str(&b),
            (_, other) => out.push(other),
        }
    }
    out
}

fn feed(format: Format, prefilled: bool, pieces: &[&str]) -> (Vec<Chunk>, bool) {
    let mut parser = OutputParser::new(format, prefilled);
    let mut chunks = Vec::new();
    for piece in pieces {
        chunks.extend(parser.push(piece));
    }
    chunks.extend(parser.finish());
    (merge(chunks), parser.saw_tool_call())
}

/// Deterministic linear congruential generator (no extra dependency).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

fn random_split(input: &str, seed: u64) -> Vec<&str> {
    let mut lcg = Lcg(seed);
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < input.len() {
        let mut end = (start + 1 + (lcg.next() % 9) as usize).min(input.len());
        while !input.is_char_boundary(end) {
            end += 1;
        }
        pieces.push(&input[start..end]);
        start = end;
    }
    pieces
}

fn chars(input: &str) -> Vec<&str> {
    input
        .char_indices()
        .map(|(i, c)| &input[i..i + c.len_utf8()])
        .collect()
}

/// Feeds the input whole, per character and at random split points; all must agree.
fn check(format: Format, prefilled: bool, input: &str, expected: &[Chunk]) {
    check_inner(format, prefilled, input, expected, true);
}

/// Like `check`; `leak_scan` off for inputs that contain marker-like text on purpose.
fn check_inner(format: Format, prefilled: bool, input: &str, expected: &[Chunk], leak_scan: bool) {
    let whole = feed(format, prefilled, &[input]);
    assert_eq!(whole.0, expected, "whole input: {input:?}");
    let per_char = feed(format, prefilled, &chars(input));
    assert_eq!(per_char.0, expected, "per character: {input:?}");
    assert_eq!(per_char.1, whole.1);
    for seed in 1..=40 {
        let pieces = random_split(input, seed);
        let split = feed(format, prefilled, &pieces);
        assert_eq!(split.0, expected, "seed {seed}, pieces {pieces:?}");
        assert_eq!(split.1, whole.1);
    }
    for chunk in whole.0.iter().filter(|_| leak_scan) {
        if let Chunk::Text(t) | Chunk::Thinking(t) = chunk {
            for marker in [
                "<|",
                "|>",
                "<tool_call",
                "</think",
                "<think>",
                "<channel|>",
                "<tool_call|>",
            ] {
                assert!(!t.contains(marker), "marker {marker:?} leaked into {t:?}");
            }
        }
    }
}

#[test]
fn detects_the_format_of_the_fixtures() {
    let dir = |name: &str| {
        std::fs::read_to_string(format!(
            "{}/../../tests/fixtures/chat_templates/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("fixture")
    };
    assert_eq!(
        detect_format(&dir("gpt-oss-20b-MXFP4.jinja")),
        Format::Harmony
    );
    assert_eq!(
        detect_format(&dir("gemma-4-26B-A4B-it-Q4_K_M.jinja")),
        Format::Gemma4
    );
    assert_eq!(
        detect_format(&dir("Qwen3.8-27B-Q4_K_M.jinja")),
        Format::QwenXml
    );
    assert_eq!(
        detect_format(&dir("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.jinja")),
        Format::QwenXml
    );
    assert_eq!(
        detect_format(&dir("Mistral-7B-Instruct-v0.3-Q4_K_M.jinja")),
        Format::Plain
    );
    assert_eq!(
        detect_format(&dir("olmOCR-2-7B-1025-Q4_K_M.jinja")),
        Format::Plain
    );
    assert_eq!(detect_format("<think> only"), Format::Think);
}

#[test]
fn plain_passes_everything_through() {
    let input = "Hello <think>not special</think> and <|channel|> either\n";
    check_inner(Format::Plain, false, input, &[text(input)], false);
    check_inner(Format::Plain, true, input, &[text(input)], false);
}

#[test]
fn think_prefilled_and_explicit() {
    check(
        Format::Think,
        true,
        "Step one.\nStep two.\n</think>\n\nThe answer is 4.",
        &[think("Step one.\nStep two."), text("The answer is 4.")],
    );
    check(
        Format::Think,
        false,
        "<think>\nreason\n</think>\n\nanswer\nmore",
        &[think("reason"), text("answer\nmore")],
    );
    // Thinking disabled: the prompt already contains the empty block, the output is plain.
    check(Format::Think, false, "just text", &[text("just text")]);
    // Only the newline directly before the closing marker is layout.
    check(
        Format::Think,
        true,
        "a\n\n</think>b",
        &[think("a\n"), text("b")],
    );
}

#[test]
fn marker_split_across_pieces() {
    let (chunks, _) = feed(Format::Think, true, &["pre</thi", "nk>post"]);
    assert_eq!(chunks, [think("pre"), text("post")]);
    let (chunks, _) = feed(Format::Think, false, &["a <", "thi", "nk>x\n</", "think>y"]);
    assert_eq!(chunks, [text("a "), think("x"), text("y")]);
    // A look-alike that is not a marker is released as soon as it is clear.
    let (chunks, _) = feed(Format::Think, false, &["a <th", "ing>"]);
    assert_eq!(chunks, [text("a <thing>")]);
}

#[test]
fn text_streams_as_soon_as_unambiguous() {
    let mut parser = OutputParser::new(Format::QwenXml, false);
    assert_eq!(parser.push("Hello world"), [text("Hello world")]);
    assert_eq!(parser.push(" and <tool"), [text(" and ")]);
    assert_eq!(parser.push("x"), [text("<toolx")]);
    let mut parser = OutputParser::new(Format::QwenXml, true);
    assert_eq!(parser.push("thinking so far"), [think("thinking so far")]);
}

const QWEN_CALLS: &str = "I will check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n<parameter=filter>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=quoted>\n\"hi\"\n</parameter>\n<parameter=nothing>\nnull\n</parameter>\n<parameter=code>\n007\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=ping>\n</function>\n</tool_call>";

fn qwen_calls_expected() -> Vec<Chunk> {
    vec![
        text("I will check."),
        tool(
            0,
            "get_weather",
            "{\"city\":\"Paris\",\"days\":3,\"flag\":true,\"note\":\"line one\\nline two\",\"filter\":{\"a\":[1,2]},\"quoted\":\"hi\",\"nothing\":null,\"code\":\"007\"}",
        ),
        tool(1, "ping", "{}"),
    ]
}

#[test]
fn qwen_tool_calls_with_typed_and_string_parameters() {
    // Coder template: no thinking.
    check(Format::QwenXml, false, QWEN_CALLS, &qwen_calls_expected());
    // Qwen3.8: the generation prompt opened a thinking block.
    let input = format!("plan the call\n</think>\n\n{QWEN_CALLS}");
    let mut expected = vec![think("plan the call")];
    expected.extend(qwen_calls_expected());
    check(Format::QwenXml, true, &input, &expected);
}

#[test]
fn qwen_tool_call_without_text_and_saw_tool_call() {
    let input =
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
    check(Format::QwenXml, false, input, &[tool(0, "a", "{\"x\":1}")]);
    assert!(feed(Format::QwenXml, false, &[input]).1);
    assert!(!feed(Format::QwenXml, false, &["no call"]).1);
}

#[test]
fn qwen_unterminated_blocks_at_finish() {
    // Unterminated tool call: dropped, text before it kept, no tool call seen.
    let input = "ok\n\n<tool_call>\n<function=x>\n<parameter=a>\n1";
    check(Format::QwenXml, false, input, &[text("ok")]);
    assert!(!feed(Format::QwenXml, false, &[input]).1);
    // Unterminated thinking is flushed as thinking.
    check(Format::QwenXml, true, "abc\n", &[think("abc\n")]);
    // A marker prefix at the very end is plain text after all.
    check(
        Format::QwenXml,
        false,
        "hello <tool_c",
        &[text("hello <tool_c")],
    );
    // Trailing newlines that no tool call follows are kept.
    check(Format::QwenXml, false, "hello\n\n", &[text("hello\n\n")]);
}

#[test]
fn gemma_thinking_text_and_tool_calls() {
    let input = "<|channel>thought\nI should call.\n<channel|>Sure.<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3,flag:true,nested:{a:[1,<|\"|>x<|\"|>],b:null},note:<|\"|>a, b}<|\"|>}<tool_call|><|tool_call>call:ping{}<tool_call|><|tool_response>";
    check(
        Format::Gemma4,
        false,
        input,
        &[
            think("I should call."),
            text("Sure."),
            tool(
                0,
                "get_weather",
                "{\"city\":\"Paris\",\"days\":3,\"flag\":true,\"nested\":{\"a\":[1,\"x\"],\"b\":null},\"note\":\"a, b}\"}",
            ),
            tool(1, "ping", "{}"),
        ],
    );
    // The prompt ended in an open thought channel (after a tool result with thinking enabled).
    check(
        Format::Gemma4,
        true,
        "reasoning\n<channel|>answer\nline",
        &[think("reasoning"), text("answer\nline")],
    );
    // Thinking disabled: the prompt contains the empty channel; the output is plain text.
    check(
        Format::Gemma4,
        false,
        "plain answer",
        &[text("plain answer")],
    );
    // Unterminated tool call is dropped.
    check(
        Format::Gemma4,
        false,
        "x<|tool_call>call:f{a:1",
        &[text("x")],
    );
}

#[test]
fn harmony_channels_and_tool_calls() {
    // The prompt ends with "<|start|>assistant"; the model continues with the channel.
    check(
        Format::Harmony,
        false,
        "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}<|call|>",
        &[
            think("Need the weather."),
            tool(0, "get_weather", "{\"city\": \"Paris\"}"),
        ],
    );
    check(
        Format::Harmony,
        false,
        "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>final<|message|>Hello!\nBye.<|return|>",
        &[think("hmm"), text("Hello!\nBye.")],
    );
    // History form of a call (as printed by the template), two calls in one answer, one with
    // a preamble on the commentary channel.
    check(
        Format::Harmony,
        false,
        "<|channel|>commentary<|message|>Let me look.<|end|><|start|>assistant to=functions.a<|channel|>commentary json<|message|>{}<|call|><|start|>assistant to=functions.b<|channel|>commentary json<|message|>{\"x\":[1,2]}<|call|>",
        &[
            text("Let me look."),
            tool(0, "a", "{}"),
            tool(1, "b", "{\"x\":[1,2]}"),
        ],
    );
}

#[test]
fn harmony_unterminated_blocks_at_finish() {
    check_inner(
        Format::Harmony,
        false,
        "<|channel|>final<|message|>Hello <|ret",
        &[text("Hello <|ret")],
        false,
    );
    let input = "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{\"a\": ";
    check(Format::Harmony, false, input, &[]);
    assert!(!feed(Format::Harmony, false, &[input]).1);
    check(
        Format::Harmony,
        false,
        "<|channel|>analysis<|message|>thinking on",
        &[think("thinking on")],
    );
}

#[test]
fn multibyte_text_is_never_split() {
    let input = "Grüße ✓ 日本語\n\n<tool_call>\n<function=f>\n<parameter=s>\nÄÖ ✓\n</parameter>\n</function>\n</tool_call>";
    check(
        Format::QwenXml,
        false,
        input,
        &[text("Grüße ✓ 日本語"), tool(0, "f", "{\"s\":\"ÄÖ ✓\"}")],
    );
}

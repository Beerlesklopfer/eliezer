//! M4: the real chat templates of the GGUF models render with minijinja.

use eliezer_llamacpp::template::{ChatTemplate, RenderOptions, TemplateError};
use serde_json::{Value, json};

const QWEN_CODER: &str =
    include_str!("../fixtures/chat_templates/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.jinja");
const QWEN_38: &str = include_str!("../fixtures/chat_templates/Qwen3.8-27B-Q4_K_M.jinja");
const GPT_OSS: &str = include_str!("../fixtures/chat_templates/gpt-oss-20b-MXFP4.jinja");
const GEMMA_4: &str = include_str!("../fixtures/chat_templates/gemma-4-26B-A4B-it-Q4_K_M.jinja");
const MISTRAL: &str =
    include_str!("../fixtures/chat_templates/Mistral-7B-Instruct-v0.3-Q4_K_M.jinja");
const OLMOCR: &str = include_str!("../fixtures/chat_templates/olmOCR-2-7B-1025-Q4_K_M.jinja");

fn compile(source: &str) -> ChatTemplate {
    ChatTemplate::new(source, "<bos>", "<eos>").expect("template compiles")
}

fn options<'a>(messages: &'a [Value]) -> RenderOptions<'a> {
    RenderOptions {
        messages,
        tools: None,
        add_generation_prompt: true,
        enable_thinking: None,
        reasoning_effort: None,
    }
}

fn simple_conversation() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "Hello there"}),
    ]
}

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns the weather <now> & here.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Forecast days"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }])
}

fn tool_conversation() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "Use tools."}),
        json!({"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\",\"days\":3}"}
            }]
        }),
        json!({"role": "tool", "tool_call_id": "call_0", "content": "Sunny, 21 degrees"}),
    ]
}

#[test]
fn all_fixtures_compile() {
    for source in [QWEN_CODER, QWEN_38, GPT_OSS, GEMMA_4, MISTRAL, OLMOCR] {
        compile(source);
    }
}

#[test]
fn broken_source_is_a_compile_error() {
    let error = ChatTemplate::new("{% if %}", "", "")
        .err()
        .expect("must fail");
    assert!(matches!(error, TemplateError::Compile(_)), "{error}");
}

#[test]
fn generation_prompts_carry_the_assistant_opener() {
    let messages = simple_conversation();
    let user_only = &messages[1..];

    let prompt = compile(QWEN_CODER)
        .render(&options(&messages))
        .expect("render");
    assert!(
        prompt.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"),
        "{prompt}"
    );
    assert!(prompt.contains("<|im_start|>user\nHello there<|im_end|>\n"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));

    let prompt = compile(QWEN_38)
        .render(&options(&messages))
        .expect("render");
    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n"),
        "{prompt}"
    );

    let prompt = compile(GPT_OSS)
        .render(&options(&messages))
        .expect("render");
    assert!(
        prompt.starts_with("<|start|>system<|message|>You are ChatGPT"),
        "{prompt}"
    );
    assert!(prompt.contains("<|start|>developer<|message|># Instructions\n\nBe brief.<|end|>"));
    assert!(prompt.contains("<|start|>user<|message|>Hello there<|end|>"));
    assert!(prompt.ends_with("<|start|>assistant"));
    assert!(
        prompt.contains("Current date: 20"),
        "strftime_now must work: {prompt}"
    );

    let prompt = compile(GEMMA_4)
        .render(&options(&messages))
        .expect("render");
    assert!(
        prompt.starts_with("<bos><|turn>system\nBe brief.<turn|>\n"),
        "{prompt}"
    );
    assert!(prompt.contains("<|turn>user\nHello there<turn|>\n"));
    assert!(
        prompt.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
        "{prompt}"
    );

    let prompt = compile(MISTRAL)
        .render(&options(user_only))
        .expect("render");
    assert_eq!(prompt, "<bos>[INST] Hello there [/INST]");

    let prompt = compile(OLMOCR).render(&options(&messages)).expect("render");
    assert!(
        prompt.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"),
        "{prompt}"
    );
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn tool_capable_templates_render_definitions_and_tool_turns() {
    let tools = tools();
    let conversation = tool_conversation();
    for (name, source) in [
        ("qwen-coder", QWEN_CODER),
        ("qwen3.8", QWEN_38),
        ("gpt-oss", GPT_OSS),
        ("gemma-4", GEMMA_4),
    ] {
        let template = compile(source);
        assert!(template.supports_tools(), "{name}");
        let mut opts = options(&conversation);
        opts.tools = Some(&tools);
        let prompt = template
            .render(&opts)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(prompt.contains("get_weather"), "{name}: {prompt}");
        assert!(prompt.contains("Sunny, 21 degrees"), "{name}: {prompt}");
        // Without a definition list the same conversation still renders.
        template
            .render(&options(&conversation))
            .unwrap_or_else(|e| panic!("{name} without tools: {e}"));
    }
}

#[test]
fn tool_calls_are_printed_the_way_each_model_was_trained() {
    let tools = tools();
    let conversation = tool_conversation();
    let render = |source: &str| {
        let mut opts = options(&conversation);
        opts.tools = Some(&tools);
        compile(source).render(&opts).expect("render")
    };

    let prompt = render(QWEN_CODER);
    assert!(prompt.contains(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>"
    ), "{prompt}");
    assert!(prompt.contains("<tool_response>\nSunny, 21 degrees\n</tool_response>"));

    let prompt = render(QWEN_38);
    assert!(prompt.contains(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>"
    ), "{prompt}");
    // tojson keeps Python spacing and does not escape HTML characters.
    assert!(
        prompt.contains("\"description\": \"Returns the weather <now> & here.\""),
        "{prompt}"
    );

    let prompt = render(GPT_OSS);
    assert!(prompt.contains(
        "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Paris\", \"days\": 3}<|call|>"
    ), "{prompt}");
    assert!(prompt.contains(
        "<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>\"Sunny, 21 degrees\"<|end|>"
    ));
    assert!(prompt.contains("type get_weather = (_: {"));

    let prompt = render(GEMMA_4);
    assert!(
        prompt.contains("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>"),
        "{prompt}"
    );
    assert!(prompt.contains(
        "<|tool_response>response:get_weather{value:<|\"|>Sunny, 21 degrees<|\"|>}<tool_response|>"
    ));
    assert!(prompt.contains("<|tool>declaration:get_weather{"));
}

#[test]
fn thinking_switch_changes_qwen_and_gemma_prompts() {
    let messages = simple_conversation();
    for (name, source) in [("qwen3.8", QWEN_38), ("gemma-4", GEMMA_4)] {
        let template = compile(source);
        let mut on = options(&messages);
        on.enable_thinking = Some(true);
        let mut off = options(&messages);
        off.enable_thinking = Some(false);
        let (on, off) = (
            template.render(&on).expect("on"),
            template.render(&off).expect("off"),
        );
        assert_ne!(on, off, "{name}");
    }
    let mut on = options(&messages);
    on.enable_thinking = Some(true);
    let mut off = options(&messages);
    off.enable_thinking = Some(false);
    let template = compile(QWEN_38);
    assert!(template.render(&on).unwrap().ends_with("<think>\n"));
    assert!(
        template
            .render(&off)
            .unwrap()
            .ends_with("<think>\n\n</think>\n\n")
    );
    let template = compile(GEMMA_4);
    assert!(template.render(&on).unwrap().contains("<|think|>"));
    assert!(!template.render(&off).unwrap().contains("<|think|>"));
}

#[test]
fn reasoning_effort_changes_gpt_oss_prompt() {
    let messages = simple_conversation();
    let template = compile(GPT_OSS);
    let mut low = options(&messages);
    low.reasoning_effort = Some("low");
    let mut high = options(&messages);
    high.reasoning_effort = Some("high");
    let (low, high) = (
        template.render(&low).unwrap(),
        template.render(&high).unwrap(),
    );
    assert!(low.contains("Reasoning: low"));
    assert!(high.contains("Reasoning: high"));
    assert_ne!(low, high);
    // Qwen3.8 also reacts to the effort and rejects values it does not know.
    let template = compile(QWEN_38);
    let mut bad = options(&messages);
    bad.reasoning_effort = Some("extreme");
    assert!(matches!(
        template.render(&bad),
        Err(TemplateError::Raised(_))
    ));
}

#[test]
fn simple_templates_ignore_tools() {
    let tools = tools();
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    for source in [MISTRAL, OLMOCR] {
        let template = compile(source);
        assert!(!template.supports_tools());
        let mut opts = options(&messages);
        opts.tools = Some(&tools);
        let with = template.render(&opts).expect("render with tools");
        let without = template.render(&options(&messages)).expect("render");
        assert_eq!(with, without);
    }
}

#[test]
fn raise_exception_surfaces_with_its_message() {
    let template = ChatTemplate::new(
        "{% if messages|length > 1 %}{{ raise_exception('only one message, please') }}{% endif %}ok",
        "",
        "",
    )
    .expect("compile");
    let one = [json!({"role": "user", "content": "a"})];
    assert_eq!(template.render(&options(&one)).expect("render"), "ok");
    let two = [one[0].clone(), one[0].clone()];
    match template.render(&options(&two)) {
        Err(TemplateError::Raised(message)) => assert_eq!(message, "only one message, please"),
        other => panic!("unexpected result: {other:?}"),
    }
    // Real templates: Mistral rejects a system role.
    let mistral = compile(MISTRAL);
    let error = mistral
        .render(&options(&simple_conversation()))
        .expect_err("must fail");
    assert!(
        error
            .to_string()
            .contains("Conversation roles must alternate"),
        "{error}"
    );
}

#[test]
fn undefined_variables_stay_undefined() {
    let template = ChatTemplate::new(
        "{{ 'defined' if enable_thinking is defined else 'undefined' }}/{{ 'defined' if tools is defined else 'undefined' }}",
        "",
        "",
    )
    .expect("compile");
    let messages = [];
    assert_eq!(
        template.render(&options(&messages)).unwrap(),
        "undefined/undefined"
    );
    let mut opts = options(&messages);
    opts.enable_thinking = Some(false);
    assert_eq!(template.render(&opts).unwrap(), "defined/undefined");
}

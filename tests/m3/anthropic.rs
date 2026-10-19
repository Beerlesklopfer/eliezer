use eliezer_api::anthropic::*;
use serde_json::json;
#[test]
fn all_request_mappings() {
    let input = json!({"model":"a","max_tokens":42,"system":[{"type":"text","text":"System"}],"messages":[
        {"role":"user","content":[{"type":"text","text":"Hallo"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]},
        {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"read","input":{"file":"a"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"Ergebnis"}]}],
        "tools":[{"name":"read","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"},"thinking":{"type":"enabled","budget_tokens":4096},"stop_sequences":["STOP"],"unknown":true});
    let output = request(&input, &["none".into(), "medium".into()]).unwrap();
    assert_eq!(
        output["messages"][0],
        json!({"role":"system","content":"System"})
    );
    assert_eq!(
        output["messages"][1]["content"][1]["image_url"]["url"],
        "data:image/png;base64,AAAA"
    );
    assert_eq!(
        output["messages"][2]["tool_calls"][0]["function"]["arguments"],
        "{\"file\":\"a\"}"
    );
    assert_eq!(output["messages"][3]["role"], "tool");
    assert_eq!(output["tool_choice"], "required");
    assert_eq!(output["reasoning_effort"], "medium");
    assert_eq!(
        output["tools"][0]["function"]["parameters"],
        json!({"type":"object"})
    );
    assert_eq!(output["stop"], json!(["STOP"]));
    assert!(output.get("unknown").is_none());
}
#[test]
fn response_mapping_and_rejection() {
    let output=response(&json!({"id":"msg_1","model":"a","choices":[{"message":{"content":"Text","reasoning_content":"Denken","tool_calls":[{"id":"t1","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":4}})).unwrap();
    assert_eq!(output["content"][0]["type"], "thinking");
    assert_eq!(output["content"][2]["type"], "tool_use");
    assert_eq!(output["stop_reason"], "tool_use");
    assert_eq!(output["usage"]["input_tokens"], 3);
    for reason in [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
    ] {
        assert_eq!(stop_reason(reason.0), reason.1);
    }
    let bad = json!({"model":"a","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document"}]}]});
    assert!(request(&bad, &[]).is_err());
}

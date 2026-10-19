//! Reine Übersetzung der Messages-Schnittstelle.
use serde_json::{Value, json};

pub fn request(input: &Value, options: &[String]) -> Result<Value, &'static str> {
    if !input["max_tokens"].as_u64().is_some_and(|n| n > 0) {
        return Err("max_tokens muss positiv sein");
    }
    let mut messages = Vec::new();
    if let Some(system) = input.get("system") {
        let text = if let Some(text) = system.as_str() {
            text.to_owned()
        } else {
            system
                .as_array()
                .ok_or("Ungültiger Systemtext")?
                .iter()
                .map(|b| b["text"].as_str().ok_or("Ungültiger Systemblock"))
                .collect::<Result<Vec<_>, _>>()?
                .join("\n")
        };
        messages.push(json!({"role":"system","content":text}));
    }
    for message in input["messages"].as_array().ok_or("Nachrichten fehlen")? {
        let role = message["role"]
            .as_str()
            .filter(|v| ["user", "assistant"].contains(v))
            .ok_or("Ungültige Nachrichtenrolle")?;
        if let Some(text) = message["content"].as_str() {
            messages.push(json!({"role":role,"content":text}));
            continue;
        }
        let mut content = Vec::new();
        let mut calls = Vec::new();
        let mut results = Vec::new();
        for block in message["content"]
            .as_array()
            .ok_or("Ungültiger Nachrichteninhalt")?
        {
            match block["type"].as_str() {
                Some("text") => content.push(json!({"type":"text","text":block["text"].as_str().ok_or("Text fehlt")?})),
                Some("image") => {
                    let source = &block["source"];
                    if source["type"] != "base64" { return Err("Nur Base64-Bilder werden unterstützt"); }
                    let mime = source["media_type"].as_str().filter(|v| ["image/png","image/jpeg","image/gif","image/webp"].contains(v)).ok_or("Ungültiges Bildformat")?;
                    let data = source["data"].as_str().ok_or("Bilddaten fehlen")?;
                    content.push(json!({"type":"image_url","image_url":{"url":format!("data:{mime};base64,{data}")}}));
                }
                Some("tool_use") => calls.push(json!({"id":block["id"].as_str().ok_or("Werkzeug-ID fehlt")?,"type":"function",
                    "function":{"name":block["name"].as_str().ok_or("Werkzeugname fehlt")?,"arguments":block["input"].to_string()}})),
                Some("tool_result") => results.push(json!({"role":"tool","tool_call_id":block["tool_use_id"].as_str().ok_or("Werkzeug-ID fehlt")?,
                    "content": if let Some(text) = block["content"].as_str() { Value::String(text.into()) } else { block["content"].clone() }})),
                Some("thinking") => {},
                _ => return Err("Dieser Inhaltsblock wird nicht unterstützt"),
            }
        }
        if !content.is_empty() || !calls.is_empty() {
            let mut m = json!({"role":role,"content":content});
            if !calls.is_empty() {
                m["tool_calls"] = json!(calls);
            }
            messages.push(m);
        }
        messages.extend(results);
    }
    let mut output =
        json!({"model":input["model"],"messages":messages,"max_tokens":input["max_tokens"]});
    for key in ["temperature", "top_p", "stream"] {
        if let Some(value) = input.get(key) {
            output[key] = value.clone();
        }
    }
    if let Some(stops) = input.get("stop_sequences") {
        output["stop"] = stops.clone();
    }
    if let Some(tools) = input.get("tools") {
        output["tools"] = Value::Array(tools.as_array().ok_or("Ungültige Werkzeuge")?.iter().map(|t| {
            if t.get("type").is_some() { return Err("Serverseitige Werkzeuge werden nicht unterstützt"); }
            Ok(json!({"type":"function","function":{"name":t["name"].as_str().ok_or("Werkzeugname fehlt")?,"description":t["description"],"parameters":t["input_schema"]}}))
        }).collect::<Result<Vec<_>, _>>()?);
    }
    if let Some(choice) = input.get("tool_choice") {
        output["tool_choice"] = match choice["type"].as_str() {
            Some("auto") => json!("auto"),
            Some("any") => json!("required"),
            Some("none") => json!("none"),
            Some("tool") => {
                json!({"type":"function","function":{"name":choice["name"].as_str().ok_or("Werkzeugname fehlt")?}})
            }
            _ => return Err("Ungültige Werkzeugauswahl"),
        };
    }
    if let Some(thinking) = input.get("thinking") {
        let target = match thinking["type"].as_str() {
            Some("disabled") => "none",
            Some("enabled") => match thinking["budget_tokens"]
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or("Denkbudget fehlt")?
            {
                0..=1024 => "low",
                1025..=8192 => "medium",
                _ => "high",
            },
            _ => return Err("Ungültige Denkoption"),
        };
        let ranks = ["none", "minimal", "low", "medium", "high", "xhigh"];
        let rank = ranks.iter().position(|s| *s == target).unwrap_or(0);
        let selected = options
            .iter()
            .filter_map(|o| {
                ranks
                    .iter()
                    .position(|s| s == o)
                    .map(|r| (r.abs_diff(rank), r, o))
            })
            .min_by_key(|(distance, rank, _)| (*distance, *rank))
            .map(|(_, _, o)| o)
            .ok_or("Modell unterstützt keine Denkoptionen")?;
        output["reasoning_effort"] = json!(selected);
    }
    Ok(output)
}
pub fn stop_reason(reason: &str) -> &'static str {
    match reason {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "stop_sequence" => "stop_sequence",
        _ => "end_turn",
    }
}
pub fn response(openai: &Value) -> Result<Value, &'static str> {
    let choice = &openai["choices"][0];
    let message = &choice["message"];
    let mut content = Vec::new();
    if let Some(text) = message["reasoning_content"].as_str() {
        content.push(json!({"type":"thinking","thinking":text}));
    }
    if let Some(text) = message["content"].as_str()
        && !text.is_empty()
    {
        content.push(json!({"type":"text","text":text}));
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            let arguments: Value = serde_json::from_str(
                call["function"]["arguments"]
                    .as_str()
                    .ok_or("Werkzeugargumente fehlen")?,
            )
            .map_err(|_| "Ungültige Werkzeugargumente")?;
            content.push(json!({"type":"tool_use","id":call["id"],"name":call["function"]["name"],"input":arguments}));
        }
    }
    Ok(
        json!({"id":openai["id"],"type":"message","role":"assistant","model":openai["model"],"content":content,
        "stop_reason":stop_reason(choice["finish_reason"].as_str().unwrap_or("stop")),"stop_sequence":null,
        "usage":{"input_tokens":openai["usage"]["prompt_tokens"],"output_tokens":openai["usage"]["completion_tokens"]}}),
    )
}

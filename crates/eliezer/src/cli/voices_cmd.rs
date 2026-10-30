//! `eliezer voices`, `eliezer voices add NAME FILE` and `eliezer voices rm NAME`: the voice library of speech
//! synthesis (`/v1/audio/voices`).
use super::{
    Client, Failure, Parsed, Res, SHORT, api_error, call, print_json, transport, usage_err,
};
use eliezer_core::config::valid_voice_name;
use serde_json::Value;
use std::{fs, time::Duration};

/// How long to wait for the service to check and store a recording.
const UPLOAD_WAIT: Duration = Duration::from_secs(300);
/// The limit of the service is a setting that only it knows; here a name only has to be safe to put into a path.
const NAME_BOUND: u32 = 255;

pub fn cmd_voices(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let words: Vec<&str> = parsed.positional.iter().map(String::as_str).collect();
    match words.as_slice() {
        [] => list(client, json),
        ["add", name, file] => add(client, json, parsed, name, file),
        ["rm", name] => remove(client, json, name),
        ["add", ..] => {
            usage_err("Usage: eliezer voices add NAME FILE [--replace] [--transcript TEXT]")
        }
        ["rm", ..] => usage_err("Usage: eliezer voices rm NAME"),
        [other, ..] => usage_err(format!(
            "Unknown voices command '{other}'; use `eliezer voices`, `voices add` or `voices rm`"
        )),
    }
}

fn check_name(name: &str) -> Res<()> {
    if valid_voice_name(name, NAME_BOUND) {
        Ok(())
    } else {
        usage_err("A voice name has only letters, digits, - and _")
    }
}

fn list(client: &Client, json: bool) -> Res<()> {
    let answer = call(client, "GET", "/v1/audio/voices", None, SHORT)?;
    if json {
        print_json(&answer);
        return Ok(());
    }
    let names: Vec<&str> = answer["voices"]
        .as_array()
        .map(|l| l.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if names.is_empty() {
        eprintln!("There are no voices; add one with `eliezer voices add NAME FILE`.");
    }
    for name in names {
        println!("{name}");
    }
    Ok(())
}

/// A multipart body of text fields and one file.
fn multipart(fields: &[(&str, &str)], file: &[u8], file_name: &str) -> (String, Vec<u8>) {
    let boundary = format!("eliezer-{:016x}", file.len() as u64 ^ 0x9e37_79b9_7f4a_7c15);
    let mut body = Vec::with_capacity(file.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn add(client: &Client, json: bool, parsed: &Parsed, name: &str, file: &str) -> Res<()> {
    check_name(name)?;
    let data = fs::read(file)
        .map_err(|e| Failure::Usage(format!("Cannot read the recording {file}: {e}")))?;
    let replace = if parsed.flag("--replace") {
        "true"
    } else {
        "false"
    };
    let mut fields = vec![("name", name), ("replace", replace)];
    if let Some(transcript) = parsed.value("--transcript") {
        fields.push(("transcript", transcript));
    }
    // The file name is for the log of the service only; a name that cannot be quoted is left out.
    let (content_type, body) = multipart(&fields, &data, "voice");
    let response = client
        .send_bytes(
            "POST",
            "/v1/audio/voices",
            Some((&content_type, &body)),
            UPLOAD_WAIT,
        )
        .map_err(|e| transport(client, e))?;
    let status = response.status;
    let text = response
        .text()
        .map_err(|e| Failure::Request(format!("Reading the answer failed: {e}")))?;
    if !(200..300).contains(&status) {
        return Err(api_error(client, status, &text));
    }
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if json {
        print_json(&answer);
    } else if answer["replaced"] == true {
        eprintln!("Voice {name} replaced");
    } else {
        eprintln!("Voice {name} added");
    }
    Ok(())
}

fn remove(client: &Client, json: bool, name: &str) -> Res<()> {
    check_name(name)?;
    let answer = call(
        client,
        "DELETE",
        &format!("/v1/audio/voices/{name}"),
        None,
        SHORT,
    )?;
    if json {
        print_json(&answer);
    } else {
        eprintln!("Voice {name} removed");
    }
    Ok(())
}

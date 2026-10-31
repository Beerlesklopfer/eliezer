//! `eliezer voices`, `eliezer voices add NAME FILE` and `eliezer voices rm NAME`: the voice library of speech
//! synthesis (`/v1/audio/voices`).
use super::{
    Client, Failure, Parsed, Res, SHORT, api_error, call, print_json, transport, usage_err,
};
use eliezer_core::config::valid_voice_name;
use serde_json::Value;
use std::{fs, path::Path, time::Duration};

/// How long to wait for the service to check and store a recording.
const UPLOAD_WAIT: Duration = Duration::from_secs(300);
/// The limit of the service is a setting that only it knows; here a name only has to be safe to put into a path.
const ADD_USAGE: &str = "Usage: eliezer voices add NAME FILE [--replace] [--transcript TEXT]\n   or: eliezer voices add --name NAME --file AUDIO [--file TEXTFILE] [--replace] [--transcript TEXT]";
const NAME_BOUND: u32 = 255;

pub fn cmd_voices(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let words: Vec<&str> = parsed.positional.iter().map(String::as_str).collect();
    match words.as_slice() {
        [] => list(client, json),
        ["add", rest @ ..] => add(client, json, parsed, rest),
        ["rm", name] => remove(client, json, name),
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

/// The voice name, the recording and the transcript from the command line. The name is `--name` or the first word
/// after `add`; the files are the other words and every `--file`, each a recording or, with the extension `.txt`,
/// the file with the words spoken in it.
/// The rate of the part that is uploaded; the service converts it to its own sample rate.
const CUT_RATE: u32 = 24_000;
/// The longest part that is cut when only `--start` is given.
const CUT_MAX_SECONDS: u64 = 600;

/// A time as seconds, minutes and seconds, or hours, minutes and seconds: `90`, `1:30`, `0:01:30.5`.
fn parse_time(option: &str, text: &str) -> Res<f64> {
    let bad = || Failure::Usage(format!("{option} is a time such as 90, 1:30 or 0:01:30"));
    let mut seconds = 0.0_f64;
    for part in text.split(':') {
        let value: f64 = part.parse().map_err(|_| bad())?;
        if !value.is_finite() || value < 0.0 {
            return Err(bad());
        }
        seconds = seconds * 60.0 + value;
    }
    Ok(seconds)
}

/// The part of the recording that `--start` and `--length` name, as a WAV file.
fn cut(parsed: &Parsed, data: Vec<u8>) -> Res<Vec<u8>> {
    let start = parsed
        .value("--start")
        .map_or(Ok(0.0), |t| parse_time("--start", t))?;
    let length = parsed
        .value("--length")
        .map(|t| parse_time("--length", t))
        .transpose()?;
    if length == Some(0.0) {
        return usage_err("--length must be more than 0");
    }
    let limit = length.map_or(CUT_MAX_SECONDS, |l| l.ceil() as u64);
    match eliezer_speech::decode_window(data, limit, CUT_RATE, start, length) {
        Ok(samples) => Ok(eliezer_speech::wav_at(&samples, CUT_RATE)),
        Err(eliezer_speech::DecodeError::NotAudio) => usage_err(
            "The recording is not audio this program can read, or --start is behind its end",
        ),
        Err(eliezer_speech::DecodeError::TooLong) => usage_err(format!(
            "The part is longer than {CUT_MAX_SECONDS} s; give --length"
        )),
    }
}

fn add_arguments(parsed: &Parsed, rest: &[&str]) -> Res<(String, String, Option<String>)> {
    let (name, words) = match parsed.value("--name") {
        Some(name) => (name, rest),
        None => match rest.split_first() {
            Some((name, words)) => (*name, words),
            None => return usage_err(ADD_USAGE),
        },
    };
    let (mut audio, mut text) = (Vec::new(), Vec::new());
    for file in words.iter().copied().chain(parsed.all_values("--file")) {
        let is_text = Path::new(file)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("txt"));
        if is_text { &mut text } else { &mut audio }.push(file);
    }
    let ([audio], true) = (audio.as_slice(), text.len() <= 1) else {
        return usage_err(ADD_USAGE);
    };
    let transcript = match (text.first(), parsed.value("--transcript")) {
        (Some(_), Some(_)) => return usage_err("Give --transcript or a .txt file, not both"),
        (Some(file), None) => Some(
            fs::read_to_string(file)
                .map_err(|e| Failure::Usage(format!("Cannot read the text file {file}: {e}")))?,
        ),
        (None, given) => given.map(str::to_owned),
    };
    Ok((name.to_owned(), (*audio).to_owned(), transcript))
}

fn add(client: &Client, json: bool, parsed: &Parsed, rest: &[&str]) -> Res<()> {
    let (name, file, transcript) = add_arguments(parsed, rest)?;
    let (name, file) = (name.as_str(), file.as_str());
    check_name(name)?;
    let mut data = fs::read(file)
        .map_err(|e| Failure::Usage(format!("Cannot read the recording {file}: {e}")))?;
    if parsed.value("--start").is_some() || parsed.value("--length").is_some() {
        data = cut(parsed, data)?;
    }
    let replace = if parsed.flag("--replace") {
        "true"
    } else {
        "false"
    };
    let mut fields = vec![("name", name), ("replace", replace)];
    if let Some(transcript) = transcript.as_deref() {
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

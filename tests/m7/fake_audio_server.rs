//! A stand-in for `audiocpp_server` for the tests: it reads the configuration file it is started with, answers the two
//! endpoints the service uses and records what it was started with. Its behavior follows the first line of
//! `config.json` in the package directory of the model:
//!
//!   `ok`                 loaded at once, composes a tiny song
//!   `crash`              ends with status 3 before it listens
//!   `slow-load MS`       listens at once, lists the model as loaded after MS milliseconds
//!   `never-loaded`       listens but never lists the model as loaded
//!   `die-after-first`    answers one request and exits
//!   `error-500`          answers every request with 500
//!   `bad-request`        answers every request with 400
//!   `not-wav`            answers 200 with something that is not a WAV file
//!
//! In the package directory it writes `fake.args` (one argument per line), `fake.config` (the configuration file),
//! `fake.env` (the names of the environment variables it got), `fake.ldpath` and `fake.request` (the last request body).

use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    time::{Duration, Instant},
};

fn wav() -> Vec<u8> {
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&36_u32.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&[1, 0, 1, 0]);
    wav.extend_from_slice(&8000_u32.to_le_bytes());
    wav.extend_from_slice(&16000_u32.to_le_bytes());
    wav.extend_from_slice(&[2, 0, 16, 0]);
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&0_u32.to_le_bytes());
    wav
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

fn handle(
    mut stream: TcpStream,
    package: &std::path::Path,
    mode: &str,
    id: &str,
    loaded_at: Instant,
) -> bool {
    let mut data = Vec::new();
    let mut buffer = [0_u8; 4096];
    let (head_end, length) = loop {
        let n = stream.read(&mut buffer).unwrap_or(0);
        if n == 0 {
            return false;
        }
        data.extend_from_slice(&buffer[..n]);
        if let Some(at) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&data[..at]).to_ascii_lowercase();
            let length = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            break (at + 4, length);
        }
    };
    while data.len() < head_end + length {
        let n = stream.read(&mut buffer).unwrap_or(0);
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..n]);
    }
    let head = String::from_utf8_lossy(&data[..head_end]).into_owned();
    let body = data[head_end..].to_vec();
    let first = head.lines().next().unwrap_or_default().to_owned();
    if first.starts_with("GET /v1/models") {
        let loaded = match mode {
            "never-loaded" => false,
            _ => Instant::now() >= loaded_at,
        };
        let text = format!(r#"{{"object":"list","data":[{{"id":"{id}","loaded":{loaded}}}]}}"#);
        respond(&mut stream, "200 OK", "application/json", text.as_bytes());
        return false;
    }
    if first.starts_with("POST /v1/audio/speech") {
        let _ = fs::write(package.join("fake.request"), &body);
        match mode {
            "error-500" => respond(
                &mut stream,
                "500 Internal Server Error",
                "application/json",
                b"{}",
            ),
            "bad-request" => respond(&mut stream, "400 Bad Request", "application/json", b"{}"),
            "not-wav" => respond(
                &mut stream,
                "200 OK",
                "application/json",
                br#"{"error":"x"}"#,
            ),
            _ => respond(&mut stream, "200 OK", "audio/wav", &wav()),
        }
        return mode == "die-after-first";
    }
    respond(&mut stream, "404 Not Found", "application/json", b"{}");
    false
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(config_path) = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
    else {
        std::process::exit(2);
    };
    let Ok(text) = fs::read_to_string(&config_path) else {
        std::process::exit(2);
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&text) else {
        std::process::exit(2);
    };
    let package = PathBuf::from(config["models"][0]["path"].as_str().unwrap_or_default());
    let id = config["models"][0]["id"]
        .as_str()
        .unwrap_or("music")
        .to_owned();
    let port = config["port"].as_u64().unwrap_or(0) as u16;
    let _ = fs::write(package.join("fake.args"), args.join("\n"));
    let _ = fs::write(package.join("fake.config"), &text);
    let mut names: Vec<String> = std::env::vars().map(|(k, _)| k).collect();
    names.sort();
    let _ = fs::write(package.join("fake.env"), names.join("\n"));
    let _ = fs::write(
        package.join("fake.ldpath"),
        std::env::var("LD_LIBRARY_PATH").unwrap_or_default(),
    );
    let mode_line = fs::read_to_string(package.join("config.json")).unwrap_or_default();
    let mode_line = mode_line.lines().next().unwrap_or("ok").trim().to_owned();
    let mut words = mode_line.split_whitespace();
    let mode = words.next().unwrap_or("ok").to_owned();
    let delay: u64 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
    if mode == "crash" {
        std::process::exit(3);
    }
    let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) else {
        std::process::exit(4);
    };
    let loaded_at = Instant::now() + Duration::from_millis(delay);
    for stream in listener.incoming().flatten() {
        if handle(stream, &package, &mode, &id, loaded_at) {
            std::process::exit(0);
        }
    }
}

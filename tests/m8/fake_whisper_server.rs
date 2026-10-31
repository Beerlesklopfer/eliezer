//! A stand-in for `whisper-server` for the tests: it reads the arguments it is started with (`--model`, `--port`),
//! answers `GET /health` and `POST /inference` and records what it was started with. Its behavior follows the first line
//! of the model file (the "ggml file"):
//!
//!   `ok`                 listens at once
//!   `crash`              ends with status 3 before it listens
//!   `slow-load MS`       starts to listen after MS milliseconds (the real program listens only after loading)
//!   `never-ready`        listens, but `/health` always answers 503
//!   `die-after-first`    answers one request and exits
//!   `error-500`          answers every inference with 500
//!   `bad-request`        answers every inference with 400
//!   `no-segments`        answers 200 with a body that has no segments
//!
//! Next to the model file it writes `fake.args` (one argument per line), `fake.env` (the names of the environment
//! variables it got), `fake.cwd`, `fake.ldpath` and `fake.request` (the last request body, raw).
//! The answer of `/inference` is a `verbose_json` made of what the request held: one segment whose text names the
//! language, the prompt and the temperature fields, so a test sees what reached the program.

use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    time::Duration,
};

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

/// The text of a form field of a multipart body.
fn field(body: &[u8], name: &str) -> Option<String> {
    let text = String::from_utf8_lossy(body);
    let marker = format!("name=\"{name}\"\r\n\r\n");
    let start = text.find(&marker)? + marker.len();
    let end = text[start..].find("\r\n")? + start;
    Some(text[start..end].to_owned())
}

fn handle(mut stream: TcpStream, dir: &Path, mode: &str, started: bool) -> bool {
    let mut data = Vec::new();
    let mut buffer = [0_u8; 8192];
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
    if first.starts_with("GET /health") {
        if mode == "never-ready" || !started {
            respond(
                &mut stream,
                "503 Service Unavailable",
                "application/json",
                br#"{"status":"loading model"}"#,
            );
        } else {
            respond(
                &mut stream,
                "200 OK",
                "application/json",
                br#"{"status":"ok"}"#,
            );
        }
        return false;
    }
    if first.starts_with("POST /inference") {
        let _ = fs::write(dir.join("fake.request"), &body);
        match mode {
            "error-500" => respond(
                &mut stream,
                "500 Internal Server Error",
                "application/json",
                br#"{"error":"failed to process audio"}"#,
            ),
            "bad-request" => respond(
                &mut stream,
                "400 Bad Request",
                "application/json",
                br#"{"error":"failed to read audio data"}"#,
            ),
            "no-segments" => respond(
                &mut stream,
                "200 OK",
                "application/json",
                br#"{"error":"x"}"#,
            ),
            _ => {
                let text = format!(
                    " lang={} prompt={} temperature={} format={}",
                    field(&body, "language").unwrap_or_default(),
                    field(&body, "prompt").unwrap_or_default(),
                    field(&body, "temperature").unwrap_or_default(),
                    field(&body, "response_format").unwrap_or_default(),
                );
                let answer = serde_json::json!({
                    "task": "transcribe",
                    "language": "german",
                    "duration": 2.0,
                    "text": text,
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.5, "text": text},
                        {"id": 1, "start": 1.5, "end": 2.0, "text": " second"},
                    ],
                });
                respond(
                    &mut stream,
                    "200 OK",
                    "application/json",
                    answer.to_string().as_bytes(),
                );
            }
        }
        return mode == "die-after-first";
    }
    respond(&mut stream, "404 Not Found", "application/json", b"{}");
    false
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let (Some(model), Some(port)) = (value("--model"), value("--port")) else {
        std::process::exit(2);
    };
    let model = PathBuf::from(model);
    let port: u16 = port.parse().unwrap_or(0);
    let dir = model.parent().map(Path::to_path_buf).unwrap_or_default();
    let _ = fs::write(dir.join("fake.args"), args.join("\n"));
    let mut names: Vec<String> = std::env::vars().map(|(k, _)| k).collect();
    names.sort();
    let _ = fs::write(dir.join("fake.env"), names.join("\n"));
    let _ = fs::write(
        dir.join("fake.cwd"),
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
    );
    let _ = fs::write(
        dir.join("fake.ldpath"),
        std::env::var("LD_LIBRARY_PATH").unwrap_or_default(),
    );
    let mode_line = fs::read_to_string(&model).unwrap_or_default();
    let mode_line = mode_line.lines().next().unwrap_or("ok").trim().to_owned();
    let mut words = mode_line.split_whitespace();
    let mode = words.next().unwrap_or("ok").to_owned();
    let delay: u64 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
    if mode == "crash" {
        std::process::exit(3);
    }
    // The real program binds its port only after the model is loaded.
    std::thread::sleep(Duration::from_millis(delay));
    let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) else {
        std::process::exit(4);
    };
    for stream in listener.incoming().flatten() {
        if handle(stream, &dir, &mode, true) {
            std::process::exit(0);
        }
    }
}

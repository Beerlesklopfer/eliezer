//! A stand-in for `sd-server` for the tests: it answers the two endpoints the service uses and records what it
//! was started with. Its behavior follows the first line of the diffusion model file:
//!
//!   `ok`                 ready at once, draws tiny images
//!   `crash`              ends with status 3 before it listens
//!   `slow-ready MS`      listens after MS milliseconds
//!   `never-ready`        runs but never listens
//!   `die-after-first`    answers one generation and exits
//!   `error-500`          answers every generation with 500
//!   `bad-request`        answers every generation with 400
//!
//! Beside the model file it writes `<model>.args` (one argument per line), `<model>.ldpath` (its library path),
//! `<model>.env` (the names of the
//! environment variables it got) and `<model>.request` (the last generation request body).

use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    time::Duration,
};

/// A 1x1 transparent PNG.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn argument(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) {
    let reply = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(reply.as_bytes());
}

fn handle(mut stream: TcpStream, model: &std::path::Path, mode: &str) -> bool {
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
    let body = String::from_utf8_lossy(&data[head_end..]).into_owned();
    let first = head.lines().next().unwrap_or_default().to_owned();
    if first.starts_with("GET /v1/models") {
        respond(&mut stream, "200 OK", r#"{"data":[{"id":"fake"}]}"#);
        return false;
    }
    if first.starts_with("POST /v1/images/generations") {
        let _ = fs::write(format!("{}.request", model.display()), &body);
        match mode {
            "error-500" => respond(&mut stream, "500 Internal Server Error", "{}"),
            "bad-request" => respond(&mut stream, "400 Bad Request", "{}"),
            _ => {
                let n = body
                    .split("\"n\":")
                    .nth(1)
                    .and_then(|r| r.split([',', '}']).next())
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(1);
                let items: Vec<String> = (0..n)
                    .map(|_| format!(r#"{{"b64_json":"{PNG}"}}"#))
                    .collect();
                respond(
                    &mut stream,
                    "200 OK",
                    &format!(
                        r#"{{"created":1,"output_format":"png","data":[{}]}}"#,
                        items.join(",")
                    ),
                );
            }
        }
        return mode == "die-after-first";
    }
    respond(&mut stream, "404 Not Found", "{}");
    false
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(model) = argument(&args, "--diffusion-model").map(PathBuf::from) else {
        std::process::exit(2);
    };
    let port: u16 = argument(&args, "--listen-port")
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);
    let _ = fs::write(format!("{}.args", model.display()), args.join("\n"));
    let mut names: Vec<String> = std::env::vars().map(|(k, _)| k).collect();
    names.sort();
    let _ = fs::write(format!("{}.env", model.display()), names.join("\n"));
    let _ = fs::write(
        format!("{}.ldpath", model.display()),
        std::env::var("LD_LIBRARY_PATH").unwrap_or_default(),
    );
    let content = fs::read_to_string(&model).unwrap_or_default();
    let mode_line = content.lines().next().unwrap_or("ok").to_owned();
    let mode = mode_line
        .split_whitespace()
        .next()
        .unwrap_or("ok")
        .to_owned();
    match mode.as_str() {
        "crash" => std::process::exit(3),
        "never-ready" => loop {
            std::thread::sleep(Duration::from_secs(3600));
        },
        "slow-ready" => {
            let ms: u64 = mode_line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(500);
            std::thread::sleep(Duration::from_millis(ms));
        }
        _ => {}
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");
    for stream in listener.incoming().flatten() {
        if handle(stream, &model, &mode) {
            std::process::exit(0);
        }
    }
}

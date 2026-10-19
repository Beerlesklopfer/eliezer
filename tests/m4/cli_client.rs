//! Command line client (`eliezer ls|ps|load|unload|status|run|show`): pure formatting, the
//! commands against a canned in-test server on loopback, and end to end against the real service.
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[allow(dead_code)]
#[path = "../../crates/eliezer/src/cli/format.rs"]
mod format;

// ---- (a) pure functions --------------------------------------------------------------------

#[test]
fn sizes_and_durations_are_readable() {
    assert_eq!(format::bytes(17_300_000_000), "17.3 GB");
    assert_eq!(format::bytes(0), "0.0 MB");
    assert_eq!(format::bytes(84_110_000), "84.1 MB");
    assert_eq!(format::bytes(100_000_000), "0.1 GB");
    assert_eq!(format::duration_ms(850), "850 ms");
    assert_eq!(format::duration_ms(12_300), "12.3 s");
    assert_eq!(format::duration_ms(245_000), "4m 05s");
    assert_eq!(format::duration_ms(7_380_000), "2h 03m");
    assert_eq!(format::duration_ms(93_600_000), "1d 02h");
    assert_eq!(format::duration_secs(12.34), "12.3 s");
    assert_eq!(format::duration_secs(f64::NAN), "-");
}

#[test]
fn tables_align_columns_and_leave_no_trailing_blanks() {
    let rows = vec![
        vec!["a".to_owned(), "1".to_owned(), "x".to_owned()],
        vec!["long-name".to_owned(), "100".to_owned(), "yy".to_owned()],
    ];
    let text = format::table(&["MODEL", "N", "STATE"], &rows, &[1]);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "MODEL          N    STATE");
    assert_eq!(lines[1], "a              1    x");
    assert_eq!(lines[2], "long-name    100    yy");
    assert!(lines.iter().all(|l| l == &l.trim_end()));
}

fn sample_models() -> Value {
    json!({"models": [
        {"type": "llm", "key": "qwen", "size_bytes": 17_300_000_000_u64, "max_context_length": 131072,
         "capabilities": {"vision": false, "reasoning": {"allowed_options": ["none", "low"], "default": "low"}},
         "loaded_instances": [{"id": "qwen", "config": {"context_length": 32768, "parallel": 2}}]},
        {"type": "embedding", "key": "nomic", "size_bytes": 84_000_000_u64, "max_context_length": 8192,
         "capabilities": {"vision": false}, "loaded_instances": []}
    ]})
}
fn sample_status() -> Value {
    json!({"gpu": {"probe": "nvml", "total_bytes": 24_000_000_000_u64, "used_bytes": 12_000_000_000_u64,
                   "free_bytes": 12_000_000_000_u64, "budget_bytes": 22_000_000_000_u64},
           "models": [
        {"key": "qwen", "state": "loaded", "vram_bytes": 11_000_000_000_u64, "last_used_ms_ago": 65_000, "in_flight": 1, "queue_len": 0},
        {"key": "nomic", "state": "unloaded", "vram_bytes": 0, "last_used_ms_ago": null, "in_flight": 0, "queue_len": 0}],
           "queue": [{"model": "nomic", "waiting_secs": 3.2}]})
}

#[test]
fn ls_lists_the_capabilities_of_each_model() {
    let models = json!({"models": [
        {"type": "llm", "key": "ocr", "size_bytes": 4_700_000_000_u64, "max_context_length": 128000,
         "capabilities": {"vision": true}, "loaded_instances": []},
        {"type": "llm", "key": "thinker", "size_bytes": 16_800_000_000_u64, "max_context_length": 262144,
         "capabilities": {"vision": true, "reasoning": {"allowed_options": ["none", "medium"], "default": "none"}},
         "loaded_instances": []},
        {"type": "llm", "key": "chat", "size_bytes": 4_400_000_000_u64, "max_context_length": 32768,
         "capabilities": {"vision": false, "reasoning": {"allowed_options": [], "default": "none"}},
         "loaded_instances": []},
        {"type": "embedding", "key": "embed", "size_bytes": 270_000_000_u64, "max_context_length": 2048,
         "loaded_instances": []}]});
    let ls = format::ls_text(&models, None);
    assert!(ls.contains("CAPABILITIES"), "{ls}");
    let row = |key: &str| ls.lines().find(|l| l.starts_with(key)).unwrap().to_owned();
    assert!(
        row("ocr").contains("vision") && !row("ocr").contains("reasoning"),
        "{ls}"
    );
    assert!(row("thinker").contains("vision, reasoning"), "{ls}");
    for key in ["chat", "embed"] {
        let row = row(key);
        assert!(
            !row.contains("vision") && !row.contains("reasoning"),
            "{row}"
        );
    }
}

#[test]
fn model_status_and_stats_texts() {
    let ls = format::ls_text(&sample_models(), Some(&sample_status()));
    assert!(
        ls.contains("You have 2 configured models, 17.4 GB in total."),
        "{ls}"
    );
    assert!(ls.contains("MODEL") && ls.contains("131072") && ls.contains("84.0 MB"));
    assert!(
        ls.lines()
            .any(|l| l.starts_with("qwen") && l.ends_with("loaded"))
    );
    assert!(
        ls.lines()
            .any(|l| l.starts_with("nomic") && l.ends_with("unloaded"))
    );
    assert_eq!(
        format::ls_text(&json!({"models": []}), None),
        "No models are configured.\n"
    );

    let ps = format::ps_text(&sample_models(), &sample_status());
    assert!(
        ps.contains("11.0 GB") && ps.contains("32768") && ps.contains("1m 05s"),
        "{ps}"
    );
    assert!(
        !ps.contains("nomic"),
        "unloaded models are not listed:\n{ps}"
    );
    let none = format::ps_text(&sample_models(), &json!({"models": []}));
    assert!(none.starts_with("No models are currently loaded."));

    let status = format::status_text("http://127.0.0.1:1234", "ok", &sample_status());
    for expected in [
        "is running (health: ok)",
        "nvml",
        "24.0 GB",
        "12.0 GB (50%)",
        "22.0 GB",
        "1 loaded of 2 configured",
        "1 waiting: nomic (3.2 s)",
    ] {
        assert!(
            status.contains(expected),
            "{expected} missing in:\n{status}"
        );
    }

    let models = sample_models();
    let show = format::show_text(&models["models"][0], None, "http://h:1");
    assert!(
        show.contains("none, low (default: low)") && show.contains("context 32768, parallel 2")
    );
    assert!(show.contains("POST /v1/chat/completions") && show.contains("/api/v1/models/unload"));

    let stats = format::stats_text(
        Some(&json!({"prompt_tokens": 10, "completion_tokens": 21, "total_tokens": 31})),
        Some(&json!({"ttft_ms": 500, "gen_ms": 2500, "tokens": 21})),
    );
    assert!(stats.contains("completion tokens:    21"), "{stats}");
    assert!(
        stats.contains("time to first token:  500 ms") && stats.contains("10.0 tokens/s"),
        "{stats}"
    );
}

// ---- (b) canned server ---------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Seen {
    method: String,
    path: String,
    auth: Option<String>,
    body: String,
}
type Handler = Box<dyn Fn(&Seen) -> Vec<u8> + Send + Sync>;
struct Canned {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}
impl Canned {
    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
    fn requests(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}
fn reply(status: &str, content_type: &str, body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 {status}\r\nServer: eliezer/test\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}
fn json_reply(status: &str, body: &Value) -> Vec<u8> {
    reply(status, "application/json", &body.to_string())
}
/// SSE answer in chunked transfer encoding; the frames are cut in the middle on purpose.
fn sse_reply(frames: &[String]) -> Vec<u8> {
    let all: String = frames
        .iter()
        .map(|f| format!("data: {f}\n\n"))
        .collect::<String>()
        + "data: [DONE]\n\n";
    let mut out = b"HTTP/1.1 200 OK\r\nServer: eliezer/test\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
    let bytes = all.as_bytes();
    for piece in bytes.chunks(37) {
        out.extend(format!("{:x}\r\n", piece.len()).as_bytes());
        out.extend(piece);
        out.extend(b"\r\n");
    }
    out.extend(b"0\r\n\r\n");
    out
}
fn serve(handler: Handler) -> Canned {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (method, path) = (
                parts.next().unwrap_or("").to_owned(),
                parts.next().unwrap_or("").to_owned(),
            );
            let (mut length, mut auth) = (0_usize, None);
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap_or(0);
                }
                if lower.starts_with("authorization:") {
                    auth = Some(line.split_once(':').unwrap().1.trim().to_owned());
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let request = Seen {
                method,
                path,
                auth,
                body: String::from_utf8_lossy(&body).into_owned(),
            };
            log.lock().unwrap().push(request.clone());
            let _ = stream.write_all(&handler(&request));
        }
    });
    Canned { port, seen }
}

fn eliezer(args: &[&str], envs: &[(&str, &str)], stdin: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eliezer"));
    command
        .env_remove("ELIEZER_HOST")
        .env_remove("ELIEZER_API_KEY")
        .env_remove("ELIEZER_API_KEY_FILE")
        // The user's own key file must not leak into the tests.
        .env("XDG_CONFIG_HOME", "/nonexistent/config")
        .env("ELIEZER_CONFIG", "/nonexistent/eliezer.toml")
        .env("NO_COLOR", "1")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in envs {
        command.env(k, v);
    }
    let mut child = command.spawn().unwrap();
    let mut pipe = child.stdin.take().unwrap();
    if let Some(text) = stdin {
        pipe.write_all(text.as_bytes()).unwrap();
    }
    drop(pipe);
    child.wait_with_output().unwrap()
}
fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}
fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}
fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("eliezer-cli-client-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn key_file(dir: &Path, key: &str) -> PathBuf {
    let path = dir.join("keys");
    fs::write(&path, format!("# keys\n{key}\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    path
}

fn management_server() -> Canned {
    serve(Box::new(|r| match (r.method.as_str(), r.path.as_str()) {
        ("GET", "/healthz") => reply("200 OK", "text/plain", "ok"),
        ("GET", "/api/v1/models") => json_reply("200 OK", &sample_models()),
        ("GET", "/api/v1/status") => json_reply("200 OK", &sample_status()),
        ("POST", "/api/v1/models/load") => json_reply(
            "200 OK",
            &json!({"type": "llm", "instance_id": "qwen", "load_time_seconds": 12.3, "status": "loaded",
                    "load_config": {"context_length": 4096, "parallel": 2}}),
        ),
        ("POST", "/api/v1/models/unload") => {
            let v: Value = serde_json::from_str(&r.body).unwrap();
            json_reply("200 OK", &json!({"instance_id": v["instance_id"]}))
        }
        _ => json_reply(
            "404 Not Found",
            &json!({"error": {"type": "not_found", "message": "no such path"}}),
        ),
    }))
}

#[test]
fn management_commands_print_tables_and_send_the_right_requests() {
    let server = management_server();
    let url = server.url();
    let host = ["--host", url.as_str()];
    let run = |args: &[&str]| {
        let mut all = host.to_vec();
        all.extend_from_slice(args);
        eliezer(&all, &[], None)
    };

    let ls = run(&["ls"]);
    assert_eq!(code(&ls), 0, "{}", err(&ls));
    assert!(out(&ls).contains("qwen") && out(&ls).contains("17.3 GB"));

    let ps = run(&["ps"]);
    assert_eq!(code(&ps), 0);
    assert!(out(&ps).contains("11.0 GB") && out(&ps).contains("IN-FLIGHT"));

    let status = run(&["status"]);
    assert_eq!(code(&status), 0);
    assert!(
        out(&status).contains("is running") && out(&status).contains("GPU budget"),
        "{}",
        out(&status)
    );

    let show = run(&["show", "qwen"]);
    assert_eq!(code(&show), 0);
    assert!(out(&show).contains("Max context") && out(&show).contains("131072"));
    let missing = run(&["show", "nope"]);
    assert_eq!(code(&missing), 1);
    assert!(err(&missing).contains("'nope' is not configured"));

    let json = run(&["ls", "--json"]);
    let parsed: Value = serde_json::from_str(&out(&json)).unwrap();
    assert_eq!(parsed["models"][0]["key"], "qwen");

    let load = run(&["load", "qwen", "--context", "4096", "--parallel=2"]);
    assert_eq!(code(&load), 0, "{}", err(&load));
    assert!(
        out(&load).contains("loaded successfully in 12.3 s")
            && out(&load).contains("context 4096, parallel 2")
    );
    let sent = server
        .requests()
        .into_iter()
        .find(|r| r.path == "/api/v1/models/load")
        .unwrap();
    let body: Value = serde_json::from_str(&sent.body).unwrap();
    assert_eq!(
        (
            body["model"].as_str(),
            body["context_length"].as_u64(),
            body["parallel"].as_u64()
        ),
        (Some("qwen"), Some(4096), Some(2))
    );

    let unload = run(&["unload", "qwen"]);
    assert_eq!(code(&unload), 0);
    assert!(out(&unload).contains("Model unloaded: qwen"));
    // exactly one model is loaded: `unload` without a name takes it
    let alone = run(&["unload"]);
    assert_eq!(code(&alone), 0, "{}", err(&alone));
    let all = run(&["unload", "--all", "--json"]);
    assert_eq!(
        serde_json::from_str::<Value>(&out(&all)).unwrap()["unloaded"],
        json!(["qwen"])
    );
}

#[test]
fn the_key_comes_from_file_or_environment_and_never_from_arguments() {
    let server = management_server();
    let dir = scratch("key");
    let file = key_file(&dir, "sk-secret-123");
    let url = server.url();
    let by_flag = eliezer(
        &["ls", "--host", &url, "--key-file", file.to_str().unwrap()],
        &[],
        None,
    );
    let by_env_file = eliezer(
        &["ps", "--host", &url],
        &[("ELIEZER_API_KEY_FILE", file.to_str().unwrap())],
        None,
    );
    let by_env = eliezer(
        &["ps", "--host", &url],
        &[("ELIEZER_API_KEY", "sk-env-456")],
        None,
    );
    for o in [&by_flag, &by_env_file, &by_env] {
        assert_eq!(code(o), 0, "{}", err(o));
        let all = out(o) + &err(o);
        assert!(!all.contains("sk-secret-123") && !all.contains("sk-env-456"));
    }
    let auth: Vec<_> = server
        .requests()
        .into_iter()
        .filter_map(|r| r.auth)
        .collect();
    assert!(auth.contains(&"Bearer sk-secret-123".to_owned()));
    assert!(auth.contains(&"Bearer sk-env-456".to_owned()));

    let refused = eliezer(&["ls", "--host", &url, "--key", "sk-abc"], &[], None);
    assert_eq!(code(&refused), 2);
    assert!(!err(&refused).contains("sk-abc"));
    let unreadable = eliezer(
        &["ls", "--host", &url, "--key-file", "/nonexistent/keys"],
        &[],
        None,
    );
    assert_eq!(code(&unreadable), 2);
}

#[test]
fn chat_streams_tokens_and_separates_reasoning_and_statistics() {
    let frame = |delta: Value| json!({"choices": [{"index": 0, "delta": delta}]}).to_string();
    let frames = vec![
        frame(json!({"reasoning_content": "Let me think. "})),
        frame(json!({"reasoning_content": "Two plus two."})),
        frame(json!({"content": "The answer "})),
        frame(json!({"content": "is 4."})),
        json!({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
               "eliezer": {"ttft_ms": 100, "gen_ms": 900, "tokens": 9}})
        .to_string(),
    ];
    let server = serve(Box::new(move |_| sse_reply(&frames)));
    let url = server.url();
    let run = eliezer(
        &[
            "run",
            "qwen",
            "What is 2+2?",
            "--host",
            &url,
            "--stats",
            "--system",
            "Be brief",
            "--temperature",
            "0.2",
            "--max-tokens",
            "50",
            "--reasoning",
            "low",
        ],
        &[],
        None,
    );
    assert_eq!(code(&run), 0, "{}", err(&run));
    assert_eq!(
        out(&run),
        "The answer is 4.\n",
        "stdout carries only the answer"
    );
    let e = err(&run);
    assert!(
        e.contains("Let me think. Two plus two."),
        "reasoning is shown on stderr when piped: {e}"
    );
    assert!(
        e.contains("completion tokens:    9") && e.contains("tokens/s"),
        "{e}"
    );
    let body: Value = serde_json::from_str(&server.requests()[0].body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(body["max_tokens"], 50);
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": "Be brief"})
    );
    assert_eq!(
        body["messages"][1],
        json!({"role": "user", "content": "What is 2+2?"})
    );

    // the prompt can come from stdin
    let piped = eliezer(&["run", "qwen", "--host", &url], &[], Some("from stdin\n"));
    assert_eq!(code(&piped), 0, "{}", err(&piped));
    let body: Value = serde_json::from_str(&server.requests()[1].body).unwrap();
    assert_eq!(body["messages"][0]["content"], "from stdin");
    // no prompt at all is a usage error
    let empty = eliezer(&["run", "qwen", "--host", &url], &[], Some(""));
    assert_eq!(code(&empty), 2);
}

#[test]
fn errors_map_to_exit_codes_and_messages() {
    let server = serve(Box::new(|r| match r.path.as_str() {
        "/api/v1/status" => json_reply(
            "401 Unauthorized",
            &json!({"error": {"type": "authentication_error", "message": "invalid API key"}}),
        ),
        "/v1/chat/completions" => json_reply(
            "404 Not Found",
            &json!({"error": {"type": "not_found", "message": "model 'x' is not configured"}}),
        ),
        _ => json_reply(
            "409 Conflict",
            &json!({"error": {"type": "insufficient_memory", "message": "not enough memory: needs 20.0 GB, free 8.0 GB"}}),
        ),
    }));
    let url = server.url();
    let unauthorized = eliezer(&["ps", "--host", &url], &[], None);
    assert_eq!(code(&unauthorized), 1);
    assert!(
        err(&unauthorized).contains("invalid API key")
            && err(&unauthorized).contains("requires an API key"),
        "{}",
        err(&unauthorized)
    );
    let wrong = eliezer(
        &["ps", "--host", &url],
        &[("ELIEZER_API_KEY", "sk-wrong-1")],
        None,
    );
    assert_eq!(code(&wrong), 1);
    assert!(
        err(&wrong).contains("did not accept the API key") && !err(&wrong).contains("sk-wrong-1")
    );
    let memory = eliezer(&["load", "big", "--host", &url], &[], None);
    assert_eq!(code(&memory), 1);
    assert!(
        err(&memory).contains("not enough memory: needs 20.0 GB, free 8.0 GB")
            && err(&memory).contains("insufficient_memory")
    );
    let chat = eliezer(&["run", "x", "hi", "--host", &url], &[], None);
    assert_eq!(code(&chat), 1);
    assert!(err(&chat).contains("model 'x' is not configured"));
}

#[test]
fn an_unreachable_service_gives_exit_code_3_and_says_how_to_start_it() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let url = format!("http://127.0.0.1:{port}");
    for args in [
        vec!["ls"],
        vec!["status"],
        vec!["run", "m", "hi"],
        vec!["unload", "--all"],
    ] {
        let mut all = args.clone();
        all.extend(["--host", &url]);
        let o = eliezer(&all, &[], None);
        assert_eq!(code(&o), 3, "{args:?}");
        assert!(
            err(&o).contains("systemctl start eliezer") && err(&o).contains("eliezer --config"),
            "{}",
            err(&o)
        );
    }
    // ELIEZER_HOST works like --host
    let o = eliezer(&["ps"], &[("ELIEZER_HOST", &url)], None);
    assert_eq!(code(&o), 3);
}

#[test]
fn usage_errors_exit_with_2_and_help_lists_the_commands() {
    for args in [
        vec!["frobnicate"],
        vec!["load"],
        vec!["show"],
        vec!["ls", "extra"],
        vec!["ls", "--bogus"],
        vec!["run", "m", "p", "--reasoning", "extreme"],
        vec!["run", "m", "p", "--max-tokens", "zero"],
        vec!["load", "m", "--context", "0"],
        vec!["unload", "m", "--all"],
        vec!["ls", "--host", "https://x:1"],
    ] {
        let o = eliezer(&args, &[], None);
        assert_eq!(code(&o), 2, "{args:?}: {}", err(&o));
        assert!(err(&o).contains("eliezer help"), "{args:?}");
    }
    for args in [vec!["help"], vec!["--help"], vec!["ls", "--help"]] {
        let o = eliezer(&args, &[], None);
        assert_eq!(code(&o), 0);
        for word in [
            "ls",
            "ps",
            "load MODEL",
            "unload",
            "status",
            "run MODEL",
            "show MODEL",
            "--key-file",
            "Exit codes",
        ] {
            assert!(out(&o).contains(word), "{args:?} lacks {word}");
        }
        assert!(
            out(&o).contains("--check-config"),
            "the service flags stay documented"
        );
    }
    // the old flags keep working
    let version = eliezer(&["--version"], &[], None);
    assert_eq!(code(&version), 0);
    assert!(out(&version).starts_with("Eliezer "));
}

// ---- (c) real service ----------------------------------------------------------------------

struct Service(Child);
impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
fn start_service(dir: &Path, port: u16, keys: Option<&Path>) -> (Service, PathBuf) {
    fs::create_dir_all(dir.join("models")).unwrap();
    let auth = keys.map_or(String::new(), |k| {
        format!("api_keys_file=\"{}\"\n", k.display())
    });
    let config = format!(
        "[server]\nlisten=\"127.0.0.1:{port}\"\n{auth}[storage]\nmodels_dir=\"{d}/models\"\n[huggingface]\ncache_dir=\"{d}/models/hf\"\n[gpu]\nvendor=\"none\"\nmeasure=\"none\"\nvram_budget_gb=8.0\nreserve_gb=1.0\n[backend.mistralrs]\nenabled=true\n[[model]]\nkey=\"alpha\"\nbackend=\"mistralrs\"\nsource={{hf=\"t/alpha\"}}\nkind=\"llm\"\nvram_gb=1.0\nsize_gb=2.5\n[[model]]\nkey=\"beta\"\nbackend=\"mistralrs\"\nsource={{hf=\"t/beta\"}}\nkind=\"embedding\"\nvram_gb=0.5\n",
        d = dir.display()
    );
    let path = dir.join("config.toml");
    fs::write(&path, config).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .arg("--config")
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let service = Service(child);
    let deadline = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "service did not start");
        std::thread::sleep(Duration::from_millis(50));
    }
    (service, path)
}

#[test]
fn end_to_end_against_the_real_service_with_an_api_key() {
    let dir = scratch("e2e");
    let port = free_port();
    let keys = key_file(&dir, "sk-e2e-key-987");
    let (_service, config) = start_service(&dir, port, Some(&keys));
    let config = config.to_str().unwrap().to_owned();
    // The configuration names the key file and the listen address: no further options needed.
    let run = |args: &[&str]| {
        let mut all = vec!["--config", config.as_str()];
        all.extend_from_slice(args);
        let o = eliezer(&all, &[], None);
        let text = out(&o) + &err(&o);
        assert!(
            !text.contains("sk-e2e-key-987"),
            "the key must never be printed: {text}"
        );
        o
    };
    let ls = run(&["ls"]);
    assert_eq!(code(&ls), 0, "{}", err(&ls));
    assert!(
        out(&ls).contains("alpha") && out(&ls).contains("2.5 GB") && out(&ls).contains("embedding"),
        "{}",
        out(&ls)
    );
    let ps = run(&["ps"]);
    assert_eq!(code(&ps), 0);
    assert!(out(&ps).contains("No models are currently loaded."));
    let status = run(&["status"]);
    assert_eq!(code(&status), 0, "{}", err(&status));
    assert!(
        out(&status).contains("is running (health: ok)")
            && out(&status).contains("0 loaded of 2 configured"),
        "{}",
        out(&status)
    );
    let show = run(&["show", "alpha"]);
    assert_eq!(code(&show), 0);
    assert!(out(&show).contains("alpha") && out(&show).contains("unloaded"));
    let unload = run(&["unload", "--all"]);
    assert_eq!(code(&unload), 0);
    assert!(out(&unload).contains("No models are currently loaded."));
    // This build has no model runtime: loading fails cleanly.
    let load = run(&["load", "alpha"]);
    assert_eq!(code(&load), 1, "{}{}", out(&load), err(&load));
    assert!(
        err(&load).starts_with("Loading alpha")
            && err(&load).contains("Error: ")
            && err(&load).contains("HTTP 502"),
        "{}",
        err(&load)
    );
    let unknown = run(&["load", "nothing"]);
    assert_eq!(code(&unknown), 1);
    assert!(err(&unknown).contains("Error: "));
    let chat = run(&["run", "nothing", "hello"]);
    assert_eq!(code(&chat), 1);

    // without a key (no configuration, explicit host) and with a wrong one
    let url = format!("http://127.0.0.1:{port}");
    let anonymous = eliezer(&["ls", "--host", &url], &[], None);
    assert_eq!(code(&anonymous), 1);
    assert!(
        err(&anonymous).contains("requires an API key") && err(&anonymous).contains("HTTP 401"),
        "{}",
        err(&anonymous)
    );
    let wrong = key_file(&scratch("e2e-wrong"), "sk-not-the-key-1");
    let rejected = eliezer(
        &["ls", "--host", &url, "--key-file", wrong.to_str().unwrap()],
        &[],
        None,
    );
    assert_eq!(code(&rejected), 1);
    assert!(
        err(&rejected).contains("did not accept the API key")
            && !err(&rejected).contains("sk-not-the-key-1")
    );
    // the health check needs no key
    let health = eliezer(&["status", "--host", &url], &[], None);
    assert_eq!(code(&health), 1, "status needs the key for /api/v1/status");
    // ELIEZER_HOST and ELIEZER_API_KEY_FILE together
    let by_env = eliezer(
        &["ps"],
        &[
            ("ELIEZER_HOST", &url),
            ("ELIEZER_API_KEY_FILE", keys.to_str().unwrap()),
        ],
        None,
    );
    assert_eq!(code(&by_env), 0, "{}", err(&by_env));
}

#[test]
fn end_to_end_against_an_open_service_and_a_wildcard_listen_address() {
    let dir = scratch("open");
    let port = free_port();
    let (_service, config) = start_service(&dir, port, None);
    // rewrite: a 0.0.0.0 listen address in the file means "connect to 127.0.0.1"
    let text = fs::read_to_string(&config).unwrap();
    let client_config = dir.join("client.toml");
    fs::write(&client_config, text.replace("127.0.0.1", "0.0.0.0")).unwrap();
    let o = eliezer(
        &["--config", client_config.to_str().unwrap(), "ls"],
        &[],
        None,
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert!(out(&o).contains("alpha"));
    let json = eliezer(
        &["--config", config.to_str().unwrap(), "status", "--json"],
        &[],
        None,
    );
    let parsed: Value = serde_json::from_str(&out(&json)).unwrap();
    assert_eq!(parsed["health"], "ok");
    assert!(parsed["status"]["gpu"]["budget_bytes"].as_u64().is_some());
}

#[test]
fn a_foreign_server_on_the_port_is_refused_not_listed() {
    // LM Studio answers /api/v1/models on 1234 too; `ls` must not show its models as ours.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let Ok(mut stream) = stream else { return };
            let mut buffer = [0_u8; 2048];
            let _ = std::io::Read::read(&mut stream, &mut buffer);
            let body = r#"{"models":[{"key":"lm-studio-model","type":"llm"}]}"#;
            let _ = std::io::Write::write_all(
                &mut stream,
                format!(
                    "HTTP/1.1 200 OK\r\nX-Powered-By: Express\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            );
        }
    });
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .env("XDG_CONFIG_HOME", "/nonexistent/config")
        .args(["ls", "--host", &format!("http://127.0.0.1:{port}")])
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8_lossy(&result.stderr);
    assert!(text.contains("is not Eliezer"), "{text}");
    assert!(
        !String::from_utf8_lossy(&result.stdout).contains("lm-studio-model"),
        "the foreign models must not be listed"
    );
}

#[test]
fn status_warns_when_the_budget_is_far_below_the_card() {
    // Pure formatting: call through the binary against a canned server instead of internals.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let Ok(mut stream) = stream else { return };
            let mut buffer = [0_u8; 2048];
            let n = std::io::Read::read(&mut stream, &mut buffer).unwrap_or(0);
            let request = String::from_utf8_lossy(&buffer[..n]).into_owned();
            let body = if request.contains("/healthz") {
                "ok".to_owned()
            } else {
                r#"{"gpu":{"probe":"sysfs","total_bytes":25753026560,"used_bytes":1100000000,"free_bytes":24600000000,"budget_bytes":8000000000},"models":[],"queue":[]}"#.to_owned()
            };
            let _ = std::io::Write::write_all(
                &mut stream,
                format!(
                    "HTTP/1.1 200 OK\r\nServer: eliezer/test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            );
        }
    });
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .env("XDG_CONFIG_HOME", "/nonexistent/config")
        .args(["status", "--host", &format!("http://127.0.0.1:{port}")])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(result.status.success(), "{text}");
    assert!(
        text.contains("budget (8.0 GB) is far below") && text.contains("vram_budget_gb"),
        "{text}"
    );
}

#[test]
fn a_server_that_accepts_but_never_answers_is_named_as_stopped_or_hung() {
    // A process stopped with Ctrl-Z keeps its listening socket but accepts nothing.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let started = std::time::Instant::now();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .env("XDG_CONFIG_HOME", "/nonexistent/config")
        .env("ELIEZER_TIMEOUT_SECS", "1")
        .args(["ls", "--host", &format!("http://127.0.0.1:{port}")])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(1), "{text}");
    assert!(
        text.contains("did not respond") && text.contains("stopped"),
        "{text}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "the wait is bounded"
    );
    drop(listener);
}

#[test]
fn binding_an_occupied_address_names_the_address_and_the_reason() {
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port();
    let dir = std::env::temp_dir().join(format!("eliezer-bind-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let config = dir.join("config.toml");
    std::fs::write(
        &config,
        format!(
            "[server]\nlisten = \"127.0.0.1:{port}\"\n[storage]\nmodels_dir = \"{d}/models\"\n[huggingface]\ncache_dir = \"{d}/models/hf\"\n[gpu]\n",
            d = dir.display()
        ),
    )
    .unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .env("XDG_CONFIG_HOME", "/nonexistent/config")
        .args(["--config", config.to_str().unwrap()])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&result.stderr).to_string()
        + &String::from_utf8_lossy(&result.stdout);
    assert_eq!(result.status.code(), Some(1), "{text}");
    assert!(
        text.contains(&format!("127.0.0.1:{port} cannot be bound")),
        "{text}"
    );
    assert!(
        text.contains("already in use") || text.contains("Address in use"),
        "{text}"
    );
    assert!(text.contains(&format!("ss -ltnp | grep :{port}")), "{text}");
    std::fs::remove_dir_all(dir).unwrap();
}

fn hub_detail() -> Value {
    json!({
        "repo": "org/Qwen-GGUF", "sha": "abc123", "task": "text-generation",
        "license": "apache-2.0", "gated": false, "downloads": 100,
        "architecture": "qwen3moe",
        "support": {"level": "verified", "backend": "llamacpp"},
        "variants": [
            {"file": "q-Q4_K_M.gguf", "quantization": "Q4_K_M", "bytes": 4_000_000_000_u64,
             "parts": 1, "estimate_bytes": 5_300_000_000_u64, "fit": "gpu"},
            {"file": "q-F16.gguf", "quantization": "F16", "bytes": 20_000_000_000_u64,
             "parts": 1, "estimate_bytes": 24_500_000_000_u64, "fit": "too_large"}],
        "projector": {"file": "mmproj-q-F16.gguf", "bytes": 500_000_000},
        "required_files": [{"file": "vae/v.safetensors", "bytes": 700_000_000}]
    })
}

#[test]
fn search_and_inspect_print_candidates_and_send_encoded_requests() {
    let server = serve(Box::new(|r| {
        if r.path.starts_with("/api/v1/hub/search") {
            json_reply(
                "200 OK",
                &json!({"object": "list", "data": [
                    {"repo": "org/Qwen-GGUF", "downloads": 1234, "likes": 1, "task": "text-generation",
                     "license": "apache-2.0", "gguf": true, "detail": hub_detail()},
                    {"repo": "org/Other-GGUF", "downloads": 5, "likes": 0, "task": null,
                     "license": null, "gguf": true, "detail": null}]}),
            )
        } else if r.path == "/api/v1/hub/models/org/Qwen-GGUF" {
            json_reply("200 OK", &hub_detail())
        } else {
            json_reply(
                "404 Not Found",
                &json!({"error": {"type": "not_found", "message": "The repository was not found on Hugging Face"}}),
            )
        }
    }));
    let url = server.url();
    let run = |args: &[&str]| {
        let mut all = vec!["--host", url.as_str()];
        all.extend_from_slice(args);
        eliezer(&all, &[], None)
    };
    let search = run(&[
        "search",
        "qwen",
        "3 coder&x=1",
        "--task",
        "llm",
        "--license",
        "apache-2.0",
        "--limit",
        "5",
        "--details",
    ]);
    assert_eq!(code(&search), 0, "{}", err(&search));
    let text = out(&search);
    assert!(
        text.contains("REPOSITORY") && text.contains("SUPPORT") && text.contains("FITS"),
        "{text}"
    );
    assert!(
        text.contains("org/Qwen-GGUF") && text.contains("verified") && text.contains("gpu"),
        "{text}"
    );
    assert!(text.contains("Nothing is downloaded"));
    let path = &server.requests()[0].path;
    // the text cannot add parameters: spaces and `&` and `=` are encoded
    assert!(path.contains("q=qwen%203%20coder%26x%3D1"), "{path}");
    for part in ["task=llm", "license=apache-2.0", "limit=5", "details=true"] {
        assert!(path.contains(part), "{path}");
    }

    let inspect = run(&["inspect", "org/Qwen-GGUF"]);
    assert_eq!(code(&inspect), 0, "{}", err(&inspect));
    let text = out(&inspect);
    for part in [
        "qwen3moe",
        "verified  (llamacpp)",
        "Q4_K_M",
        "5.3 GB",
        "too large",
        "mmproj-q-F16.gguf",
        "vae/v.safetensors",
        "Nothing is downloaded",
    ] {
        assert!(text.contains(part), "{part} missing in\n{text}");
    }
    let missing = run(&["inspect", "org/Missing"]);
    assert_eq!(code(&missing), 1);
    assert!(err(&missing).contains("not found"), "{}", err(&missing));
    assert_eq!(code(&run(&["inspect"])), 2);
    let json = run(&["inspect", "org/Qwen-GGUF", "--json"]);
    assert_eq!(
        serde_json::from_str::<Value>(&out(&json)).unwrap()["sha"],
        "abc123"
    );
}

fn variants_detail() -> Value {
    json!({
        "repo": "org/Model-GGUF", "sha": "0123456789abcdef0123456789abcdef01234567",
        "variants": [
            {"file": "m-Q8_0.gguf", "files": ["m-Q8_0.gguf"], "quantization": "Q8_0", "bytes": 8_000_000_000_u64, "fit": "gpu"},
            {"file": "m-Q4_K_M.gguf", "files": ["m-Q4_K_M.gguf"], "quantization": "Q4_K_M", "bytes": 4_400_000_000_u64, "fit": "gpu"},
            {"file": "m-F16.gguf", "files": ["m-F16-00001-of-00002.gguf", "m-F16-00002-of-00002.gguf"],
             "quantization": "F16", "bytes": 28_000_000_000_u64, "fit": "too_large"}],
        "projector": {"file": "mmproj-m-F16.gguf", "bytes": 900_000_000}
    })
}

#[test]
fn get_picks_the_files_by_quantization_and_fit() {
    let detail = variants_detail();
    // the preferred quantization that fits, and the vision projector
    assert_eq!(
        format::choose_files(&detail, None, &[]).unwrap(),
        ["m-Q4_K_M.gguf", "mmproj-m-F16.gguf"]
    );
    // a named one, case does not matter, and a split file brings all its parts
    assert_eq!(
        format::choose_files(&detail, Some("q8_0"), &["vae/v.safetensors".to_owned()]).unwrap(),
        ["m-Q8_0.gguf", "mmproj-m-F16.gguf", "vae/v.safetensors"]
    );
    assert_eq!(
        format::choose_files(&detail, Some("F16"), &[]).unwrap()[..2],
        ["m-F16-00001-of-00002.gguf", "m-F16-00002-of-00002.gguf"]
    );
    let missing = format::choose_files(&detail, Some("Q2_K"), &[]).unwrap_err();
    assert!(
        missing.contains("no Q2_K file") && missing.contains("Q4_K_M"),
        "{missing}"
    );
    // nothing fits: say so and list what there is
    let mut big = variants_detail();
    for v in big["variants"].as_array_mut().unwrap() {
        v["fit"] = json!("too_large");
    }
    let refused = format::choose_files(&big, None, &[]).unwrap_err();
    assert!(
        refused.contains("fits the memory budgets") && refused.contains("Q8_0"),
        "{refused}"
    );
    // a single file is taken whatever it is; none is an error
    let one = json!({"repo": "o/r", "variants": [{"files": ["only.gguf"], "quantization": null, "fit": "gpu"}]});
    assert_eq!(
        format::choose_files(&one, None, &[]).unwrap(),
        ["only.gguf"]
    );
    assert!(format::choose_files(&json!({"repo": "o/r", "variants": []}), None, &[]).is_err());
}

fn job_json(state: &str, error: Option<&str>) -> Value {
    json!({"id": "dl-1", "repo": "org/Model-GGUF", "revision": "0123456789abcdef0123456789abcdef01234567",
           "directory": "/home/eliezer/models/org/Model-GGUF", "state": state, "error": error,
           "total_bytes": 5_300_000_000_u64, "done_bytes": 5_300_000_000_u64,
           "files": [{"file": "m-Q4_K_M.gguf", "total_bytes": 4_400_000_000_u64, "done_bytes": 4_400_000_000_u64, "state": "done", "error": null},
                     {"file": "mmproj-m-F16.gguf", "total_bytes": 900_000_000_u64, "done_bytes": 900_000_000_u64, "state": "done", "error": null}]})
}

#[test]
fn get_and_download_follow_the_job_and_report_failures() {
    let server = serve(Box::new(|r| match (r.method.as_str(), r.path.as_str()) {
        ("GET", "/api/v1/hub/models/org/Model-GGUF") => json_reply("200 OK", &variants_detail()),
        ("POST", "/api/v1/models/download") => {
            let body: Value = serde_json::from_str(&r.body).unwrap();
            if body["repo"] == "org/Denied" {
                return json_reply(
                    "403 Forbidden",
                    &json!({"error": {"type": "not_allowed", "message": "This repository is not on the allowlist ([huggingface] allow)"}}),
                );
            }
            json_reply("202 Accepted", &job_json("queued", None))
        }
        ("GET", "/api/v1/models/download/dl-1") => {
            // the second kind of request in this test fails
            json_reply("200 OK", &job_json("done", None))
        }
        ("GET", "/api/v1/models/pins") => json_reply(
            "200 OK",
            &json!({"pins": [{"repo": "org/Model-GGUF", "files": [
                {"file": "m-Q4_K_M.gguf", "bytes": 4_400_000_000_u64, "revision": "0123456789abcdef", "sha256": null, "status": "ok", "found_bytes": null},
                {"file": "x.gguf", "bytes": 10, "revision": "0123456789abcdef", "sha256": null, "status": "size_mismatch", "found_bytes": 5}]}],
                "verified": false, "deviations": 1}),
        ),
        _ => json_reply(
            "404 Not Found",
            &json!({"error": {"type": "not_found", "message": "no"}}),
        ),
    }));
    let url = server.url();
    let run = |args: &[&str]| {
        let mut all = vec!["--host", url.as_str()];
        all.extend_from_slice(args);
        eliezer(&all, &[], None)
    };
    let got = run(&["get", "org/Model-GGUF"]);
    assert_eq!(code(&got), 0, "{}", err(&got));
    let text = out(&got);
    for part in [
        "Fetching 2 file(s)",
        "m-Q4_K_M.gguf",
        "mmproj-m-F16.gguf",
        "commit 01234567",
        "Done.",
        "/home/eliezer/models/org/Model-GGUF",
        "--suggest",
    ] {
        assert!(text.contains(part), "{part} missing in\n{text}");
    }
    let posted: Value = serde_json::from_str(
        &server
            .requests()
            .iter()
            .find(|r| r.method == "POST")
            .unwrap()
            .body,
    )
    .unwrap();
    assert_eq!(
        posted["files"],
        json!(["m-Q4_K_M.gguf", "mmproj-m-F16.gguf"])
    );
    assert_eq!(
        posted["revision"], "0123456789abcdef0123456789abcdef01234567",
        "the listing is pinned"
    );

    let named = run(&[
        "download",
        "org/Model-GGUF",
        "a.gguf",
        "b/c.gguf",
        "--revision",
        &"f".repeat(40),
    ]);
    assert_eq!(code(&named), 0, "{}", err(&named));
    let last: Value = serde_json::from_str(
        &server
            .requests()
            .iter()
            .rev()
            .find(|r| r.method == "POST")
            .unwrap()
            .body,
    )
    .unwrap();
    assert_eq!(last["files"], json!(["a.gguf", "b/c.gguf"]));
    assert_eq!(last["revision"], "f".repeat(40));

    let denied = run(&["download", "org/Denied", "a.gguf"]);
    assert_eq!(code(&denied), 1);
    assert!(err(&denied).contains("allowlist"), "{}", err(&denied));
    assert_eq!(
        code(&run(&["download", "org/Model-GGUF"])),
        2,
        "a file name is required"
    );
    assert_eq!(code(&run(&["get"])), 2);

    let pins = run(&["pins"]);
    assert_eq!(code(&pins), 0, "{}", err(&pins));
    let text = out(&pins);
    assert!(
        text.contains("REPOSITORY")
            && text.contains("size mismatch")
            && text.contains("1 file(s) differ"),
        "{text}"
    );
}

#[test]
fn a_failed_download_ends_with_the_reason_and_exit_code_1() {
    let server = serve(Box::new(|r| match (r.method.as_str(), r.path.as_str()) {
        ("POST", "/api/v1/models/download") => {
            json_reply("202 Accepted", &job_json("queued", None))
        }
        ("GET", "/api/v1/models/download/dl-1") => json_reply(
            "200 OK",
            &job_json(
                "failed",
                Some("The checksum of the download does not match; the file was discarded"),
            ),
        ),
        _ => json_reply(
            "404 Not Found",
            &json!({"error": {"type": "not_found", "message": "no"}}),
        ),
    }));
    let result = eliezer(
        &[
            "--host",
            &server.url(),
            "download",
            "org/Model-GGUF",
            "a.gguf",
        ],
        &[],
        None,
    );
    assert_eq!(code(&result), 1);
    assert!(
        err(&result).contains("The download failed: The checksum"),
        "{}",
        err(&result)
    );
}

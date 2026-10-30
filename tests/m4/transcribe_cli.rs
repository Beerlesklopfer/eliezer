//! `eliezer transcribe`: the pure parts (collecting files, the upload body, the four output formats)
//! and the command against a fake service on loopback. No network, no GPU.
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[allow(dead_code)]
#[path = "../../crates/eliezer/src/cli/transcribe.rs"]
mod transcribe;
use transcribe as t;

// ---- (a) pure functions --------------------------------------------------------------------

const ANSWER: &str = r#"{"task":"transcribe","language":"de","duration":3725.5,"text":"Hallo Welt. Zweite Zeile, ok.","segments":[{"id":0,"start":0.0,"end":2.5,"text":" Hallo Welt."},{"id":1,"start":3661.2504,"end":3725.5,"text":" Zweite Zeile, ok. "}]}"#;

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("eliezer-transcribe-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn touch(path: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, b"audio").unwrap();
}

#[test]
fn the_four_formats_are_built_from_one_verbose_answer() {
    let transcript = t::parse_transcript(ANSWER).unwrap();
    assert_eq!(transcript.duration, 3725.5);
    assert_eq!(
        t::render("txt", ANSWER, &transcript),
        "Hallo Welt.\nZweite Zeile, ok.\n"
    );
    assert_eq!(
        t::render("srt", ANSWER, &transcript),
        "1\n00:00:00,000 --> 00:00:02,500\nHallo Welt.\n\n2\n01:01:01,250 --> 01:02:05,500\nZweite Zeile, ok.\n\n"
    );
    assert_eq!(
        t::render("vtt", ANSWER, &transcript),
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHallo Welt.\n\n01:01:01.250 --> 01:02:05.500\nZweite Zeile, ok.\n\n"
    );
    // json is the answer exactly as received
    assert_eq!(t::render("json", ANSWER, &transcript), ANSWER);
}

#[test]
fn an_answer_without_segments_becomes_one_segment_and_garbage_is_refused() {
    let one = t::parse_transcript(r#"{"text":"Nur Text","duration":4.0}"#).unwrap();
    assert_eq!(one.segments.len(), 1);
    assert_eq!(t::txt(&one), "Nur Text\n");
    assert!(t::parse_transcript("not json").is_err());
    assert!(t::parse_transcript(r#"{"duration":1.0}"#).is_err());
    assert!(t::parse_transcript(r#"{"segments":[{"start":"x"}]}"#).is_err());
}

#[test]
fn timestamps_are_zero_padded_with_comma_or_dot() {
    assert_eq!(t::timestamp(0.0, ','), "00:00:00,000");
    assert_eq!(t::timestamp(5.0005, '.'), "00:00:05.001");
    assert_eq!(t::timestamp(3599.9996, '.'), "01:00:00.000");
    assert_eq!(t::timestamp(360000.0, ','), "100:00:00,000");
    assert_eq!(t::timestamp(-1.0, ','), "00:00:00,000");
}

#[test]
fn format_and_extension_lists_are_checked() {
    assert_eq!(t::parse_formats("txt, SRT,txt").unwrap(), ["txt", "srt"]);
    assert!(t::parse_formats("txt,pdf").is_err());
    assert_eq!(t::parse_extensions(".MP3,flac,").unwrap(), ["mp3", "flac"]);
    assert!(t::parse_extensions(" , ").is_err());
    assert_eq!(
        t::DEFAULT_EXTENSIONS,
        ["mp3", "m4a", "aac", "ogg", "oga", "opus", "flac", "wav"]
    );
}

#[test]
fn collecting_honours_extensions_recursion_and_hidden_files() {
    let dir = scratch("collect");
    for name in [
        "b.mp3",
        "a.FLAC",
        "notes.txt",
        ".hidden.mp3",
        "sub/c.wav",
        "sub/deeper/d.ogg",
        ".secret/e.mp3",
    ] {
        touch(&dir.join(name));
    }
    let ext: Vec<String> = t::DEFAULT_EXTENSIONS
        .iter()
        .map(|e| (*e).to_owned())
        .collect();
    let names = |items: &[t::Item]| -> Vec<String> { items.iter().map(t::Item::display).collect() };

    let flat = t::collect(std::slice::from_ref(&dir), false, &ext).unwrap();
    assert_eq!(names(&flat), ["a.FLAC", "b.mp3"]);
    let deep = t::collect(std::slice::from_ref(&dir), true, &ext).unwrap();
    assert_eq!(
        names(&deep),
        ["a.FLAC", "b.mp3", "sub/c.wav", "sub/deeper/d.ogg"]
    );
    assert_eq!(deep[2].rel_dir, Path::new("sub"));
    assert_eq!(
        deep[2].output(Path::new("/out"), "srt"),
        Path::new("/out/sub/c.srt")
    );
    let only = t::collect(std::slice::from_ref(&dir), true, &["wav".to_owned()]).unwrap();
    assert_eq!(names(&only), ["sub/c.wav"]);
    // a named file is taken whatever its extension, and a dotted stem keeps its dots
    touch(&dir.join("x.y.z"));
    let named = t::collect(&[dir.join("x.y.z")], false, &ext).unwrap();
    assert_eq!(named[0].stem, "x.y");
    assert!(t::collect(&[dir.join("missing.mp3")], false, &ext).is_err());
}

#[test]
fn two_inputs_with_the_same_output_name_are_refused() {
    let dir = scratch("clash");
    touch(&dir.join("a.mp3"));
    touch(&dir.join("a.m4a"));
    let ext: Vec<String> = t::DEFAULT_EXTENSIONS
        .iter()
        .map(|e| (*e).to_owned())
        .collect();
    let error = t::collect(&[dir], false, &ext).unwrap_err();
    assert!(error.contains("same output"), "{error}");
}

#[test]
fn the_upload_escapes_the_file_name() {
    assert_eq!(
        t::quote_filename("a \"q\"\r\n,b;ä.mp3"),
        "a %22q%22%0D%0A,b;ä.mp3"
    );
    let (content_type, body) = t::multipart(
        &[("model", "w"), ("prompt", "Jörg, \"Z\"")],
        "a,b ä.mp3",
        b"DATA",
    );
    let boundary = content_type
        .strip_prefix("multipart/form-data; boundary=")
        .unwrap();
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("filename=\"a,b ä.mp3\""), "{text}");
    assert!(
        text.contains("name=\"prompt\"\r\n\r\nJörg, \"Z\"\r\n"),
        "{text}"
    );
    assert!(text.ends_with(&format!("DATA\r\n--{boundary}--\r\n")));
    // the boundary never occurs in the data
    let data = format!("xx--{boundary}yy");
    let (content_type, _) = t::multipart(&[], "f.mp3", data.as_bytes());
    assert!(
        !data.contains(
            content_type
                .strip_prefix("multipart/form-data; boundary=")
                .unwrap()
        )
    );
}

// ---- (b) the command against a fake service -------------------------------------------------

#[derive(Clone, Debug)]
struct Seen {
    method: String,
    path: String,
    auth: Option<String>,
    body: Vec<u8>,
}
impl Seen {
    /// The text of a multipart field.
    fn field(&self, name: &str) -> Option<String> {
        let text = String::from_utf8_lossy(&self.body).into_owned();
        let marker = format!("name=\"{name}\"\r\n\r\n");
        let start = text.find(&marker)? + marker.len();
        Some(text[start..text[start..].find("\r\n")? + start].to_owned())
    }
    fn file_name(&self) -> Option<String> {
        let text = String::from_utf8_lossy(&self.body).into_owned();
        let start = text.find("filename=\"")? + 10;
        let end = text[start..].find("\"\r\n")? + start;
        Some(text[start..end].to_owned())
    }
}
type Handler = Box<dyn Fn(&Seen) -> Vec<u8> + Send + Sync>;
struct Fake {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}
impl Fake {
    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
    fn requests(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
    fn uploads(&self) -> Vec<Seen> {
        self.requests()
            .into_iter()
            .filter(|r| r.path == "/v1/audio/transcriptions")
            .collect()
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
/// Serves `connections` requests (all of them when `None`), then closes the listener.
fn serve(handler: Handler, connections: Option<usize>) -> Fake {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        let mut left = connections;
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
                body,
            };
            log.lock().unwrap().push(request.clone());
            let _ = stream.write_all(&handler(&request));
            left = left.map(|n| n - 1);
            if left == Some(0) {
                break;
            }
        }
    });
    Fake { port, seen }
}

fn models(types: &[(&str, &str)]) -> Value {
    json!({"models": types.iter().map(|(t, k)| json!({"type": t, "key": k})).collect::<Vec<_>>()})
}
/// A service with the given models that answers every upload with the canned answer.
fn service(types: &[(&str, &str)]) -> Fake {
    let list = models(types);
    serve(
        Box::new(move |r| match r.path.as_str() {
            "/api/v1/models" => json_reply("200 OK", &list),
            "/v1/audio/transcriptions" => reply("200 OK", "application/json", ANSWER),
            _ => json_reply(
                "404 Not Found",
                &json!({"error": {"message": "no such path"}}),
            ),
        }),
        None,
    )
}

fn command(args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eliezer"));
    command
        .env_remove("ELIEZER_HOST")
        .env_remove("ELIEZER_API_KEY")
        .env_remove("ELIEZER_API_KEY_FILE")
        .env_remove("ELIEZER_TIMEOUT_SECS")
        .env("XDG_CONFIG_HOME", "/nonexistent/config")
        .env("ELIEZER_CONFIG", "/nonexistent/eliezer.toml")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}
fn run(args: &[&str]) -> Output {
    command(args).output().unwrap()
}
fn transcribe(fake: &Fake, args: &[&str]) -> Output {
    let url = fake.url();
    let mut all = vec!["--host", url.as_str(), "transcribe"];
    all.extend_from_slice(args);
    run(&all)
}
fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}
fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}
fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}
fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

#[test]
fn one_run_writes_all_four_formats_for_awkward_file_names() {
    let dir = scratch("basic");
    let (input, output) = (dir.join("in"), dir.join("out"));
    let names = [
        "Mit Leerzeichen.mp3",
        "a,b.m4a",
        "Grüße ä ö ü.flac",
        "He said \"hi\".wav",
    ];
    for name in names {
        touch(&input.join(name));
    }
    let key = dir.join("key");
    fs::write(&key, "sk-test-key\n").unwrap();
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
    let fake = service(&[("llm", "chat"), ("stt", "whisper")]);
    let hints = dir.join("hints.txt");
    fs::write(&hints, "Jörg Bernau, Eliezer\n").unwrap();
    let o = transcribe(
        &fake,
        &[
            s(&input),
            "--out",
            s(&output),
            "--language",
            "de",
            "--prompt-file",
            s(&hints),
            "--key-file",
            s(&key),
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    let mut expected: Vec<String> = Vec::new();
    for stem in ["Mit Leerzeichen", "a,b", "Grüße ä ö ü", "He said \"hi\""] {
        for ext in ["json", "srt", "txt", "vtt"] {
            expected.push(format!("{stem}.{ext}"));
        }
    }
    expected.sort();
    assert_eq!(files_in(&output), expected);
    let srt = fs::read_to_string(output.join("a,b.srt")).unwrap();
    assert!(
        srt.starts_with("1\n00:00:00,000 --> 00:00:02,500\nHallo Welt.\n"),
        "{srt}"
    );
    assert_eq!(fs::read_to_string(output.join("a,b.json")).unwrap(), ANSWER);
    assert!(
        fs::read_to_string(output.join("a,b.vtt"))
            .unwrap()
            .starts_with("WEBVTT\n\n")
    );

    let uploads = fake.uploads();
    assert_eq!(uploads.len(), 4, "one request per file");
    let mut sent: Vec<String> = uploads.iter().map(|u| u.file_name().unwrap()).collect();
    sent.sort();
    // the quote is percent-encoded, everything else arrives as is
    assert!(sent.contains(&"a,b.m4a".to_owned()) && sent.contains(&"Grüße ä ö ü.flac".to_owned()));
    assert!(
        sent.contains(&"He said %22hi%22.wav".to_owned()),
        "{sent:?}"
    );
    for u in &uploads {
        assert_eq!(u.method, "POST");
        assert_eq!(u.auth.as_deref(), Some("Bearer sk-test-key"));
        assert_eq!(u.field("model").as_deref(), Some("whisper"));
        assert_eq!(u.field("response_format").as_deref(), Some("verbose_json"));
        assert_eq!(u.field("language").as_deref(), Some("de"));
        assert_eq!(u.field("prompt").as_deref(), Some("Jörg Bernau, Eliezer"));
    }
    let log = err(&o);
    assert!(
        log.contains("[1/4]") && log.contains("[4/4]") && log.contains("done"),
        "{log}"
    );
    assert!(
        log.contains("audio 1h 02m") && log.contains("x  done"),
        "{log}"
    );
    assert!(
        log.contains("Summary: 4 done, 0 skipped, 0 failed; audio 4h 08m"),
        "{log}"
    );
    assert!(out(&o).is_empty(), "progress goes to stderr");
}

#[test]
fn a_finished_file_is_skipped_and_a_partial_one_gets_only_the_missing_formats() {
    let dir = scratch("resume");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    touch(&input.join("b.mp3"));
    let fake = service(&[("stt", "whisper")]);
    let args = [s(&input), "--out", s(&output), "--format", "txt,srt"];
    assert_eq!(code(&transcribe(&fake, &args)), 0);
    assert_eq!(fake.uploads().len(), 2);

    // nothing to do: no upload and not even the model list
    let before = fake.requests().len();
    let again = transcribe(&fake, &args);
    assert_eq!(code(&again), 0, "{}", err(&again));
    assert_eq!(fake.requests().len(), before);
    assert!(
        err(&again).contains("skipped (outputs exist)")
            && err(&again).contains("0 done, 2 skipped")
    );

    // b lost its srt: only b is sent, and a's outputs stay untouched
    fs::write(output.join("a.txt"), "mine").unwrap();
    fs::remove_file(output.join("b.srt")).unwrap();
    fs::write(output.join("b.txt"), "mine too").unwrap();
    let third = transcribe(&fake, &args);
    assert_eq!(code(&third), 0, "{}", err(&third));
    assert_eq!(fake.uploads().len(), 3);
    assert_eq!(fs::read_to_string(output.join("a.txt")).unwrap(), "mine");
    assert_eq!(
        fs::read_to_string(output.join("b.txt")).unwrap(),
        "mine too"
    );
    assert!(output.join("b.srt").exists());
}

#[test]
fn replace_overwrites_existing_outputs() {
    let dir = scratch("replace");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join("a.txt"), "old").unwrap();
    let fake = service(&[("stt", "whisper")]);
    let args = [s(&input), "--out", s(&output), "--format", "txt"];
    assert_eq!(code(&transcribe(&fake, &args)), 0);
    assert_eq!(fs::read_to_string(output.join("a.txt")).unwrap(), "old");
    let mut with = args.to_vec();
    with.push("--replace");
    assert_eq!(code(&transcribe(&fake, &with)), 0);
    assert_eq!(
        fs::read_to_string(output.join("a.txt")).unwrap(),
        "Hallo Welt.\nZweite Zeile, ok.\n"
    );
}

#[test]
fn recursive_keeps_the_directory_structure_and_ignores_hidden_files() {
    let dir = scratch("tree");
    let (input, output) = (dir.join("in"), dir.join("out"));
    for name in [
        "top.mp3",
        "one/two.mp3",
        ".hidden.mp3",
        "one/.x.mp3",
        "one/note.txt",
    ] {
        touch(&input.join(name));
    }
    let fake = service(&[("stt", "whisper")]);
    let o = transcribe(
        &fake,
        &[
            s(&input),
            "--out",
            s(&output),
            "--recursive",
            "--format",
            "txt",
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(files_in(&output), ["one", "top.txt"]);
    assert_eq!(files_in(&output.join("one")), ["two.txt"]);
    // without --recursive the sub-directory is left alone
    let flat = dir.join("flat");
    let o = transcribe(
        &fake,
        &[
            s(&input),
            "--out",
            s(&flat),
            "--format",
            "txt",
            "--extensions",
            "mp3",
        ],
    );
    assert_eq!(code(&o), 0);
    assert_eq!(files_in(&flat), ["top.txt"]);
}

#[test]
fn a_failing_file_does_not_stop_the_run_but_sets_exit_code_1() {
    let dir = scratch("failing");
    let (input, output) = (dir.join("in"), dir.join("out"));
    for name in ["a.mp3", "b.mp3", "c.mp3"] {
        touch(&input.join(name));
    }
    let fake = serve(
        Box::new(|r| {
            if r.path == "/api/v1/models" {
                return json_reply("200 OK", &models(&[("stt", "whisper")]));
            }
            if r.file_name().as_deref() == Some("b.mp3") {
                return json_reply(
                    "400 Bad Request",
                    &json!({"error": {"type": "invalid_request", "message": "The file is not audio this service can read"}}),
                );
            }
            reply("200 OK", "application/json", ANSWER)
        }),
        None,
    );
    let o = transcribe(&fake, &[s(&input), "--out", s(&output), "--format", "txt"]);
    assert_eq!(code(&o), 1, "{}", err(&o));
    assert_eq!(files_in(&output), ["a.txt", "c.txt"]);
    let log = err(&o);
    assert!(
        log.contains("[2/3] b.mp3") && log.contains("failed: The file is not audio"),
        "{log}"
    );
    assert!(log.contains("2 done, 0 skipped, 1 failed"), "{log}");
    assert!(
        log.contains("Failed files:\n  b.mp3: The file is not audio"),
        "{log}"
    );
}

#[test]
fn an_unreadable_output_directory_fails_the_file_and_leaves_no_part_file() {
    let dir = scratch("readonly");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    fs::create_dir_all(&output).unwrap();
    fs::set_permissions(&output, fs::Permissions::from_mode(0o500)).unwrap();
    let writable = fs::write(output.join("probe"), b"x").is_ok();
    let fake = service(&[("stt", "whisper")]);
    let o = transcribe(&fake, &[s(&input), "--out", s(&output), "--format", "txt"]);
    fs::set_permissions(&output, fs::Permissions::from_mode(0o700)).unwrap();
    if !writable {
        assert_eq!(code(&o), 1, "{}", err(&o));
        assert!(err(&o).contains("Cannot write"), "{}", err(&o));
    }
    assert!(files_in(&output).iter().all(|n| !n.ends_with(".part")));
}

#[test]
fn a_successful_run_leaves_no_part_files() {
    let dir = scratch("atomic");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    let fake = service(&[("stt", "whisper")]);
    assert_eq!(
        code(&transcribe(&fake, &[s(&input), "--out", s(&output)])),
        0
    );
    assert_eq!(files_in(&output), ["a.json", "a.srt", "a.txt", "a.vtt"]);
}

#[test]
fn an_unreachable_service_is_exit_code_3() {
    let dir = scratch("down");
    let input = dir.join("in");
    touch(&input.join("a.mp3"));
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let host = format!("http://127.0.0.1:{port}");
    let out_dir = dir.join("out");
    for extra in [vec![], vec!["--model", "whisper"]] {
        let mut args = vec![
            "--host",
            host.as_str(),
            "transcribe",
            s(&input),
            "--out",
            s(&out_dir),
        ];
        args.extend(extra);
        let o = run(&args);
        assert_eq!(code(&o), 3, "{}", err(&o));
    }
}

#[test]
fn a_service_that_disappears_stops_the_run_after_the_current_file() {
    let dir = scratch("gone");
    let (input, output) = (dir.join("in"), dir.join("out"));
    for name in ["a.mp3", "b.mp3", "c.mp3"] {
        touch(&input.join(name));
    }
    // answers the first upload and then closes its port
    let fake = serve(
        Box::new(|_| reply("200 OK", "application/json", ANSWER)),
        Some(1),
    );
    let o = transcribe(
        &fake,
        &[
            s(&input),
            "--out",
            s(&output),
            "--model",
            "whisper",
            "--format",
            "txt",
        ],
    );
    assert_eq!(code(&o), 3, "{}", err(&o));
    let log = err(&o);
    assert!(
        log.contains("[1/3] a.mp3") && log.contains("[2/3] b.mp3") && !log.contains("[3/3]"),
        "{log}"
    );
    assert!(log.contains("1 done, 0 skipped, 1 failed"), "{log}");
    assert_eq!(files_in(&output), ["a.txt"]);
}

#[test]
fn usage_errors_are_exit_code_2_and_send_nothing() {
    let dir = scratch("usage");
    let input = dir.join("in");
    touch(&input.join("a.mp3"));
    fs::create_dir_all(dir.join("empty")).unwrap();
    let fake = service(&[("stt", "whisper")]);
    let (i, o) = (s(&input), s(&dir.join("out")).to_owned());
    let empty = dir.join("empty");
    let cases: Vec<(Vec<&str>, &str)> = vec![
        (vec![i], "--out DIR is required"),
        (vec![i, "--out", &o, "--format", "txt,pdf"], "--format"),
        (
            vec![i, "--out", &o, "--prompt", "a", "--prompt-file", "b"],
            "not both",
        ),
        (
            vec![i, "--out", &o, "--prompt-file", "/nonexistent/hints"],
            "prompt file",
        ),
        (vec![i, "--out", &o, "--language", "German"], "--language"),
        (vec![i, "--out", &o, "--limit", "0"], "--limit"),
        (
            vec![i, "--out", &o, "--timeout-secs", "x"],
            "--timeout-secs",
        ),
        (vec![i, "--out", &o, "--extensions", ","], "--extensions"),
        (vec![i, "--out", &o, "--bogus"], "Unknown option"),
        (vec!["/nonexistent/file.mp3", "--out", &o], "Cannot read"),
        (vec![s(&empty), "--out", &o], "No audio files found"),
        (vec![], "An argument is missing"),
    ];
    for (args, message) in cases {
        let result = transcribe(&fake, &args);
        assert_eq!(code(&result), 2, "{args:?}: {}", err(&result));
        assert!(err(&result).contains(message), "{args:?}: {}", err(&result));
    }
    assert!(fake.requests().is_empty());
}

#[test]
fn the_model_is_the_only_stt_model_or_named_explicitly() {
    let dir = scratch("model");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    let none = service(&[("llm", "chat")]);
    let o = transcribe(&none, &[s(&input), "--out", s(&output)]);
    assert_eq!(code(&o), 2);
    assert!(err(&o).contains("no speech recognition"), "{}", err(&o));
    assert!(none.uploads().is_empty());

    let several = service(&[("stt", "small"), ("stt", "large"), ("llm", "chat")]);
    let o = transcribe(&several, &[s(&input), "--out", s(&output)]);
    assert_eq!(code(&o), 2);
    assert!(
        err(&o).contains("several") && err(&o).contains("small, large"),
        "{}",
        err(&o)
    );
    assert!(several.uploads().is_empty());

    // --model decides, and the list is not even asked for
    let o = transcribe(
        &several,
        &[
            s(&input),
            "--out",
            s(&output),
            "--model",
            "large",
            "--format",
            "txt",
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(
        several.uploads()[0].field("model").as_deref(),
        Some("large")
    );
    assert!(
        several
            .requests()
            .iter()
            .filter(|r| r.path == "/api/v1/models")
            .count()
            == 1
    );
}

#[test]
fn dry_run_lists_the_work_and_sends_nothing() {
    let dir = scratch("dry");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    touch(&input.join("b.mp3"));
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join("b.txt"), "done").unwrap();
    let fake = service(&[("stt", "whisper")]);
    let o = transcribe(
        &fake,
        &[
            s(&input),
            "--out",
            s(&output),
            "--format",
            "txt",
            "--dry-run",
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    let listing = out(&o);
    assert!(
        listing.contains("a.mp3  ->  ") && listing.contains("a.txt"),
        "{listing}"
    );
    assert!(
        listing.contains("b.mp3  skipped (outputs exist)"),
        "{listing}"
    );
    assert!(err(&o).contains("1 to transcribe, 1 skipped"));
    assert!(fake.requests().is_empty());
    assert_eq!(files_in(&output), ["b.txt"]);
    let json = transcribe(
        &fake,
        &[
            s(&input),
            "--out",
            s(&output),
            "--format",
            "txt",
            "--dry-run",
            "--json",
        ],
    );
    let value: Value = serde_json::from_str(&out(&json)).unwrap();
    assert_eq!(value["to_do"], 1);
    assert_eq!(value["files"][1]["skipped"], true);
}

#[test]
fn limit_stops_after_n_processed_files_and_skipped_ones_do_not_count() {
    let dir = scratch("limit");
    let (input, output) = (dir.join("in"), dir.join("out"));
    for name in ["a.mp3", "b.mp3", "c.mp3", "d.mp3"] {
        touch(&input.join(name));
    }
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join("a.txt"), "done").unwrap();
    let fake = service(&[("stt", "whisper")]);
    let o = transcribe(
        &fake,
        &[
            s(&input),
            "--out",
            s(&output),
            "--format",
            "txt",
            "--limit",
            "2",
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(files_in(&output), ["a.txt", "b.txt", "c.txt"]);
    assert_eq!(fake.uploads().len(), 2);
    assert!(
        err(&o).contains("2 done, 1 skipped, 0 failed"),
        "{}",
        err(&o)
    );
}

#[test]
fn json_output_is_a_machine_readable_summary() {
    let dir = scratch("json");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    let fake = service(&[("stt", "whisper")]);
    let o = transcribe(
        &fake,
        &[s(&input), "--out", s(&output), "--format", "txt", "--json"],
    );
    let value: Value = serde_json::from_str(&out(&o)).unwrap();
    assert_eq!(
        (value["done"].as_u64(), value["failed"].as_u64()),
        (Some(1), Some(0))
    );
    assert_eq!(value["audio_seconds"], 3725.5);
    assert_eq!(value["files"][0]["status"], "done");
}

#[test]
fn ctrl_c_removes_the_part_file_prints_the_summary_and_exits_non_zero() {
    let dir = scratch("interrupt");
    let (input, output) = (dir.join("in"), dir.join("out"));
    touch(&input.join("a.mp3"));
    let fake = serve(
        Box::new(|r| {
            if r.path == "/api/v1/models" {
                return json_reply("200 OK", &models(&[("stt", "whisper")]));
            }
            std::thread::sleep(Duration::from_secs(30));
            reply("200 OK", "application/json", ANSWER)
        }),
        None,
    );
    let url = fake.url();
    let mut child = command(&["--host", &url, "transcribe", s(&input), "--out", s(&output)])
        .spawn()
        .unwrap();
    let begun = Instant::now();
    while fake.uploads().is_empty() {
        assert!(
            begun.elapsed() < Duration::from_secs(20),
            "the upload never arrived"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(300));
    let pid = child.id().to_string();
    assert!(
        Command::new("kill")
            .args(["-INT", &pid])
            .status()
            .unwrap()
            .success()
    );
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success());
    assert!(
        begun.elapsed() < Duration::from_secs(15),
        "did not stop at once"
    );
    assert!(
        stderr.contains("Interrupted") && stderr.contains("Summary: 0 done"),
        "{stderr}"
    );
    assert!(files_in(&output).iter().all(|n| !n.ends_with(".part")));
}

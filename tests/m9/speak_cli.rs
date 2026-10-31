//! `eliezer speak` and `eliezer voices` against a fake service on loopback: the audio goes to stdout, a regular
//! file or a named pipe while it arrives, a failed stream is a failed command, a closed reader ends quietly. No
//! network, no GPU.
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("eliezer-speak-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

// ---- the fake service ------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Seen {
    method: String,
    path: String,
    body: Vec<u8>,
}
impl Seen {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    /// The text of a multipart field.
    fn field(&self, name: &str) -> Option<String> {
        let text = String::from_utf8_lossy(&self.body).into_owned();
        let marker = format!("name=\"{name}\"\r\n\r\n");
        let start = text.find(&marker)? + marker.len();
        Some(text[start..text[start..].find("\r\n")? + start].to_owned())
    }
}
/// Answers one request; may write a streamed answer itself.
type Handler = Box<dyn Fn(&Seen, &mut TcpStream) + Send + Sync>;
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
    fn speeches(&self) -> Vec<Seen> {
        self.requests()
            .into_iter()
            .filter(|r| r.path == "/v1/audio/speech")
            .collect()
    }
}
fn reply(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {status}\r\nServer: eliezer/test\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    );
}
fn json_reply(stream: &mut TcpStream, status: &str, body: &Value) {
    reply(stream, status, "application/json", &body.to_string());
}
fn stream_head(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nServer: eliezer/test\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )
}
fn chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    stream.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}
fn finish(stream: &mut TcpStream) {
    let _ = stream.write_all(b"0\r\n\r\n");
}
/// Every connection is served on a thread of its own, so a long stream does not block the model list.
fn serve(handler: Handler) -> Fake {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    let handler = Arc::new(handler);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let (log, handler) = (log.clone(), handler.clone());
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let mut parts = line.split_whitespace();
                let (method, path) = (
                    parts.next().unwrap_or("").to_owned(),
                    parts.next().unwrap_or("").to_owned(),
                );
                let mut length = 0_usize;
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let request = Seen { method, path, body };
                log.lock().unwrap().push(request.clone());
                handler(&request, &mut stream);
            });
        }
    });
    Fake { port, seen }
}

fn models(types: &[(&str, &str)]) -> Value {
    json!({"models": types.iter().map(|(t, k)| json!({"type": t, "key": k})).collect::<Vec<_>>()})
}
/// An open-length WAV header (what the service streams), 8 kHz mono 16 bit.
fn open_header() -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16_u32.to_le_bytes());
    h.extend_from_slice(&1_u16.to_le_bytes());
    h.extend_from_slice(&1_u16.to_le_bytes());
    h.extend_from_slice(&8000_u32.to_le_bytes());
    h.extend_from_slice(&16000_u32.to_le_bytes());
    h.extend_from_slice(&2_u16.to_le_bytes());
    h.extend_from_slice(&16_u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h
}
fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|n| ((n + seed) % 251) as u8).collect()
}
/// The pieces of a stream: the header with the first audio, then two more pieces.
fn pieces() -> Vec<Vec<u8>> {
    let mut first = open_header();
    first.extend_from_slice(&pattern(1000, 0));
    vec![first, pattern(2000, 7), pattern(500, 13)]
}

/// A service with the given models that streams `pieces` for every speech request.
fn service(types: &[(&str, &str)], pieces: Vec<Vec<u8>>, complete: bool) -> Fake {
    let list = models(types);
    serve(Box::new(move |r, stream| match r.path.as_str() {
        "/api/v1/models" => json_reply(stream, "200 OK", &list),
        "/v1/audio/speech" => {
            let _ = stream_head(stream);
            for piece in &pieces {
                if chunk(stream, piece).is_err() {
                    return;
                }
            }
            if complete {
                finish(stream);
            }
            // otherwise the connection just ends: a failed stream
        }
        "/v1/audio/voices" if r.method == "GET" => {
            json_reply(stream, "200 OK", &json!({"voices": ["mama", "papa"]}))
        }
        "/v1/audio/voices" => json_reply(
            stream,
            "201 Created",
            &json!({"name": r.field("name"), "replaced": r.field("replace").as_deref() == Some("true")}),
        ),
        p if p.starts_with("/v1/audio/voices/") && r.method == "DELETE" => json_reply(
            stream,
            "200 OK",
            &json!({"name": p.rsplit('/').next(), "deleted": true}),
        ),
        _ => json_reply(
            stream,
            "404 Not Found",
            &json!({"error": {"message": "no such path"}}),
        ),
    }))
}
fn tts() -> Fake {
    service(&[("llm", "chat"), ("tts", "reader")], pieces(), true)
}
fn all_bytes(pieces: &[Vec<u8>]) -> Vec<u8> {
    pieces.concat()
}

fn command(args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eliezer"));
    command
        .env_remove("ELIEZER_HOST")
        .env_remove("ELIEZER_API_KEY")
        .env_remove("ELIEZER_API_KEY_FILE")
        // a hung test ends with an error instead of hanging
        .env("ELIEZER_TIMEOUT_SECS", "30")
        .env("XDG_CONFIG_HOME", "/nonexistent/config")
        .env("ELIEZER_CONFIG", "/nonexistent/eliezer.toml")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}
fn with_host(fake: &Fake, args: &[&str]) -> Command {
    let url = fake.url();
    let mut all = vec!["--host", url.as_str()];
    all.extend_from_slice(args);
    command(&all)
}
fn speak(fake: &Fake, args: &[&str]) -> Output {
    let mut all = vec!["speak"];
    all.extend_from_slice(args);
    with_host(fake, &all).output().unwrap()
}
fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}
fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}
/// Waits for a flag that another thread sets; bounded, so a failure is an error and not a hang.
fn until(flag: &AtomicBool) {
    let start = Instant::now();
    while !flag.load(Ordering::SeqCst) {
        assert!(start.elapsed() < Duration::from_secs(20), "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

// ---- standard output -------------------------------------------------------------------------

#[test]
fn the_audio_goes_to_stdout_and_nothing_else_does() {
    let fake = tts();
    let o = speak(&fake, &["--text", "Hallo Welt.", "--out", "-"]);
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(o.stdout, all_bytes(&pieces()));
    assert_eq!(err(&o), "");
    // the only tts model was chosen, and the request asks for a stream
    let requests = fake.requests();
    assert_eq!(requests[0].path, "/api/v1/models");
    let body = fake.speeches()[0].json();
    assert_eq!(
        body,
        json!({"model": "reader", "input": "Hallo Welt.", "response_format": "wav", "stream": true})
    );
}

#[test]
fn the_options_reach_the_request() {
    let fake = tts();
    let o = speak(
        &fake,
        &[
            "--model",
            "other",
            "--voice",
            "mama",
            "--language",
            "de",
            "--speed",
            "1.5",
            "--seed",
            "0",
            "--format",
            "pcm",
            "--text",
            "Hallo.",
            "--out",
            "-",
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    // a model that is named needs no model list
    assert!(fake.requests().iter().all(|r| r.path != "/api/v1/models"));
    assert_eq!(
        fake.speeches()[0].json(),
        json!({"model": "other", "input": "Hallo.", "response_format": "pcm", "stream": true,
               "voice": "mama", "language": "de", "speed": 1.5, "seed": 0})
    );
}

#[test]
fn the_text_comes_from_a_file_or_from_stdin() {
    let dir = scratch("text");
    let file = dir.join("kapitel.txt");
    fs::write(&file, "Aus der Datei.\nZweite Zeile.\n").unwrap();
    let fake = tts();
    let o = speak(&fake, &["--file", file.to_str().unwrap(), "--out", "-"]);
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(
        fake.speeches()[0].json()["input"],
        "Aus der Datei.\nZweite Zeile.\n"
    );

    let mut child = with_host(&fake, &["speak", "--out", "-"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all("Von stdin.".as_bytes())
        .unwrap();
    let o = child.wait_with_output().unwrap();
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(fake.speeches()[1].json()["input"], "Von stdin.");
    assert_eq!(o.stdout, all_bytes(&pieces()));
}

#[test]
fn the_first_audio_is_written_before_the_stream_has_ended() {
    // the fake sends the first piece and then waits until the test has seen it on stdout
    let (release, wait) = mpsc::channel::<()>();
    let wait = Mutex::new(wait);
    let first = pieces().remove(0);
    let sent = first.clone();
    let fake = serve(Box::new(move |r, stream| match r.path.as_str() {
        "/api/v1/models" => json_reply(stream, "200 OK", &models(&[("tts", "reader")])),
        _ => {
            let _ = stream_head(stream);
            let _ = chunk(stream, &sent);
            let _ = wait.lock().unwrap().recv();
            let _ = chunk(stream, b"tail");
            finish(stream);
        }
    }));
    let mut child = with_host(&fake, &["speak", "--text", "Hallo.", "--out", "-"])
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut got = vec![0; first.len()];
    // this read returns only if the command passes the piece on before the end of the stream
    stdout.read_exact(&mut got).unwrap();
    assert_eq!(got, first);
    release.send(()).unwrap();
    let mut rest = Vec::new();
    stdout.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, b"tail");
    assert!(child.wait().unwrap().success());
}

#[test]
fn a_reader_that_quits_ends_the_command_quietly_and_closes_the_request() {
    let closed = Arc::new(AtomicBool::new(false));
    let flag = closed.clone();
    let fake = serve(Box::new(move |r, stream| match r.path.as_str() {
        "/api/v1/models" => json_reply(stream, "200 OK", &models(&[("tts", "reader")])),
        _ => {
            let _ = stream_head(stream);
            // far more than the buffers of the sockets and the pipe hold; the writes fail once the client has gone
            let block = pattern(64 * 1024, 3);
            for _ in 0..4000 {
                if chunk(stream, &block).is_err() {
                    flag.store(true, Ordering::SeqCst);
                    return;
                }
            }
        }
    }));
    let mut child: Child = with_host(&fake, &["speak", "--text", "Hallo."])
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    // like `head -c 10`
    let mut head = [0_u8; 10];
    stdout.read_exact(&mut head).unwrap();
    drop(stdout);
    let o = child.wait_with_output().unwrap();
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(err(&o), "");
    until(&closed);
}

// ---- files and named pipes -------------------------------------------------------------------

fn wav_sizes(file: &[u8]) -> (u32, u32) {
    (
        u32::from_le_bytes(file[4..8].try_into().unwrap()),
        u32::from_le_bytes(file[40..44].try_into().unwrap()),
    )
}

#[test]
fn a_regular_file_is_written_atomically_and_gets_the_real_wav_sizes() {
    let dir = scratch("file");
    let target = dir.join("sub/kapitel.wav");
    let fake = tts();
    let o = speak(
        &fake,
        &["--text", "Hallo.", "--out", target.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert!(o.stdout.is_empty());
    let file = fs::read(&target).unwrap();
    let streamed = all_bytes(&pieces());
    assert_eq!(file.len(), streamed.len());
    // the audio is as streamed, the two open sizes are now the real ones
    assert_eq!(&file[44..], &streamed[44..]);
    assert_eq!(&file[..4], b"RIFF");
    assert_eq!(
        wav_sizes(&file),
        (file.len() as u32 - 8, file.len() as u32 - 44)
    );
    // nothing left over
    assert!(!dir.join("sub/kapitel.wav.part").exists());
    assert!(err(&o).contains("kapitel.wav"));
}

#[test]
fn raw_samples_in_a_file_are_not_touched() {
    let dir = scratch("pcm");
    let target = dir.join("a.pcm");
    let fake = service(&[("tts", "reader")], vec![pattern(3000, 1)], true);
    let o = speak(
        &fake,
        &[
            "--text",
            "Hallo.",
            "--format",
            "pcm",
            "--out",
            target.to_str().unwrap(),
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(fs::read(&target).unwrap(), pattern(3000, 1));
}

#[test]
fn an_existing_file_is_kept_unless_replace_is_given() {
    let dir = scratch("replace");
    let target = dir.join("a.wav");
    fs::write(&target, b"old").unwrap();
    let fake = tts();
    let o = speak(
        &fake,
        &["--text", "Hallo.", "--out", target.to_str().unwrap()],
    );
    assert_eq!(code(&o), 2);
    assert!(err(&o).contains("--replace"), "{}", err(&o));
    assert_eq!(fs::read(&target).unwrap(), b"old");
    // refused before anything was sent
    assert!(fake.requests().is_empty());
    let o = speak(
        &fake,
        &[
            "--text",
            "Hallo.",
            "--replace",
            "--out",
            target.to_str().unwrap(),
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(fs::read(&target).unwrap().len(), all_bytes(&pieces()).len());
}

#[test]
fn a_failed_stream_is_a_failed_command_and_leaves_no_file() {
    let dir = scratch("cut");
    let target = dir.join("a.wav");
    // the service ends the connection without the closing chunk
    let fake = service(&[("tts", "reader")], pieces(), false);
    let o = speak(
        &fake,
        &["--text", "Hallo.", "--out", target.to_str().unwrap()],
    );
    assert_eq!(code(&o), 1, "{}", err(&o));
    assert!(err(&o).contains("interrupted"), "{}", err(&o));
    assert!(!target.exists());
    assert!(!dir.join("a.wav.part").exists());
    // on stdout the exit code is the only thing that tells
    let o = speak(&fake, &["--text", "Hallo.", "--out", "-"]);
    assert_eq!(code(&o), 1);
    assert_eq!(o.stdout, all_bytes(&pieces()));
}

/// Makes a named pipe; `None` (with a message) when the system has no `mkfifo`.
fn fifo(path: &Path) -> Option<()> {
    match Command::new("mkfifo").arg(path).status() {
        Ok(status) if status.success() => Some(()),
        _ => {
            eprintln!("SKIPPED: mkfifo is not available, the named pipe test cannot run");
            None
        }
    }
}

#[test]
fn a_named_pipe_is_written_into_as_it_is() {
    let dir = scratch("fifo");
    let pipe = dir.join("speech.fifo");
    if fifo(&pipe).is_none() {
        return;
    }
    let fake = tts();
    let url = fake.url();
    let child = command(&[
        "--host",
        &url,
        "speak",
        "--text",
        "Hallo.",
        "--out",
        pipe.to_str().unwrap(),
    ])
    .spawn()
    .unwrap();
    // the command opens the pipe for writing; this opens it for reading
    let mut audio = Vec::new();
    fs::File::open(&pipe)
        .unwrap()
        .read_to_end(&mut audio)
        .unwrap();
    let o = child.wait_with_output().unwrap();
    assert_eq!(code(&o), 0, "{}", err(&o));
    // the open sizes stay open (a pipe cannot be patched), and there is no .part file or rename
    assert_eq!(audio, all_bytes(&pieces()));
    assert!(fs::metadata(&pipe).unwrap().file_type().is_fifo_like());
    assert!(!dir.join("speech.fifo.part").exists());
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    // --replace is not needed for a pipe either
}

#[test]
fn a_named_pipe_whose_reader_quits_ends_quietly() {
    let dir = scratch("fifo-quit");
    let pipe = dir.join("speech.fifo");
    if fifo(&pipe).is_none() {
        return;
    }
    let closed = Arc::new(AtomicBool::new(false));
    let flag = closed.clone();
    let fake = serve(Box::new(move |r, stream| match r.path.as_str() {
        "/api/v1/models" => json_reply(stream, "200 OK", &models(&[("tts", "reader")])),
        _ => {
            let _ = stream_head(stream);
            let block = pattern(64 * 1024, 5);
            for _ in 0..4000 {
                if chunk(stream, &block).is_err() {
                    flag.store(true, Ordering::SeqCst);
                    return;
                }
            }
        }
    }));
    let url = fake.url();
    let child = command(&[
        "--host",
        &url,
        "speak",
        "--text",
        "Hallo.",
        "--out",
        pipe.to_str().unwrap(),
    ])
    .spawn()
    .unwrap();
    let mut reader = fs::File::open(&pipe).unwrap();
    let mut head = [0_u8; 10];
    reader.read_exact(&mut head).unwrap();
    drop(reader);
    let o = child.wait_with_output().unwrap();
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(err(&o), "");
    until(&closed);
}

/// `FileType` has `is_fifo` only through an extension trait; the test keeps its use in one place.
trait FifoLike {
    fn is_fifo_like(&self) -> bool;
}
impl FifoLike for std::fs::FileType {
    fn is_fifo_like(&self) -> bool {
        use std::os::unix::fs::FileTypeExt;
        self.is_fifo()
    }
}

// ---- exit codes ------------------------------------------------------------------------------

#[test]
fn the_model_is_the_only_tts_model_or_named() {
    let none = service(&[("llm", "chat")], pieces(), true);
    let o = speak(&none, &["--text", "Hallo.", "--out", "-"]);
    assert_eq!(code(&o), 2);
    assert!(err(&o).contains("no speech synthesis"), "{}", err(&o));
    assert!(none.speeches().is_empty());

    let several = service(&[("tts", "a"), ("tts", "b"), ("stt", "c")], pieces(), true);
    let o = speak(&several, &["--text", "Hallo.", "--out", "-"]);
    assert_eq!(code(&o), 2);
    let message = err(&o);
    assert!(
        message.contains("a, b") && message.contains("--model"),
        "{message}"
    );
    assert!(several.speeches().is_empty());
    let o = speak(
        &several,
        &["--model", "b", "--text", "Hallo.", "--out", "-"],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(several.speeches()[0].json()["model"], "b");
}

#[test]
fn wrong_usage_is_exit_2_and_sends_nothing() {
    let fake = tts();
    for args in [
        vec!["--out", "-"],
        vec!["--text", "  \n", "--out", "-"],
        vec!["--text", "a", "--file", "b", "--out", "-"],
        vec!["--text", "a", "--format", "mp3", "--out", "-"],
        vec!["--text", "a", "--speed", "9", "--out", "-"],
        vec!["--text", "a", "--seed", "x", "--out", "-"],
        vec!["--text", "a", "--language", "German", "--out", "-"],
        vec!["--file", "/nonexistent/text.txt", "--out", "-"],
        vec!["--text", "a", "--out", "/tmp"],
        vec!["--text", "a", "--bogus"],
    ] {
        let o = speak(&fake, &args);
        assert_eq!(code(&o), 2, "{args:?}: {}", err(&o));
        assert!(o.stdout.is_empty(), "{args:?}");
    }
    assert!(fake.speeches().is_empty());
}

#[test]
fn an_error_of_the_service_is_exit_1_with_its_message() {
    let fake = serve(Box::new(|r, stream| match r.path.as_str() {
        "/api/v1/models" => json_reply(stream, "200 OK", &models(&[("tts", "reader")])),
        _ => json_reply(
            stream,
            "413 Payload Too Large",
            &json!({"error": {"type": "payload_too_large", "message": "input is longer than the limit of the model"}}),
        ),
    }));
    let dir = scratch("413");
    let target = dir.join("a.wav");
    let o = speak(
        &fake,
        &["--text", "Hallo.", "--out", target.to_str().unwrap()],
    );
    assert_eq!(code(&o), 1);
    assert!(err(&o).contains("longer than the limit"), "{}", err(&o));
    assert!(err(&o).contains("413"));
    // nothing was created for an answer that never came
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    let o = speak(&fake, &["--text", "Hallo.", "--out", "-"]);
    assert_eq!(code(&o), 1);
    assert!(o.stdout.is_empty());
}

#[test]
fn a_service_that_is_not_there_is_exit_3() {
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let url = format!("http://127.0.0.1:{port}");
    let o = command(&["--host", &url, "speak", "--text", "Hallo.", "--out", "-"])
        .output()
        .unwrap();
    assert_eq!(code(&o), 3, "{}", err(&o));
    assert!(o.stdout.is_empty());
}

#[test]
fn a_dash_as_the_file_reads_stdin() {
    let fake = tts();
    let mut child = with_host(&fake, &["speak", "--file", "-", "--out", "-"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all("Hallo Jörg".as_bytes())
        .unwrap();
    let o = child.wait_with_output().unwrap();
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(fake.speeches()[0].json()["input"], "Hallo Jörg");
}

// ---- voices ----------------------------------------------------------------------------------

fn voices(fake: &Fake, args: &[&str]) -> Output {
    let mut all = vec!["voices"];
    all.extend_from_slice(args);
    with_host(fake, &all).output().unwrap()
}

#[test]
fn voices_lists_the_names() {
    let fake = tts();
    let o = voices(&fake, &[]);
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(String::from_utf8_lossy(&o.stdout), "mama\npapa\n");
    let o = with_host(&fake, &["--json", "voices"]).output().unwrap();
    let parsed: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(parsed, json!({"voices": ["mama", "papa"]}));
}

#[test]
fn list_voices_is_an_alias_of_voices() {
    let fake = tts();
    let o = with_host(&fake, &["--list-voices"]).output().unwrap();
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert_eq!(String::from_utf8_lossy(&o.stdout), "mama\npapa\n");
}

#[test]
fn voices_add_takes_the_name_with_name() {
    let dir = scratch("add-name");
    let file = dir.join("mama.wav");
    fs::write(&file, b"RIFFdata").unwrap();
    let fake = tts();
    let o = voices(&fake, &["add", "--name", "mama", file.to_str().unwrap()]);
    assert_eq!(code(&o), 0, "{}", err(&o));
    let sent = fake
        .requests()
        .into_iter()
        .find(|r| r.method == "POST")
        .unwrap();
    assert_eq!(sent.field("name").as_deref(), Some("mama"));
    // Neither a name nor a file is a usage error.
    let o = voices(&fake, &["add", "--name", "mama"]);
    assert_eq!(code(&o), 2, "{}", err(&o));
}

#[test]
fn voices_add_takes_the_recording_and_the_text_with_file() {
    let dir = scratch("add-files");
    let audio = dir.join("mama.mp3");
    let text = dir.join("mama.txt");
    fs::write(&audio, b"RIFFdata").unwrap();
    fs::write(&text, "Die Worte der Aufnahme.").unwrap();
    let (a, t) = (audio.to_str().unwrap(), text.to_str().unwrap());
    let fake = tts();
    // in either order
    let o = voices(&fake, &["add", "--name", "mama", "--file", t, "--file", a]);
    assert_eq!(code(&o), 0, "{}", err(&o));
    let sent = fake
        .requests()
        .into_iter()
        .find(|r| r.method == "POST")
        .unwrap();
    assert_eq!(sent.field("name").as_deref(), Some("mama"));
    assert_eq!(
        sent.field("transcript").as_deref(),
        Some("Die Worte der Aufnahme.")
    );
    // two recordings, or a text file together with --transcript, are usage errors
    let o = voices(&fake, &["add", "--name", "x", "--file", a, "--file", a]);
    assert_eq!(code(&o), 2, "{}", err(&o));
    let o = voices(
        &fake,
        &[
            "add",
            "--name",
            "x",
            "--file",
            a,
            "--file",
            t,
            "--transcript",
            "Hi",
        ],
    );
    assert_eq!(code(&o), 2, "{}", err(&o));
    // speak takes one --file only
    let o = speak(&fake, &["--file", t, "--file", t, "--out", "-"]);
    assert_eq!(code(&o), 2, "{}", err(&o));
}

/// The WAV file inside an upload.
fn uploaded_wav(body: &[u8]) -> &[u8] {
    let at = body.windows(4).position(|w| w == b"RIFF").unwrap();
    let end = body
        .windows(2)
        .rposition(|w| w == b"\r\n")
        .unwrap_or(body.len());
    let tail = &body[at..];
    // the multipart closing boundary follows the file
    let boundary_at = tail.windows(4).rposition(|w| w == b"\r\n--").unwrap_or(end);
    &tail[..boundary_at]
}

#[test]
fn voices_add_cuts_a_part_with_start_and_length() {
    let dir = scratch("add-cut");
    let file = dir.join("long.wav");
    // ten seconds at 16 kHz; the value of a sample is its second, so the part can be recognized
    let samples: Vec<i16> = (0..160_000_u32)
        .map(|i| (i / 16_000) as i16 * 1000)
        .collect();
    fs::write(&file, eliezer_speech::wav_at(&samples, 16_000)).unwrap();
    let f = file.to_str().unwrap();
    let fake = tts();
    let o = voices(
        &fake,
        &[
            "add", "--name", "x", "--file", f, "--start", "0:02", "--length", "3",
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    let sent = fake
        .requests()
        .into_iter()
        .find(|r| r.method == "POST")
        .unwrap();
    let wav = uploaded_wav(&sent.body);
    // three seconds at 24 kHz, 16 bit, behind a 44 byte header
    assert_eq!(wav.len(), 44 + 3 * 24_000 * 2);
    let sample = |n: usize| i16::from_le_bytes([wav[44 + 2 * n], wav[45 + 2 * n]]);
    assert!(
        (1500..=2500).contains(&sample(12_000)),
        "{}",
        sample(12_000)
    );
    assert!(
        (3500..=4500).contains(&sample(60_000)),
        "{}",
        sample(60_000)
    );
    // a start behind the end, a bad time and a length of 0 are usage errors
    for args in [["--start", "20"], ["--start", "x"], ["--length", "0"]] {
        let mut all = vec!["add", "--name", "x", "--file", f];
        all.extend_from_slice(&args);
        let o = voices(&fake, &all);
        assert_eq!(code(&o), 2, "{args:?}: {}", err(&o));
    }
}

#[test]
fn an_upload_refused_early_shows_the_answer_of_the_service() {
    // The service answers 413 and closes before the body has been read; the command shows the answer, not a
    // broken pipe.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut head = [0_u8; 1024];
        let _ = stream.read(&mut head);
        let body =
            r#"{"error":{"message":"The file is larger than the limit","type":"invalid_request"}}"#;
        let _ = write!(
            stream,
            "HTTP/1.1 413 Payload Too Large\r\nServer: eliezer/test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        // Like the service: read the rest of the body, so that the answer is not lost to a reset.
        let _ = stream.shutdown(std::net::Shutdown::Write);
        let _ = std::io::copy(&mut stream, &mut std::io::sink());
    });
    let dir = scratch("early-413");
    let file = dir.join("big.bin");
    fs::write(&file, vec![0_u8; 32 << 20]).unwrap();
    let o = command(&["--host", &url, "voices", "add", "x", file.to_str().unwrap()])
        .output()
        .unwrap();
    server.join().unwrap();
    assert_eq!(code(&o), 1, "{}", err(&o));
    assert!(err(&o).contains("larger than the limit"), "{}", err(&o));
}

#[test]
fn voices_add_uploads_the_recording() {
    let dir = scratch("add");
    let file = dir.join("mama.wav");
    fs::write(&file, b"RIFFdata-of-the-recording").unwrap();
    let fake = tts();
    let o = voices(&fake, &["add", "mama", file.to_str().unwrap()]);
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert!(o.stdout.is_empty());
    let sent = fake
        .requests()
        .into_iter()
        .find(|r| r.method == "POST")
        .unwrap();
    assert_eq!(sent.path, "/v1/audio/voices");
    assert_eq!(sent.field("name").as_deref(), Some("mama"));
    assert_eq!(sent.field("replace").as_deref(), Some("false"));
    assert!(
        String::from_utf8_lossy(&sent.body).contains("RIFFdata-of-the-recording"),
        "the recording is part of the upload"
    );
    let o = voices(
        &fake,
        &[
            "add",
            "mama",
            file.to_str().unwrap(),
            "--replace",
            "--transcript",
            "Es war einmal.",
        ],
    );
    assert_eq!(code(&o), 0, "{}", err(&o));
    assert!(err(&o).contains("replaced"), "{}", err(&o));
    let sent = fake
        .requests()
        .into_iter()
        .filter(|r| r.method == "POST")
        .nth(1)
        .unwrap();
    assert_eq!(sent.field("replace").as_deref(), Some("true"));
    assert_eq!(sent.field("transcript").as_deref(), Some("Es war einmal."));
}

#[test]
fn voices_rm_deletes_by_name() {
    let fake = tts();
    let o = voices(&fake, &["rm", "mama"]);
    assert_eq!(code(&o), 0, "{}", err(&o));
    let sent = fake.requests().pop().unwrap();
    assert_eq!(
        (sent.method.as_str(), sent.path.as_str()),
        ("DELETE", "/v1/audio/voices/mama")
    );
}

#[test]
fn voices_usage_errors_and_service_errors() {
    let fake = serve(Box::new(|r, stream| {
        if r.method == "POST" {
            json_reply(
                stream,
                "409 Conflict",
                &json!({"error": {"message": "A voice with this name exists; send replace=true to overwrite it"}}),
            );
        } else {
            json_reply(stream, "200 OK", &json!({"voices": []}));
        }
    }));
    let dir = scratch("voices-usage");
    let file = dir.join("a.wav");
    fs::write(&file, b"x").unwrap();
    for args in [
        vec!["add", "mama"],
        vec!["add", "../x", file.to_str().unwrap()],
        vec!["add", "mama", "/nonexistent/a.wav"],
        vec!["rm"],
        vec!["rm", "a/b"],
        vec!["rm", ".."],
        vec!["frobnicate"],
    ] {
        let o = voices(&fake, &args);
        assert_eq!(code(&o), 2, "{args:?}: {}", err(&o));
    }
    assert!(fake.requests().is_empty());
    let o = voices(&fake, &["add", "mama", file.to_str().unwrap()]);
    assert_eq!(code(&o), 1);
    assert!(err(&o).contains("replace=true"), "{}", err(&o));
    // an empty library says so on stderr and prints nothing
    let o = voices(&fake, &[]);
    assert_eq!(code(&o), 0);
    assert!(o.stdout.is_empty());
    assert!(err(&o).contains("no voices"), "{}", err(&o));
}

#[test]
fn a_terminal_gets_no_audio_unless_out_dash_is_given() {
    // `script` gives the command a terminal as stdout; without it the test cannot run
    if Command::new("script").arg("--version").output().is_err() {
        eprintln!("SKIPPED: script is not available, the terminal test cannot run");
        return;
    }
    let fake = tts();
    let url = fake.url();
    let o = Command::new("script")
        .args(["-qec"])
        .arg(format!(
            "{} --host {url} speak --text Hallo.; echo rc=$?",
            env!("CARGO_BIN_EXE_eliezer")
        ))
        .arg("/dev/null")
        .env("ELIEZER_CONFIG", "/nonexistent/eliezer.toml")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&o.stdout);
    assert!(text.contains("terminal") && text.contains("rc=2"), "{text}");
    assert!(fake.requests().is_empty());
}

#[test]
fn a_rejected_request_shows_the_message_of_the_service_and_exits_with_1() {
    let message = "The model \"reader\" needs a voice: pass \"voice\" (see GET /v1/audio/voices) or set extra.default_voice in the model's configuration.";
    let list = models(&[("tts", "reader")]);
    let fake = serve(Box::new(move |r, stream| match r.path.as_str() {
        "/api/v1/models" => json_reply(stream, "200 OK", &list),
        _ => json_reply(
            stream,
            "400 Bad Request",
            &json!({"error": {"type": "invalid_request", "message": message}}),
        ),
    }));
    let o = speak(&fake, &["--text", "Hallo.", "--out", "-"]);
    assert_eq!(code(&o), 1, "{}", err(&o));
    assert!(o.stdout.is_empty());
    assert!(err(&o).contains(message), "{}", err(&o));
    assert!(err(&o).contains("HTTP 400"), "{}", err(&o));
    // the help tells that a model may need a voice
    let help = String::from_utf8_lossy(&command(&["help"]).output().unwrap().stdout).into_owned();
    assert!(help.contains("needs one"), "{help}");
}

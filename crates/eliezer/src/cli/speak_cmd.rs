//! `eliezer speak`: reads a text aloud through `POST /v1/audio/speech` with `stream = true` and copies the audio
//! to standard output, a regular file or a named pipe while it arrives, so that `sox`, `ffmpeg` or `play` can take it
//! over. The service writes `wav` (with an open length) and `pcm`; `mp3` is made here from the WAV stream, block by
//! block, by [`super::mp3`].
use super::{Client, Failure, Parsed, Res, SHORT, api_error, call, mp3, transport, usage_err};
use serde_json::{Map, Value, json};
use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{self, IsTerminal, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

/// How long to wait for the next piece of audio. The first piece comes after the model is loaded and the first
/// passage is read, which can take minutes on a busy GPU.
const WAIT: Duration = Duration::from_secs(3600);
/// The size of the blocks that are copied.
const BLOCK: usize = 64 * 1024;
/// A WAV header written by the service for a stream: 44 bytes with both sizes left open.
const WAV_HEADER: usize = 44;

/// Where the audio goes.
enum Sink {
    /// Standard output.
    Stdout,
    /// A named pipe or another file that is no regular file: written block by block, nothing renamed.
    Stream(PathBuf),
    /// A regular file: written as `<path>.part` and renamed when the stream is complete.
    File { path: PathBuf, replace: bool },
}

/// An open sink.
enum Open {
    Stdout(io::Stdout),
    Stream(File),
    File {
        file: File,
        part: PathBuf,
        path: PathBuf,
    },
}
impl Open {
    fn write_block(&mut self, block: &[u8]) -> io::Result<()> {
        // Every block is flushed: a pipe must see the audio when it is there, not when a buffer is full.
        match self {
            Self::Stdout(out) => {
                let mut out = out.lock();
                out.write_all(block)?;
                out.flush()
            }
            Self::Stream(file) | Self::File { file, .. } => file.write_all(block),
        }
    }
}

/// The `.part` file being written; removed when the run is interrupted.
type Pending = Arc<Mutex<Option<PathBuf>>>;

fn part_name(path: &Path) -> PathBuf {
    let mut name: OsString = path.file_name().unwrap_or_default().to_owned();
    name.push(".part");
    path.with_file_name(name)
}

/// Decides where the audio goes, before anything is sent.
fn sink(parsed: &Parsed) -> Res<Sink> {
    let replace = parsed.flag("--replace");
    match parsed.value("--out") {
        None | Some("-") => {
            // Raw audio on a terminal is noise; `--out -` says that it is meant.
            if parsed.value("--out").is_none() && io::stdout().is_terminal() {
                return usage_err(
                    "Standard output is a terminal; the audio would not be readable. Pipe it, for example\n\
                     `eliezer speak --file text.txt --format mp3 > text.mp3` or `| play -t wav -`,\n\
                     or name a file with --out PATH (or --out - to write to the terminal anyway)",
                );
            }
            Ok(Sink::Stdout)
        }
        Some(path) => {
            let path = PathBuf::from(path);
            // `metadata` follows links, so a link to a named pipe is a pipe.
            match fs::metadata(&path) {
                Ok(meta) if meta.is_dir() => usage_err(format!(
                    "{} is a directory; name a file with --out",
                    path.display()
                )),
                Ok(meta) if meta.is_file() => {
                    if replace {
                        Ok(Sink::File { path, replace })
                    } else {
                        usage_err(format!(
                            "{} exists; use --replace to overwrite it",
                            path.display()
                        ))
                    }
                }
                // Not a regular file and not a directory: a named pipe (FIFO), a device or a socket. It is written
                // into as it is, block by block, without `.part` and rename.
                Ok(_) => Ok(Sink::Stream(path)),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Sink::File { path, replace }),
                Err(e) => usage_err(format!("Cannot use {}: {e}", path.display())),
            }
        }
    }
}

/// Opens the sink once the service has accepted the request. A named pipe blocks here until a reader opens it.
fn open(sink: Sink, pending: &Pending) -> Res<Open> {
    let fail = |path: &Path, e: io::Error| {
        Failure::Request(format!("Cannot write {}: {e}", path.display()))
    };
    match sink {
        Sink::Stdout => Ok(Open::Stdout(io::stdout())),
        Sink::Stream(path) => OpenOptions::new()
            .write(true)
            .open(&path)
            .map(Open::Stream)
            .map_err(|e| fail(&path, e)),
        Sink::File { path, replace } => {
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                fs::create_dir_all(dir).map_err(|e| fail(&path, e))?;
            }
            // Between the first check and now another program can have made the file.
            if !replace && fs::symlink_metadata(&path).is_ok() {
                return usage_err(format!(
                    "{} exists; use --replace to overwrite it",
                    path.display()
                ));
            }
            let part = part_name(&path);
            let file = OpenOptions::new()
                .write(true)
                .read(true)
                .create(true)
                .truncate(true)
                .open(&part)
                .map_err(|e| fail(&part, e))?;
            *pending.lock().unwrap_or_else(PoisonError::into_inner) = Some(part.clone());
            Ok(Open::File { file, part, path })
        }
    }
}

/// A streamed WAV has its sizes left open; a file on disk gets the real ones, so that every player takes it.
fn close_wav(file: &mut File, total: u64) -> io::Result<()> {
    if total < WAV_HEADER as u64 {
        return Ok(());
    }
    let mut head = [0_u8; WAV_HEADER];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut head)?;
    let open = |at: usize| head[at..at + 4] == [0xFF; 4];
    if &head[..4] != b"RIFF" || !open(4) || &head[36..40] != b"data" || !open(40) {
        return Ok(());
    }
    let riff = u32::try_from(total - 8).unwrap_or(u32::MAX);
    let data = u32::try_from(total - WAV_HEADER as u64).unwrap_or(u32::MAX);
    file.seek(SeekFrom::Start(4))?;
    file.write_all(&riff.to_le_bytes())?;
    file.seek(SeekFrom::Start(40))?;
    file.write_all(&data.to_le_bytes())?;
    file.flush()
}

fn read_stdin() -> Res<String> {
    if io::stdin().is_terminal() {
        eprintln!("Reading the text from the terminal; end it with Ctrl-D.");
    }
    let mut text = String::new();
    io::stdin()
        .read_to_string(&mut text)
        .map_err(|e| Failure::Usage(format!("Cannot read the text from stdin: {e}")))?;
    Ok(text)
}

fn read_text(parsed: &Parsed) -> Res<String> {
    if parsed.all_values("--file").len() > 1 {
        return usage_err("--file may only be given once");
    }
    let text = match (parsed.value("--text"), parsed.value("--file")) {
        (Some(_), Some(_)) => return usage_err("Give --text or --file, not both"),
        (Some(text), None) => text.to_owned(),
        // `--file -` reads stdin, as usual for command line tools.
        (None, Some("-")) | (None, None) => read_stdin()?,
        (None, Some(file)) => fs::read_to_string(file)
            .map_err(|e| Failure::Usage(format!("Cannot read the text file {file}: {e}")))?,
    };
    if text.trim().is_empty() {
        return usage_err("There is no text to read; give --text, --file or text on stdin");
    }
    Ok(text)
}

/// The model to use: the one named, else the only speech synthesis model of the service.
fn resolve_model(client: &Client, named: Option<&str>) -> Res<String> {
    if let Some(model) = named {
        return Ok(model.to_owned());
    }
    let models = call(client, "GET", "/api/v1/models", None, SHORT)?;
    let keys: Vec<&str> = models["models"]
        .as_array()
        .map(|list| {
            list.iter()
                .filter(|m| m["type"] == "tts")
                .filter_map(|m| m["key"].as_str())
                .collect()
        })
        .unwrap_or_default();
    match keys.as_slice() {
        [only] => Ok((*only).to_owned()),
        [] => usage_err("The service has no speech synthesis (tts) model; configure one first"),
        many => usage_err(format!(
            "The service has several speech synthesis models ({}); name one with --model",
            many.join(", ")
        )),
    }
}

/// The request body from the options. The values are checked by the service as well; the checks here give a
/// message before anything is sent.
fn request(parsed: &Parsed, model: String, text: String, format: &str) -> Res<Value> {
    let mut body = Map::new();
    body.insert("model".to_owned(), json!(model));
    body.insert("input".to_owned(), json!(text));
    body.insert("response_format".to_owned(), json!(format));
    body.insert("stream".to_owned(), json!(true));
    if let Some(voice) = parsed.value("--voice") {
        body.insert("voice".to_owned(), json!(voice));
    }
    if let Some(language) = parsed.value("--language") {
        body.insert("language".to_owned(), json!(language));
    }
    if let Some(speed) = parsed.value("--speed") {
        match speed.parse::<f64>() {
            Ok(s) if (0.25..=4.0).contains(&s) => body.insert("speed".to_owned(), json!(s)),
            _ => return usage_err("--speed is a number from 0.25 to 4"),
        };
    }
    if let Some(seed) = parsed.value("--seed") {
        match seed.parse::<u32>() {
            Ok(s) => body.insert("seed".to_owned(), json!(s)),
            Err(_) => return usage_err("--seed is a whole number from 0 to 4294967295"),
        };
    }
    Ok(Value::Object(body))
}

pub fn cmd_speak(client: &Client, parsed: &Parsed) -> Res<()> {
    let format = parsed.value("--format").unwrap_or("wav");
    if !["wav", "pcm", "mp3"].contains(&format) {
        return usage_err(
            "--format is wav, pcm or mp3 (use sox or ffmpeg for flac or other formats)",
        );
    }
    if let Some(l) = parsed.value("--language")
        && !((2..=3).contains(&l.len()) && l.bytes().all(|b| b.is_ascii_lowercase()))
    {
        return usage_err("--language is a language code such as de");
    }
    let sink = sink(parsed)?;
    let text = read_text(parsed)?;
    let model = resolve_model(client, parsed.value("--model"))?;
    // The service makes wav; mp3 is encoded from it here.
    let body = request(
        parsed,
        model,
        text,
        if format == "mp3" { "wav" } else { format },
    )?;

    let response = client
        .send("POST", "/v1/audio/speech", Some(&body.to_string()), WAIT)
        .map_err(|e| transport(client, e))?;
    if !(200..300).contains(&response.status) {
        let status = response.status;
        let text = response
            .text()
            .map_err(|e| Failure::Request(format!("Reading the answer failed: {e}")))?;
        let failure = api_error(client, status, &text);
        return Err(match failure {
            Failure::Request(m) if status == 400 && m.starts_with("Unknown voice") => {
                Failure::Request(format!(
                    "{m}{}",
                    voice_hint(client, parsed.value("--voice"))
                ))
            }
            other => other,
        });
    }
    let pending: Pending = Arc::default();
    watch_interrupt(pending.clone());
    let mut sink = open(sink, &pending)?;
    let mut reader = response.into_reader();
    let mut block = vec![0_u8; BLOCK];
    let mut total = 0_u64;
    let mut mp3 = (format == "mp3").then(mp3::Mp3Encoder::new);
    let mut encoded: Vec<u8> = Vec::new();
    let outcome: Result<(), Failure> = loop {
        let n = match reader.read(&mut block) {
            Ok(0) => {
                // The last MP3 frames.
                let Some(encoder) = mp3.as_mut() else {
                    break Ok(());
                };
                encoded.clear();
                if let Err(m) = encoder.finish(&mut encoded) {
                    break Err(Failure::Request(m));
                }
                match sink.write_block(&encoded) {
                    Ok(()) => break Ok(()),
                    Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
                        drop(reader);
                        discard(&sink, &pending);
                        return Ok(());
                    }
                    Err(e) => {
                        break Err(Failure::Request(format!("Writing the audio failed: {e}")));
                    }
                }
            }
            Ok(n) => n,
            // The service ends a failed stream without the closing chunk, so a cut stream is an error here and
            // never a short file that looks complete.
            Err(e) => {
                break Err(Failure::Request(format!(
                    "The audio stream was interrupted ({e}); the output is incomplete. See the log of the service"
                )));
            }
        };
        let piece: &[u8] = match mp3.as_mut() {
            Some(encoder) => {
                encoded.clear();
                if let Err(m) = encoder.push(&block[..n], &mut encoded) {
                    break Err(Failure::Request(m));
                }
                &encoded
            }
            None => &block[..n],
        };
        match sink.write_block(piece) {
            Ok(()) => total += piece.len() as u64,
            // The reader quit (for example `head -c`): nothing is wrong, stop asking for more. Dropping the
            // connection tells the service to stop the synthesis.
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
                drop(reader);
                discard(&sink, &pending);
                return Ok(());
            }
            Err(e) => break Err(Failure::Request(format!("Writing the audio failed: {e}"))),
        }
    };
    drop(reader);
    if let Err(failure) = outcome {
        discard(&sink, &pending);
        return Err(failure);
    }
    if let Open::File { file, part, path } = &mut sink {
        let finish = (if format == "wav" {
            close_wav(file, total)
        } else {
            Ok(())
        })
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::rename(&*part, &*path));
        if let Err(e) = finish {
            let _ = fs::remove_file(&*part);
            *pending.lock().unwrap_or_else(PoisonError::into_inner) = None;
            return Err(Failure::Request(format!(
                "Cannot write {}: {e}",
                path.display()
            )));
        }
        *pending.lock().unwrap_or_else(PoisonError::into_inner) = None;
        eprintln!("Wrote {} ({total} bytes)", path.display());
    }
    Ok(())
}

/// What to tell the user after the service did not know the voice: the voices there are and the option that names one.
fn voice_hint(client: &Client, named: Option<&str>) -> String {
    let names: Vec<String> = call(client, "GET", "/v1/audio/voices", None, SHORT)
        .ok()
        .and_then(|answer| {
            answer["voices"].as_array().map(|l| {
                l.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
        })
        .unwrap_or_default();
    let given = named.map_or_else(
        || {
            "No voice was named, and the voice that the model's settings name is not there."
                .to_owned()
        },
        |v| format!("There is no voice named \"{v}\"."),
    );
    if names.is_empty() {
        format!(
            "\n{given}\nThere are no voices yet. Add one with `eliezer voices add NAME FILE`, then name it with --voice NAME."
        )
    } else {
        format!(
            "\n{given}\nVoices: {}\nName one with --voice NAME (list them with `eliezer voices`).",
            names.join(", ")
        )
    }
}

/// Removes the half-written `.part` file, if there is one.
fn discard(sink: &Open, pending: &Pending) {
    if let Open::File { part, .. } = sink {
        let _ = fs::remove_file(part);
    }
    *pending.lock().unwrap_or_else(PoisonError::into_inner) = None;
}

/// On Ctrl-C: remove the `.part` file and exit with 130.
fn watch_interrupt(pending: Pending) {
    std::thread::spawn(move || {
        let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        if runtime.block_on(tokio::signal::ctrl_c()).is_err() {
            return;
        }
        if let Some(part) = pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            let _ = fs::remove_file(part);
        }
        eprintln!("\nInterrupted.");
        std::process::exit(130);
    });
}

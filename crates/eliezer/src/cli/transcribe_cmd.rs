//! `eliezer transcribe PATH... --out DIR`: transcribes a collection of audio files one after
//! the other through `POST /v1/audio/transcriptions` and writes `txt`, `srt`, `vtt` and `json`
//! files. The pure parts (collecting files, upload body, output formats) are in `transcribe.rs`.
use super::transcribe::{self as t, Item};
use super::{
    Client, Failure, Parsed, Res, SHORT, api_error, call, format, print_json, transport, usage_err,
};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::Instant,
};

/// What happened to one file.
struct Record {
    file: String,
    status: &'static str,
    audio_secs: f64,
    wall_secs: f64,
    outputs: Vec<String>,
    reason: Option<String>,
}

/// The state of the run, shared with the Ctrl-C watcher so that it can clean up and report.
struct Progress {
    total: usize,
    records: Vec<Record>,
    /// The `.part` file being written; removed when the run is interrupted.
    part: Option<PathBuf>,
    started: Instant,
}
impl Progress {
    fn count(&self, status: &str) -> usize {
        self.records.iter().filter(|r| r.status == status).count()
    }
    fn audio_secs(&self) -> f64 {
        self.records.iter().map(|r| r.audio_secs).sum()
    }
    fn summary(&self) -> String {
        let mut text = format!(
            "Summary: {} done, {} skipped, {} failed; audio {}, wall {}\n",
            self.count("done"),
            self.count("skipped"),
            self.count("failed"),
            format::duration_secs(self.audio_secs()),
            format::duration_secs(self.started.elapsed().as_secs_f64()),
        );
        let failed: Vec<&Record> = self
            .records
            .iter()
            .filter(|r| r.status == "failed")
            .collect();
        if !failed.is_empty() {
            text.push_str("Failed files:\n");
            for r in failed {
                text.push_str(&format!(
                    "  {}: {}\n",
                    r.file,
                    r.reason.as_deref().unwrap_or("unknown")
                ));
            }
        }
        text
    }
    fn json(&self) -> Value {
        json!({
            "done": self.count("done"),
            "skipped": self.count("skipped"),
            "failed": self.count("failed"),
            "audio_seconds": self.audio_secs(),
            "wall_seconds": self.started.elapsed().as_secs_f64(),
            "files": self.records.iter().map(|r| json!({
                "file": r.file, "status": r.status, "audio_seconds": r.audio_secs,
                "seconds": r.wall_secs, "outputs": r.outputs, "reason": r.reason,
            })).collect::<Vec<_>>(),
        })
    }
}
fn lock(progress: &Mutex<Progress>) -> MutexGuard<'_, Progress> {
    progress.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The settings every request shares.
struct Job {
    out: PathBuf,
    formats: Vec<String>,
    replace: bool,
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    timeout: std::time::Duration,
}

/// Why one file did not work out.
enum FileError {
    /// The file failed; the run goes on.
    Failed(String),
    /// The service is gone; the run stops after this file.
    Gone(String),
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The outputs of `item` that have to be written: all requested ones with `--replace`,
/// otherwise those that do not exist yet.
fn pending(item: &Item, out: &Path, formats: &[String], replace: bool) -> Vec<(String, PathBuf)> {
    formats
        .iter()
        .map(|f| (f.clone(), item.output(out, f)))
        .filter(|(_, path)| replace || fs::symlink_metadata(path).is_err())
        .collect()
}

pub fn cmd_transcribe(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let out = PathBuf::from(
        parsed
            .value("--out")
            .ok_or_else(|| Failure::Usage("--out DIR is required".to_owned()))?,
    );
    let formats = match parsed.value("--format") {
        Some(text) => t::parse_formats(text).map_err(Failure::Usage)?,
        None => t::FORMATS.iter().map(|f| (*f).to_owned()).collect(),
    };
    let extensions = match parsed.value("--extensions") {
        Some(text) => t::parse_extensions(text).map_err(Failure::Usage)?,
        None => t::DEFAULT_EXTENSIONS
            .iter()
            .map(|e| (*e).to_owned())
            .collect(),
    };
    let prompt = match (parsed.value("--prompt"), parsed.value("--prompt-file")) {
        (Some(_), Some(_)) => return usage_err("Give --prompt or --prompt-file, not both"),
        (Some(text), None) => Some(text.to_owned()),
        (None, Some(file)) => Some(
            fs::read_to_string(file)
                .map_err(|e| Failure::Usage(format!("Cannot read the prompt file {file}: {e}")))?
                .trim()
                .to_owned(),
        ),
        (None, None) => None,
    }
    .filter(|p| !p.is_empty());
    let language = parsed.value("--language").map(str::to_owned);
    if let Some(l) = &language
        && !(l == "auto"
            || ((2..=3).contains(&l.len()) && l.bytes().all(|b| b.is_ascii_lowercase())))
    {
        return usage_err("--language is auto or a language code such as de");
    }
    let limit = parsed.number("--limit")?;
    let timeout = parsed
        .number("--timeout-secs")?
        .map_or(t::DEFAULT_TIMEOUT, std::time::Duration::from_secs);
    let paths: Vec<PathBuf> = parsed.positional.iter().map(PathBuf::from).collect();
    let items =
        t::collect(&paths, parsed.flag("--recursive"), &extensions).map_err(Failure::Usage)?;
    if items.is_empty() {
        return usage_err(format!(
            "No audio files found (looked for: {})",
            extensions.join(", ")
        ));
    }
    let replace = parsed.flag("--replace");

    if parsed.flag("--dry-run") {
        return dry_run(&items, &out, &formats, replace, json);
    }

    // Everything done already: no request, not even the model list, is needed.
    let anything = items
        .iter()
        .any(|i| !pending(i, &out, &formats, replace).is_empty());
    let model = if anything {
        resolve_model(client, parsed.value("--model"))?
    } else {
        String::new()
    };
    let job = Job {
        out,
        formats,
        replace,
        model,
        language,
        prompt,
        timeout,
    };

    let progress = Arc::new(Mutex::new(Progress {
        total: items.len(),
        records: Vec::new(),
        part: None,
        started: Instant::now(),
    }));
    watch_interrupt(progress.clone());
    let mut processed = 0_u64;
    let mut gone: Option<String> = None;
    for (index, item) in items.iter().enumerate() {
        let name = item.display();
        let wanted = pending(item, &job.out, &job.formats, job.replace);
        let (record, line) = if wanted.is_empty() {
            let record = Record {
                file: name.clone(),
                status: "skipped",
                audio_secs: 0.0,
                wall_secs: 0.0,
                outputs: Vec::new(),
                reason: None,
            };
            (record, "skipped (outputs exist)".to_owned())
        } else {
            if limit.is_some_and(|n| processed >= n) {
                break;
            }
            processed += 1;
            let begun = Instant::now();
            let result = transcribe_one(client, item, &job, &wanted, &progress);
            let wall = begun.elapsed().as_secs_f64();
            lock(&progress).part = None;
            match result {
                Ok((audio, outputs)) => {
                    let speed = if wall > 0.0 { audio / wall } else { 0.0 };
                    let line = format!(
                        "audio {}  took {}  {speed:.1}x  done",
                        format::duration_secs(audio),
                        format::duration_secs(wall)
                    );
                    let record = Record {
                        file: name.clone(),
                        status: "done",
                        audio_secs: audio,
                        wall_secs: wall,
                        outputs,
                        reason: None,
                    };
                    (record, line)
                }
                Err(error) => {
                    let reason = match error {
                        FileError::Failed(m) => one_line(&m),
                        FileError::Gone(m) => {
                            let first = m.lines().next().unwrap_or("").to_owned();
                            gone = Some(m);
                            one_line(&first)
                        }
                    };
                    let line = format!("took {}  failed: {reason}", format::duration_secs(wall));
                    let record = Record {
                        file: name.clone(),
                        status: "failed",
                        audio_secs: 0.0,
                        wall_secs: wall,
                        outputs: Vec::new(),
                        reason: Some(reason),
                    };
                    (record, line)
                }
            }
        };
        eprintln!("[{}/{}] {name}  {line}", index + 1, items.len());
        lock(&progress).records.push(record);
        if gone.is_some() {
            break;
        }
    }

    let progress = lock(&progress);
    eprint!("{}", progress.summary());
    if json {
        print_json(&progress.json());
    }
    if let Some(message) = gone {
        eprintln!("The service went away; the run was stopped.");
        return Err(Failure::Unreachable(message));
    }
    let failed = progress.count("failed");
    if failed > 0 {
        return Err(Failure::Request(format!(
            "{failed} of {} files failed",
            progress.total
        )));
    }
    Ok(())
}

fn dry_run(items: &[Item], out: &Path, formats: &[String], replace: bool, json: bool) -> Res<()> {
    let mut files = Vec::new();
    let (mut todo, mut skip) = (0, 0);
    for item in items {
        let wanted = pending(item, out, formats, replace);
        if wanted.is_empty() {
            skip += 1;
        } else {
            todo += 1;
        }
        let outputs: Vec<String> = wanted
            .iter()
            .map(|(_, p)| p.display().to_string())
            .collect();
        if json {
            files.push(json!({
                "file": item.file.display().to_string(),
                "skipped": wanted.is_empty(),
                "outputs": outputs,
            }));
        } else if wanted.is_empty() {
            println!("{}  skipped (outputs exist)", item.file.display());
        } else {
            println!("{}  ->  {}", item.file.display(), outputs.join(", "));
        }
    }
    if json {
        print_json(&json!({"dry_run": true, "to_do": todo, "skipped": skip, "files": files}));
    } else {
        eprintln!("Dry run: {todo} to transcribe, {skip} skipped; nothing was sent.");
    }
    Ok(())
}

/// The model to use: the one named, else the only speech recognition model of the service.
fn resolve_model(client: &Client, named: Option<&str>) -> Res<String> {
    if let Some(model) = named {
        return Ok(model.to_owned());
    }
    let models = call(client, "GET", "/api/v1/models", None, SHORT)?;
    let keys: Vec<&str> = models["models"]
        .as_array()
        .map(|list| {
            list.iter()
                .filter(|m| m["type"] == "stt")
                .filter_map(|m| m["key"].as_str())
                .collect()
        })
        .unwrap_or_default();
    match keys.as_slice() {
        [only] => Ok((*only).to_owned()),
        [] => usage_err("The service has no speech recognition (stt) model; configure one first"),
        many => usage_err(format!(
            "The service has several speech recognition models ({}); name one with --model",
            many.join(", ")
        )),
    }
}

/// Sends one file and writes its outputs. Returns the audio length and the files written.
fn transcribe_one(
    client: &Client,
    item: &Item,
    job: &Job,
    wanted: &[(String, PathBuf)],
    progress: &Mutex<Progress>,
) -> Result<(f64, Vec<String>), FileError> {
    let data = fs::read(&item.file)
        .map_err(|e| FileError::Failed(format!("Cannot read the file: {e}")))?;
    let name = item
        .file
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    // One `verbose_json` request per file gives the text, the segment times and the length; the
    // four output formats are built from that one answer on the client, so a file is decoded
    // and transcribed once however many formats are wanted.
    let mut fields = vec![
        ("model", job.model.as_str()),
        ("response_format", "verbose_json"),
    ];
    if let Some(language) = &job.language {
        fields.push(("language", language));
    }
    if let Some(prompt) = &job.prompt {
        fields.push(("prompt", prompt));
    }
    let (content_type, body) = t::multipart(&fields, &name, &data);
    drop(data);
    let response = client
        .send_bytes(
            "POST",
            "/v1/audio/transcriptions",
            Some((&content_type, &body)),
            job.timeout,
        )
        .map_err(|e| match transport(client, e) {
            Failure::Unreachable(m) => FileError::Gone(m),
            Failure::Request(m) | Failure::Usage(m) => FileError::Failed(m),
        })?;
    drop(body);
    let status = response.status;
    let text = response
        .text()
        .map_err(|e| FileError::Failed(format!("Reading the answer failed: {e}")))?;
    if !(200..300).contains(&status) {
        return Err(FileError::Failed(match api_error(client, status, &text) {
            Failure::Request(m) | Failure::Usage(m) | Failure::Unreachable(m) => m,
        }));
    }
    let transcript = t::parse_transcript(&text).map_err(FileError::Failed)?;
    let mut written = Vec::new();
    for (format, path) in wanted {
        let content = t::render(format, &text, &transcript);
        write_atomic(path, content.as_bytes(), progress)
            .map_err(|e| FileError::Failed(format!("Cannot write {}: {e}", path.display())))?;
        written.push(path.display().to_string());
    }
    Ok((transcript.duration, written))
}

/// Writes `<path>.part`, then renames it, so that an interrupted run leaves no half file.
fn write_atomic(path: &Path, content: &[u8], progress: &Mutex<Progress>) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut name: OsString = path.file_name().unwrap_or_default().to_owned();
    name.push(".part");
    let part = path.with_file_name(name);
    lock(progress).part = Some(part.clone());
    let result = fs::write(&part, content).and_then(|()| fs::rename(&part, path));
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    lock(progress).part = None;
    result
}

/// On Ctrl-C: remove the `.part` file, report what was done so far and exit with 130.
fn watch_interrupt(progress: Arc<Mutex<Progress>>) {
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
        let progress = lock(&progress);
        if let Some(part) = &progress.part {
            let _ = fs::remove_file(part);
        }
        eprintln!("\nInterrupted.");
        eprint!("{}", progress.summary());
        std::process::exit(130);
    });
}

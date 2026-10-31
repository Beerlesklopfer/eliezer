//! Pure parts of `eliezer transcribe`: collecting the audio files, the multipart upload body, and
//! the four output formats built from one `verbose_json` answer. No network and no terminal I/O
//! here (only reading directories), so that tests can include this file directly.
use serde_json::Value;
use std::{
    collections::HashMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

/// File extensions searched for in a directory when `--extensions` is not given: the audio
/// formats the service decodes (MP3, M4A/AAC, Ogg Vorbis and Opus, FLAC, WAV).
pub const DEFAULT_EXTENSIONS: &[&str] = &["mp3", "m4a", "aac", "ogg", "oga", "opus", "flac", "wav"];

/// How long one request may wait for its answer when `--timeout-secs` is not given. The service
/// answers only when the whole file is transcribed, so this must cover a 3 hour recording on a
/// slow machine: 6 hours.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(6 * 3600);

/// The output formats, in the order they are listed and written.
pub const FORMATS: &[&str] = &["txt", "srt", "vtt", "json"];

/// Parses `--format txt,srt`: known names only, duplicates dropped, order kept.
pub fn parse_formats(text: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for name in text.split(',').map(|n| n.trim().to_ascii_lowercase()) {
        if !FORMATS.contains(&name.as_str()) {
            return Err(format!(
                "--format takes a comma separated list of txt, srt, vtt and json, not '{name}'"
            ));
        }
        if !out.contains(&name) {
            out.push(name);
        }
    }
    Ok(out)
}

/// Parses `--extensions mp3,.M4A`: lower case, without the dot, at least one entry.
pub fn parse_extensions(text: &str) -> Result<Vec<String>, String> {
    let list: Vec<String> = text
        .split(',')
        .map(|e| e.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|e| !e.is_empty())
        .collect();
    if list.is_empty() {
        return Err("--extensions needs at least one file extension".to_owned());
    }
    Ok(list)
}

/// One audio file to transcribe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// The file itself.
    pub file: PathBuf,
    /// Directory below `--out` the outputs go to (empty unless `--recursive` found the file in a
    /// sub-directory of the directory named on the command line).
    pub rel_dir: PathBuf,
    /// File name without its last extension; the outputs are `<stem>.<format>`.
    pub stem: OsString,
}
impl Item {
    /// Where the output of one format goes.
    pub fn output(&self, out: &Path, format: &str) -> PathBuf {
        let mut name = self.stem.clone();
        name.push(".");
        name.push(format);
        out.join(&self.rel_dir).join(name)
    }
    /// The name shown in progress lines: relative directory and file name.
    pub fn display(&self) -> String {
        let name = self.file.file_name().unwrap_or_default();
        self.rel_dir.join(name).display().to_string()
    }
}

fn hidden(name: &std::ffi::OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}

fn walk(
    dir: &Path,
    rel: &Path,
    recursive: bool,
    extensions: &[String],
    out: &mut Vec<Item>,
) -> Result<(), String> {
    let read = fs::read_dir(dir).map_err(|e| format!("Cannot read {}: {e}", dir.display()))?;
    let mut entries: Vec<_> = read.filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name();
        if hidden(&name) {
            continue;
        }
        let path = entry.path();
        // `metadata` follows symbolic links, so a link to a file counts as a file; a link to a
        // directory is not descended into, which rules out loops.
        let Ok(meta) = fs::metadata(&path) else {
            continue;
        };
        let is_link = entry.file_type().is_ok_and(|t| t.is_symlink());
        if meta.is_dir() {
            if recursive && !is_link {
                walk(&path, &rel.join(&name), recursive, extensions, out)?;
            }
        } else if meta.is_file() && has_extension(&path, extensions) {
            out.push(Item {
                stem: path.file_stem().unwrap_or(&name).to_owned(),
                file: path,
                rel_dir: rel.to_owned(),
            });
        }
    }
    Ok(())
}

fn has_extension(path: &Path, extensions: &[String]) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|e| extensions.contains(&e))
}

/// Collects the files named by `paths`. A file is taken as it is, whatever its extension; a
/// directory is searched for files with one of `extensions` (hidden files and directories are
/// ignored, sub-directories only with `recursive`). Two inputs that would write the same output
/// (`a.mp3` and `a.m4a`) are an error, because one would silently replace the other.
pub fn collect(
    paths: &[PathBuf],
    recursive: bool,
    extensions: &[String],
) -> Result<Vec<Item>, String> {
    let mut items = Vec::new();
    for path in paths {
        let meta =
            fs::metadata(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
        if meta.is_dir() {
            walk(path, Path::new(""), recursive, extensions, &mut items)?;
        } else {
            items.push(Item {
                file: path.clone(),
                rel_dir: PathBuf::new(),
                stem: path.file_stem().unwrap_or_default().to_owned(),
            });
        }
    }
    let mut seen: HashMap<(PathBuf, OsString), PathBuf> = HashMap::new();
    for item in &items {
        let key = (item.rel_dir.clone(), item.stem.clone());
        if let Some(first) = seen.insert(key, item.file.clone()) {
            return Err(format!(
                "{} and {} would write the same output files",
                first.display(),
                item.file.display()
            ));
        }
    }
    Ok(items)
}

/// Quotes a file name for the `filename="..."` parameter of a multipart part the way browsers do
/// (WHATWG HTML): a double quote, CR and LF are percent-encoded; everything else, including
/// commas, spaces, semicolons and non-ASCII characters, goes through as UTF-8.
pub fn quote_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '"' => out.push_str("%22"),
            '\r' => out.push_str("%0D"),
            '\n' => out.push_str("%0A"),
            c => out.push(c),
        }
    }
    out
}

/// Builds a `multipart/form-data` body: the text fields, then the file as field `file`. Returns
/// the value of the `Content-Type` header and the body.
pub fn multipart(fields: &[(&str, &str)], file_name: &str, data: &[u8]) -> (String, Vec<u8>) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let mut boundary = format!("----eliezer-{}-{nanos:x}", std::process::id());
    // The boundary must not occur in the data.
    while data
        .windows(boundary.len())
        .any(|window| window == boundary.as_bytes())
    {
        boundary.push('x');
    }
    let mut body = Vec::with_capacity(data.len() + 1024);
    for (name, value) in fields {
        body.extend(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\nContent-Type: application/octet-stream\r\n\r\n",
            quote_filename(file_name)
        )
        .as_bytes(),
    );
    body.extend(data);
    body.extend(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// One recognized piece of speech.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// The parts of a `verbose_json` answer the outputs are built from.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub duration: f64,
    pub segments: Vec<Segment>,
}

/// Reads the `verbose_json` answer of the service. Without `segments` the whole `text` becomes
/// one segment.
pub fn parse_transcript(body: &str) -> Result<Transcript, String> {
    let value: Value = serde_json::from_str(body)
        .map_err(|_| "The answer of the service is not valid JSON".to_owned())?;
    let duration = value["duration"].as_f64().unwrap_or(0.0).max(0.0);
    let segments = match value["segments"].as_array() {
        Some(list) => list
            .iter()
            .map(|s| {
                Some(Segment {
                    start: s["start"].as_f64()?,
                    end: s["end"].as_f64()?,
                    text: s["text"].as_str()?.to_owned(),
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("The answer of the service has a malformed segment")?,
        None => match value["text"].as_str() {
            Some(text) => vec![Segment {
                start: 0.0,
                end: duration,
                text: text.to_owned(),
            }],
            None => return Err("The answer of the service has neither segments nor text".into()),
        },
    };
    Ok(Transcript { duration, segments })
}

/// `01:02:03,456` (SRT, comma) or `01:02:03.456` (VTT, dot); hours at least two digits.
pub fn timestamp(seconds: f64, separator: char) -> String {
    let total = if seconds.is_finite() && seconds > 0.0 {
        (seconds * 1000.0).round() as u64
    } else {
        0
    };
    let (ms, secs) = (total % 1000, total / 1000);
    format!(
        "{:02}:{:02}:{:02}{separator}{ms:03}",
        secs / 3600,
        secs / 60 % 60,
        secs % 60
    )
}

/// Plain text, one segment per line.
pub fn txt(transcript: &Transcript) -> String {
    let mut out = String::new();
    for segment in &transcript.segments {
        out.push_str(segment.text.trim());
        out.push('\n');
    }
    out
}

/// SubRip: numbered cues with comma milliseconds.
pub fn srt(transcript: &Transcript) -> String {
    let mut out = String::new();
    for (number, s) in transcript.segments.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            number + 1,
            timestamp(s.start, ','),
            timestamp(s.end, ','),
            s.text.trim()
        ));
    }
    out
}

/// WebVTT: `WEBVTT` header, cues with dot milliseconds.
pub fn vtt(transcript: &Transcript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for s in &transcript.segments {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            timestamp(s.start, '.'),
            timestamp(s.end, '.'),
            s.text.trim()
        ));
    }
    out
}

/// The content of one output file. `json` is the answer exactly as received.
pub fn render(format: &str, raw: &str, transcript: &Transcript) -> String {
    match format {
        "txt" => txt(transcript),
        "srt" => srt(transcript),
        "vtt" => vtt(transcript),
        _ => raw.to_owned(),
    }
}

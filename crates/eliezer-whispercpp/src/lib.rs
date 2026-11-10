//! Speech recognition through whisper.cpp. Each loaded model is one `whisper-server` child process on a loopback port;
//! the service talks to it over HTTP (`GET /health` shows that the model is loaded, `POST /inference` with a
//! `multipart/form-data` body transcribes one piece of audio).
//!
//! Interfaces: docs/Runtimes.md says which parts were read from the program's source and which were run.
//!
//! The child is started without a shell, with a fixed list of arguments made of the configured paths, numbers and names
//! from a list, with a cleared environment (only what a GPU program needs is passed on) and in a private run directory
//! that is removed when the model is unloaded. Text of a request goes into form fields only, never into a command line
//! or a path. The audio conversion of the program (`--convert`, which starts ffmpeg) is never switched on: the service
//! sends 16 kHz mono 16 bit WAV files.

use eliezer_backend::{
    Backend, BackendError, BackendFuture, ChatRequest, Chunk, LoadSpec, Running,
    TranscribedSegment, TranscribedWord, Transcription, TranscriptionRequest,
};
use eliezer_core::config::{DTW_PRESETS, DevicePolicy};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{Child, Command},
    sync::{Mutex as AsyncMutex, mpsc},
};

/// Generations of this backend start here, above those of the music runtime (`1 << 41`), so a router can tell them apart.
pub const GENERATION_BASE: u64 = 1 << 42;

/// The variables a child process inherits from the service (see the music runtime).
const INHERITED_ENV: &[&str] = &[
    "PATH",
    "LD_LIBRARY_PATH",
    "HOME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
];

/// What the configuration says about one speech model.
#[derive(Debug, Clone)]
pub struct SpeechModel {
    /// The ggml model file.
    pub file: PathBuf,
    /// The language the program is started with (`auto` detects it).
    pub language: String,
    pub threads: Option<u32>,
    /// Break segments only between words (`split_on_word`).
    pub split_on_word: bool,
    /// Wrap segments after this many characters (`max_len`); 0 leaves the program's own wrapping. With word timing
    /// the program does not wrap at all and the service cuts the segments itself (`eliezer_speech::resegment`).
    pub max_segment_chars: u32,
    /// Word timing: the alignment heads preset of the program's `--dtw` for this model.
    pub word_timing: WordTiming,
}

/// Whether and how the program measures when each word is said (`--dtw`, token times by dynamic time warping over the
/// alignment heads of the model). The preset must fit the architecture of the model; a fine-tuned model uses the
/// preset of the model it was made from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WordTiming {
    /// No word timing: the segments of the program as they are.
    Off,
    /// The preset that fits the shape in the header of the ggml file; off when no preset fits.
    Auto,
    /// One of [`DTW_PRESETS`].
    Preset(String),
}

impl WordTiming {
    /// From the setting `dtw`: `auto`, `off` or a preset name (checked by the configuration).
    pub fn from_setting(value: &str) -> Self {
        match value {
            "off" => Self::Off,
            "auto" => Self::Auto,
            preset => Self::Preset(preset.to_owned()),
        }
    }
}

/// The `--dtw` preset for a ggml model file, from the shape in its header: the number of encoder and decoder layers,
/// the width and the vocabulary (multilingual models have 51865 or more tokens). `None` when the file cannot be read,
/// is no ggml whisper file, or has a shape without a preset (large-v1 and large-v2 have the same shape, so they get
/// none). Read from `whisper_model_load` of whisper.cpp v1.9.5: the magic `ggml` and then eleven 32 bit numbers.
pub fn dtw_preset_of(file: &Path) -> Option<&'static str> {
    use std::io::Read;
    let mut header = [0_u8; 48];
    std::fs::File::open(file)
        .ok()?
        .read_exact(&mut header)
        .ok()?;
    let number = |index: usize| {
        let at = index * 4;
        i32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    if number(0) as u32 != 0x6767_6d6c {
        return None;
    }
    let (vocab, audio_state, audio_layer, text_layer) =
        (number(1), number(3), number(5), number(9));
    let multilingual = vocab >= 51865;
    let preset = match (audio_layer, text_layer, audio_state, multilingual) {
        (4, 4, 384, false) => "tiny.en",
        (4, 4, 384, true) => "tiny",
        (6, 6, 512, false) => "base.en",
        (6, 6, 512, true) => "base",
        (12, 12, 768, false) => "small.en",
        (12, 12, 768, true) => "small",
        (24, 24, 1024, false) => "medium.en",
        (24, 24, 1024, true) => "medium",
        (32, 32, 1280, true) if vocab == 51866 => "large.v3",
        (32, 4, 1280, true) if vocab == 51866 => "large.v3.turbo",
        _ => return None,
    };
    DTW_PRESETS.contains(&preset).then_some(preset)
}

/// The settings of `[backend.whispercpp]` that reach the command line of the program.
#[derive(Debug, Clone, Default)]
pub struct ProgramSettings {
    pub flash_attn: Option<bool>,
    pub gpu_device: Option<u32>,
}

struct Instance {
    port: u16,
    /// The program was started with `--dtw`: the answers carry a time for every word.
    word_timing: bool,
    child: AsyncMutex<Child>,
    stopped: AtomicBool,
    /// The private run directory of this instance.
    directory: PathBuf,
}

pub struct WhispercppBackend {
    program: PathBuf,
    ports: (u16, u16),
    settings: ProgramSettings,
    library_path: Vec<PathBuf>,
    models: RwLock<HashMap<String, SpeechModel>>,
    instances: Mutex<HashMap<u64, Arc<Instance>>>,
    next_generation: AtomicU64,
    boundary_counter: AtomicU64,
    http: reqwest::Client,
    ready_timeout: Duration,
    request_timeout: Duration,
    stop_grace: Duration,
}

impl WhispercppBackend {
    /// `program` is the `whisper-server` binary and `ports` the loopback ports children may use (inclusive).
    pub fn new(program: PathBuf, ports: (u16, u16), settings: ProgramSettings) -> Self {
        Self::with_timeouts(
            program,
            ports,
            settings,
            Duration::from_secs(300),
            Duration::from_secs(3600),
        )
    }

    /// Like [`WhispercppBackend::new`]; `ready` is how long a model may take to load, `request` how long one piece
    /// may take.
    pub fn with_timeouts(
        program: PathBuf,
        ports: (u16, u16),
        settings: ProgramSettings,
        ready: Duration,
        request: Duration,
    ) -> Self {
        Self {
            program,
            ports,
            settings,
            library_path: Vec::new(),
            models: RwLock::new(HashMap::new()),
            instances: Mutex::new(HashMap::new()),
            next_generation: AtomicU64::new(GENERATION_BASE),
            boundary_counter: AtomicU64::new(0),
            // The children are on this machine: never through a proxy of the environment.
            http: reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_default(),
            ready_timeout: ready,
            request_timeout: request,
            stop_grace: Duration::from_secs(15),
        }
    }

    /// How long a child gets to end after it is told to stop, before it is killed.
    pub fn with_stop_grace(mut self, grace: Duration) -> Self {
        self.stop_grace = grace;
        self
    }

    /// Directories put in front of the library path of the child process (the ROCm libraries).
    pub fn with_library_path(mut self, directories: Vec<PathBuf>) -> Self {
        self.library_path = directories;
        self
    }

    /// The speech models of the configuration, by key (at start and on every reload).
    pub fn set_models(&self, models: HashMap<String, SpeechModel>) {
        match self.models.write() {
            Ok(mut guard) => *guard = models,
            Err(poisoned) => *poisoned.into_inner() = models,
        }
    }

    fn model(&self, key: &str) -> Option<SpeechModel> {
        match self.models.read() {
            Ok(guard) => guard.get(key).cloned(),
            Err(poisoned) => poisoned.into_inner().get(key).cloned(),
        }
    }

    fn instance(&self, generation: u64) -> Option<Arc<Instance>> {
        self.instances
            .lock()
            .ok()
            .and_then(|map| map.get(&generation).cloned())
    }

    /// A port of the range that no running instance uses and that nothing listens on.
    fn free_port(&self) -> Option<u16> {
        let used: Vec<u16> = self
            .instances
            .lock()
            .map(|map| map.values().map(|i| i.port).collect())
            .unwrap_or_default();
        (self.ports.0..=self.ports.1).find(|port| {
            !used.contains(port) && std::net::TcpListener::bind(("127.0.0.1", *port)).is_ok()
        })
    }
}

/// A language the program is started with: `auto` or a lowercase code of two or three letters.
fn plain_language(language: &str) -> bool {
    language == "auto"
        || ((2..=3).contains(&language.len()) && language.bytes().all(|b| b.is_ascii_lowercase()))
}

/// The arguments of `whisper-server`. Only the configured paths, numbers, a language code from a checked shape and a
/// preset name from [`DTW_PRESETS`] are in it. `cpu` keeps the program off the GPU (`--no-gpu`). `dtw` turns word
/// timing on (`--dtw PRESET`); the program cannot do that with flash attention and would silently drop it
/// (`whisper_init_with_params_no_state`, whisper.cpp v1.9.5), so flash attention is then off (`--no-flash-attn`)
/// whatever the setting says.
pub fn arguments(
    model: &SpeechModel,
    port: u16,
    run_directory: &Path,
    settings: &ProgramSettings,
    cpu: bool,
    dtw: Option<&str>,
) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = vec![
        "--model".into(),
        model.file.clone().into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string().into(),
        "--language".into(),
        model.language.clone().into(),
        // The program serves files from this directory; it is private and empty.
        "--public".into(),
        run_directory.into(),
    ];
    if let Some(threads) = model.threads {
        args.extend(["--threads".into(), threads.to_string().into()]);
    }
    if cpu {
        args.push("--no-gpu".into());
    } else {
        if let Some(device) = settings.gpu_device {
            args.extend(["--device".into(), device.to_string().into()]);
        }
        if dtw.is_none() {
            match settings.flash_attn {
                Some(true) => args.push("--flash-attn".into()),
                Some(false) => args.push("--no-flash-attn".into()),
                None => {}
            }
        }
    }
    if let Some(preset) = dtw {
        args.extend(["--dtw".into(), preset.into(), "--no-flash-attn".into()]);
    }
    args
}

/// A multipart body made of text fields and one file. The boundary is chosen so that no field contains it.
pub fn multipart_body(boundary: &str, fields: &[(&str, String)], file: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(file.len() + 1024);
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

/// The text fields of `POST /inference` for a request. `verbose_json` is asked for in every case: the service builds
/// its own `json`, `text`, `srt` and `vtt` from the segments, so it needs their times.
///
/// `split_on_word` is on by default (`backend.whispercpp.split_on_word`, per model `extra.split_on_word`): with `verbose_json` the program turns the token times on, and then it wraps
/// every segment after about 60 characters on a token boundary, which cuts words in two ("Füs" and "se.", measured with
/// whisper.cpp v1.9.5 on 2026-10-30). The service joins the segments with a space, so a cut word would reach the text.
/// `max_segment_chars` other than 0 is sent as `max_len`.
///
/// With `word_timing` the program must not wrap at all: its wrap after `max_len` characters (60 when the field is 0 or
/// missing, `server.cpp`) places the cut by an estimate from the number of characters, which put the segment starts
/// about 0.5 s before the first word and the ends before the last word (measured 2026-11-10). [`NO_WRAP`] keeps the
/// segments of the decoder whole; the service cuts them itself at word times (`eliezer_speech::resegment`).
pub fn inference_fields(
    request: &TranscriptionRequest,
    split_on_word: bool,
    max_segment_chars: u32,
    word_timing: bool,
) -> Vec<(&'static str, String)> {
    let mut fields = vec![
        ("response_format", "verbose_json".to_owned()),
        ("no_language_probabilities", "true".to_owned()),
        ("split_on_word", split_on_word.to_string()),
    ];
    if word_timing {
        fields.push(("max_len", NO_WRAP.to_string()));
    } else if max_segment_chars > 0 {
        fields.push(("max_len", max_segment_chars.to_string()));
    }
    if let Some(language) = &request.language {
        fields.push(("language", language.clone()));
    }
    if let Some(prompt) = &request.prompt {
        fields.push(("prompt", prompt.clone()));
    }
    if let Some(temperature) = request.temperature {
        fields.push(("temperature", format!("{temperature}")));
    }
    fields
}

/// A `max_len` no segment reaches: the program does not wrap (`whisper_wrap_segment` cuts only past `max_len`).
pub const NO_WRAP: u32 = 1_000_000;

/// The segments of a `verbose_json` answer. When every token of the answer has a DTW time (`t_dtw`, centiseconds; −1
/// without word timing), the segments also get their words ([`segment_words`]); otherwise none has words.
pub fn parse_transcription(value: &serde_json::Value) -> Option<Transcription> {
    let segments = value["segments"].as_array()?;
    let mut out = Vec::with_capacity(segments.len());
    let mut tokens = Vec::with_capacity(segments.len());
    for segment in segments {
        let (start, end) = (segment["start"].as_f64()?, segment["end"].as_f64()?);
        out.push(TranscribedSegment {
            start,
            end,
            text: segment["text"].as_str()?.trim().to_owned(),
            words: Vec::new(),
        });
        tokens.push(token_words(segment));
    }
    if tokens.iter().all(Option::is_some) {
        let tokens: Vec<Vec<TokenWord>> = tokens.into_iter().flatten().collect();
        for (segment, words) in out.iter_mut().zip(segment_words(&tokens)) {
            segment.words = words;
        }
    }
    Some(Transcription {
        language: value["language"]
            .as_str()
            .filter(|l| !l.is_empty())
            .map(str::to_owned),
        segments: out,
    })
}

/// A word put together from the tokens of a segment, before its end is known.
#[derive(Debug, Clone)]
struct TokenWord {
    text: String,
    /// The DTW time of the first token (seconds).
    start: f64,
    /// The DTW time of the last token (seconds).
    last: f64,
}

/// The words of one segment of `verbose_json` from its `words` list, which holds the tokens (`word` is the text of a
/// token): a token that starts with a space starts a word, any other token (the rest of a word, punctuation) belongs
/// to the word before it. `None` when a token has no DTW time.
fn token_words(segment: &serde_json::Value) -> Option<Vec<TokenWord>> {
    let mut words: Vec<TokenWord> = Vec::new();
    for token in segment["words"].as_array().map_or(&[][..], Vec::as_slice) {
        let text = token["word"].as_str()?;
        let dtw = token["t_dtw"].as_f64().filter(|t| *t >= 0.0)? / 100.0;
        match words.last_mut() {
            Some(word) if !text.starts_with(char::is_whitespace) => {
                word.text.push_str(text);
                word.last = dtw;
            }
            _ => words.push(TokenWord {
                text: text.trim_start().to_owned(),
                start: dtw,
                last: dtw,
            }),
        }
    }
    words.retain(|w| !w.text.trim().is_empty());
    for word in &mut words {
        word.text = word.text.trim_end().to_owned();
    }
    Some(words)
}

/// DTW marks a token about 0.15 s after its sound begins (measured 2026-11-10 against the ends of pauses found by
/// `ffmpeg silencedetect n=-32dB d=0.35` in two sermons: median +0.16 s and +0.14 s for the first word after a pause), so
/// word times are moved this much earlier.
pub const DTW_LEAD_SECS: f64 = 0.1;

/// Seconds a word lasts when nothing follows it closely: 0.15 s and 0.03 s for each character. Measured the same way:
/// the speech ends a median 0.2 to 0.3 s after the DTW time of the last word of a sentence before a pause.
fn word_length(text: &str) -> f64 {
    0.15 + 0.03 * text.chars().count() as f64
}

/// Gives every word its times. DTW gives one time per token, the moment the token is heard; a word starts at the time
/// of its first token, moved [`DTW_LEAD_SECS`] earlier. Where it ends is not measured and is estimated instead: at the
/// start of the next word, but not later than [`word_length`] after its own start (a pause follows otherwise), and not
/// before its last token is heard (plus 0.1 s, also bounded by the next word).
fn segment_words(tokens: &[Vec<TokenWord>]) -> Vec<Vec<TranscribedWord>> {
    let lead = |t: f64| (t - DTW_LEAD_SECS).max(0.0);
    let starts: Vec<f64> = tokens.iter().flatten().map(|w| lead(w.start)).collect();
    let mut index = 0;
    let mut out = Vec::with_capacity(tokens.len());
    for words in tokens {
        let mut list = Vec::with_capacity(words.len());
        for word in words {
            let start = starts[index];
            index += 1;
            let mut end = (start + word_length(&word.text)).max(lead(word.last) + 0.1);
            if let Some(next) = starts.get(index) {
                end = end.min(*next);
            }
            list.push(TranscribedWord {
                text: word.text.clone(),
                start,
                end: end.max(start),
            });
        }
        out.push(list);
    }
    out
}

async fn pump(model: String, stream: impl tokio::io::AsyncRead + Unpin) {
    let mut lines = BufReader::new(stream).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::debug!(model = %model, "whisper-server: {line}");
    }
}

fn remove_directory(directory: &Path) {
    let _ = std::fs::remove_dir_all(directory);
}

/// Creates the private run directory (mode 0700).
fn make_run_directory(port: u16) -> std::io::Result<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    let base = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let directory = base.join(format!("eliezer-whispercpp-{}-{port}", std::process::id()));
    // A leftover of an earlier instance on the same port.
    remove_directory(&directory);
    std::fs::DirBuilder::new().mode(0o700).create(&directory)?;
    Ok(directory)
}

impl Backend for WhispercppBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        Box::pin(async move {
            let model = self.model(&spec.model).ok_or(BackendError::Unavailable)?;
            if !plain_language(&model.language) {
                tracing::warn!(model = %spec.model, "The language of the speech model is not a language code");
                return Err(BackendError::InvalidRequest);
            }
            if !std::fs::metadata(&model.file).is_ok_and(|m| m.is_file()) {
                tracing::warn!(model = %spec.model, "The ggml file of the speech model cannot be read");
                return Err(BackendError::ModelFile);
            }
            let cpu = spec.device == DevicePolicy::Cpu;
            let dtw: Option<&str> = match &model.word_timing {
                WordTiming::Off => None,
                WordTiming::Preset(preset) => DTW_PRESETS.iter().copied().find(|p| p == preset),
                WordTiming::Auto => {
                    let found = dtw_preset_of(&model.file);
                    if found.is_none() {
                        tracing::warn!(model = %spec.model, "No word timing preset fits the speech model (extra.dtw); its segments keep the times of the program");
                    }
                    found
                }
            };
            let (child, port, directory, word_timing) = match self
                .launch(spec, &model, cpu, dtw)
                .await
            {
                Ok(started) => (started.0, started.1, started.2, dtw.is_some()),
                // A preset that does not fit the model makes the program stop while loading: start it once more
                // without word timing rather than not at all.
                Err(BackendError::Crashed) if dtw.is_some() => {
                    tracing::warn!(model = %spec.model, preset = dtw.unwrap_or_default(), "whisper-server did not load the model with word timing (extra.dtw); starting it without, its segments keep the times of the program");
                    let started = self.launch(spec, &model, cpu, None).await?;
                    (started.0, started.1, started.2, false)
                }
                Err(error) => return Err(error),
            };
            let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
            let mut placed = spec.clone();
            if placed.device == DevicePolicy::Auto {
                placed.device = DevicePolicy::Gpu;
            }
            let running = Running::new(placed, generation);
            let alive = running.alive_flag();
            let instance = Arc::new(Instance {
                port,
                word_timing,
                child: AsyncMutex::new(child),
                stopped: AtomicBool::new(false),
                directory,
            });
            if let Ok(mut map) = self.instances.lock() {
                map.insert(generation, instance.clone());
            }
            // A child that dies clears the flag the supervisor watches.
            tokio::spawn(async move {
                while !instance.stopped.load(Ordering::Acquire) {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if let Ok(Some(_)) = instance.child.lock().await.try_wait() {
                        alive.store(false, Ordering::Release);
                        break;
                    }
                }
            });
            tracing::info!(event = "whisper_server_ready", model = %spec.model, port, cpu, word_timing, "Speech model loaded");
            Ok(running)
        })
    }

    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            let instance = self
                .instances
                .lock()
                .ok()
                .and_then(|mut map| map.remove(&running.generation));
            if let Some(instance) = instance {
                instance.stopped.store(true, Ordering::Release);
                let mut child = instance.child.lock().await;
                let _ = child.start_kill();
                let _ = tokio::time::timeout(self.stop_grace, child.wait()).await;
                remove_directory(&instance.directory);
            }
            Ok(())
        })
    }

    fn health<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, bool> {
        Box::pin(
            async move { Ok(running.is_alive() && self.instance(running.generation).is_some()) },
        )
    }

    fn chat<'a>(
        &'a self,
        _: &'a Running,
        _: ChatRequest,
    ) -> BackendFuture<'a, mpsc::Receiver<Result<Chunk, BackendError>>> {
        // A speech model does not chat.
        Box::pin(async { Err(BackendError::InvalidRequest) })
    }

    fn transcribe<'a>(
        &'a self,
        running: &'a Running,
        request: TranscriptionRequest,
    ) -> BackendFuture<'a, Transcription> {
        Box::pin(async move {
            let instance = self
                .instance(running.generation)
                .ok_or(BackendError::Crashed)?;
            if request.wav.len() < 44 || !request.wav.starts_with(b"RIFF") {
                return Err(BackendError::InvalidRequest);
            }
            if request
                .language
                .as_deref()
                .is_some_and(|l| !plain_language(l))
            {
                return Err(BackendError::InvalidRequest);
            }
            let (split_on_word, max_segment_chars) = self
                .model(&running.spec.model)
                .map_or((true, 0), |m| (m.split_on_word, m.max_segment_chars));
            let fields = inference_fields(
                &request,
                split_on_word,
                max_segment_chars,
                instance.word_timing,
            );
            // A boundary that is in none of the texts.
            let boundary = loop {
                let n = self.boundary_counter.fetch_add(1, Ordering::Relaxed);
                let candidate = format!("eliezer-{}-{n}-{:x}", std::process::id(), nanos());
                if fields.iter().all(|(_, v)| !v.contains(&candidate)) {
                    break candidate;
                }
            };
            let body = multipart_body(&boundary, &fields, &request.wav);
            let response = self
                .http
                .post(format!("http://127.0.0.1:{}/inference", instance.port))
                .timeout(self.request_timeout)
                .header(
                    "Content-Type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(body)
                .send()
                .await
                .map_err(|_| {
                    // No answer: most likely the program died.
                    running.alive_flag().store(false, Ordering::Release);
                    BackendError::Crashed
                })?;
            let status = response.status();
            let bytes = response.bytes().await.map_err(|_| BackendError::Crashed)?;
            if status.is_client_error() {
                return Err(BackendError::InvalidRequest);
            }
            if !status.is_success() {
                tracing::warn!(
                    status = status.as_u16(),
                    "whisper-server could not transcribe"
                );
                return Err(BackendError::Crashed);
            }
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|value| parse_transcription(&value))
                .ok_or_else(|| {
                    tracing::warn!("whisper-server answered without segments");
                    BackendError::Crashed
                })
        })
    }
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}

impl WhispercppBackend {
    /// Starts the program for a model on a free port in a new run directory and waits until it has loaded the model.
    /// `Crashed` when the program ended while loading.
    async fn launch(
        &self,
        spec: &LoadSpec,
        model: &SpeechModel,
        cpu: bool,
        dtw: Option<&str>,
    ) -> Result<(Child, u16, PathBuf), BackendError> {
        let port = self.free_port().ok_or_else(|| {
            tracing::error!(model = %spec.model, "No free port in backend.whispercpp.port_range");
            BackendError::Unavailable
        })?;
        let directory = make_run_directory(port).map_err(|error| {
            tracing::error!(model = %spec.model, kind = ?error.kind(), "The run directory of the speech server cannot be made");
            BackendError::Unavailable
        })?;
        let mut command = Command::new(&self.program);
        command
            .args(arguments(model, port, &directory, &self.settings, cpu, dtw))
            .current_dir(&directory)
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for name in INHERITED_ENV {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        if !self.library_path.is_empty() {
            let mut dirs: Vec<PathBuf> = self.library_path.clone();
            if let Some(inherited) = std::env::var_os("LD_LIBRARY_PATH") {
                dirs.extend(std::env::split_paths(&inherited));
            }
            if let Ok(joined) = std::env::join_paths(dirs) {
                command.env("LD_LIBRARY_PATH", joined);
            }
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                tracing::error!(model = %spec.model, kind = ?error.kind(), "The whisper-server program cannot be started (backend.whispercpp.program)");
                remove_directory(&directory);
                return Err(BackendError::Unavailable);
            }
        };
        if let Some(out) = child.stdout.take() {
            tokio::spawn(pump(spec.model.clone(), out));
        }
        if let Some(err) = child.stderr.take() {
            tokio::spawn(pump(spec.model.clone(), err));
        }
        // The program listens only after the model is loaded; wait until /health says ready.
        let url = format!("http://127.0.0.1:{port}/health");
        let deadline = tokio::time::Instant::now() + self.ready_timeout;
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                tracing::error!(model = %spec.model, ?status, "The whisper-server program ended while loading");
                remove_directory(&directory);
                return Err(BackendError::Crashed);
            }
            if self.ready(&url).await {
                return Ok((child, port, directory));
            }
            if tokio::time::Instant::now() >= deadline {
                tracing::error!(model = %spec.model, "The whisper-server program did not load the model in time");
                let _ = child.start_kill();
                let _ = child.wait().await;
                remove_directory(&directory);
                return Err(BackendError::Unavailable);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// Does the program answer `GET /health` with a success status?
    async fn ready(&self, url: &str) -> bool {
        self.http
            .get(url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
    }
}

//! Music models through audio.cpp. Each loaded model is one `audiocpp_server` child process on a loopback port,
//! started with a configuration file that this backend writes (a private directory, removed when the model is
//! unloaded); the service talks to it over HTTP (`GET /v1/models` shows that the model is loaded,
//! `POST /v1/audio/speech` composes a song).
//!
//! Interfaces as probed on 2026-10-20 (docs/Runtimes.md).
//!
//! The child is started without a shell, with a fixed list of arguments, with a configuration file made of the
//! configured paths, names from a list and numbers, and with a cleared environment: only what a GPU program needs is
//! passed on. Request text goes into JSON strings only, never into a command line or a path.

use eliezer_backend::{
    Backend, BackendError, BackendFuture, ChatRequest, Chunk, LoadSpec, MusicRequest, Running,
};
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

/// Generations of this backend start here, above those of the image runtime (`1 << 40`), so a router can tell them apart.
pub const GENERATION_BASE: u64 = 1 << 41;

/// The variables a child process inherits from the service (see the image runtime).
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

/// What the configuration says about one music model.
#[derive(Debug, Clone)]
pub struct MusicModel {
    /// The package directory: the component files, their configs and the tokenizer.
    pub package: PathBuf,
    pub language_model: String,
    pub rvq_depth_decoder: String,
    pub flow_transformer: String,
    /// Flow matching steps (the program's default is 30).
    pub steps: Option<u32>,
    /// Flow guidance scale (the program's default is 1.7).
    pub guidance: Option<f64>,
}

struct Instance {
    port: u16,
    child: AsyncMutex<Child>,
    stopped: AtomicBool,
    /// The private directory with the configuration file of this instance.
    directory: PathBuf,
}

pub struct AudiocppBackend {
    program: PathBuf,
    ports: (u16, u16),
    compute: String,
    library_path: Vec<PathBuf>,
    models: RwLock<HashMap<String, MusicModel>>,
    instances: Mutex<HashMap<u64, Arc<Instance>>>,
    next_generation: AtomicU64,
    http: reqwest::Client,
    ready_timeout: Duration,
    request_timeout: Duration,
}

impl AudiocppBackend {
    /// `program` is the `audiocpp_server` binary, `ports` the loopback ports children may use (inclusive) and
    /// `compute` the backend of the program (`hip`, `cuda`, `vulkan` or `cpu`).
    pub fn new(program: PathBuf, ports: (u16, u16), compute: &str) -> Self {
        Self::with_timeouts(
            program,
            ports,
            compute,
            Duration::from_secs(600),
            Duration::from_secs(3600),
        )
    }

    /// Like [`AudiocppBackend::new`]; `ready` is how long a model may take to load, `request` how long one song
    /// may take.
    pub fn with_timeouts(
        program: PathBuf,
        ports: (u16, u16),
        compute: &str,
        ready: Duration,
        request: Duration,
    ) -> Self {
        Self {
            program,
            ports,
            compute: compute.to_owned(),
            library_path: Vec::new(),
            models: RwLock::new(HashMap::new()),
            instances: Mutex::new(HashMap::new()),
            next_generation: AtomicU64::new(GENERATION_BASE),
            // The children are on this machine: never through a proxy of the environment.
            http: reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_default(),
            ready_timeout: ready,
            request_timeout: request,
        }
    }

    /// Directories put in front of the library path of the child process (the ROCm libraries).
    pub fn with_library_path(mut self, directories: Vec<PathBuf>) -> Self {
        self.library_path = directories;
        self
    }

    /// The music models of the configuration, by key (at start and on every reload).
    pub fn set_models(&self, models: HashMap<String, MusicModel>) {
        match self.models.write() {
            Ok(mut guard) => *guard = models,
            Err(poisoned) => *poisoned.into_inner() = models,
        }
    }

    fn model(&self, key: &str) -> Option<MusicModel> {
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

/// A file name that stays inside the package: plain characters, no separator.
fn plain_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

/// The configuration file of one `audiocpp_server`: built from numbers, names of a list and the configured paths.
pub fn server_config(model: &MusicModel, port: u16, compute: &str, id: &str) -> serde_json::Value {
    let mut defaults = serde_json::json!({});
    if let Some(steps) = model.steps {
        defaults["num_inference_steps"] = serde_json::json!(steps);
    }
    if let Some(guidance) = model.guidance {
        defaults["guidance_scale"] = serde_json::json!(guidance);
    }
    serde_json::json!({
        "host": "127.0.0.1",
        "port": port,
        "backend": compute,
        "device": 0,
        // The weights go to the GPU at start, so the service knows when the model is ready.
        "lazy_load": false,
        "max_loaded_models": 1,
        "ui": false,
        "ui_management": false,
        // Waiting is the service's job; the program must not give up on a request on its own.
        "busy_timeout_ms": 0,
        "models": [{
            "id": id,
            "family": "minimax_music3",
            "path": model.package,
            "task": "gen",
            "mode": "offline",
            "busy_timeout_ms": 0,
            "session_options": {
                "minimax_music3.language_model_gguf": model.language_model,
                "minimax_music3.rvq_depth_decoder_gguf": model.rvq_depth_decoder,
                "minimax_music3.flow_transformer_gguf": model.flow_transformer,
                "minimax_music3.mem_saver": true,
            },
            "default_request_options": defaults,
        }],
    })
}

/// The arguments of `audiocpp_server`: only the configuration file and the log switch.
pub fn arguments(config: &Path) -> Vec<std::ffi::OsString> {
    vec![
        "--config".into(),
        config.into(),
        "--no-ui".into(),
        "--log".into(),
    ]
}

/// The body of `POST /v1/audio/speech` for a song: the description is the text of the program, the lyrics, the
/// length and the settings are its request options. Nothing of the request is put anywhere else.
pub fn speech_body(id: &str, request: &MusicRequest) -> serde_json::Value {
    let mut options = serde_json::json!({
        "lyrics": request.lyrics,
        "duration_sec": request.duration_sec.clamp(1, 300),
    });
    if let Some(steps) = request.steps {
        options["num_inference_steps"] = serde_json::json!(steps);
    }
    if let Some(seed) = request.seed {
        options["seed"] = serde_json::json!(seed);
    }
    serde_json::json!({
        "model": id,
        "input": request.description,
        "response_format": "wav",
        "options": options,
    })
}

async fn pump(model: String, stream: impl tokio::io::AsyncRead + Unpin) {
    let mut lines = BufReader::new(stream).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::debug!(model = %model, "audiocpp_server: {line}");
    }
}

fn remove_directory(directory: &Path) {
    let _ = std::fs::remove_dir_all(directory);
}

/// Writes the configuration file into a new private directory (mode 0700, the file 0600).
fn write_config(config: &serde_json::Value, port: u16) -> std::io::Result<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let base = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let directory = base.join(format!("eliezer-audiocpp-{}-{port}", std::process::id()));
    // A leftover of an earlier instance on the same port.
    remove_directory(&directory);
    std::fs::DirBuilder::new().mode(0o700).create(&directory)?;
    let file = directory.join("server.json");
    let result = (|| {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&file)?;
        out.write_all(config.to_string().as_bytes())
    })();
    match result {
        Ok(()) => Ok(directory),
        Err(error) => {
            remove_directory(&directory);
            Err(error)
        }
    }
}

impl Backend for AudiocppBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        Box::pin(async move {
            let model = self.model(&spec.model).ok_or(BackendError::Unavailable)?;
            let components = [
                &model.language_model,
                &model.rvq_depth_decoder,
                &model.flow_transformer,
            ];
            if !components.iter().all(|name| plain_name(name)) {
                tracing::warn!(model = %spec.model, "A component name of the music model is not a plain file name");
                return Err(BackendError::ModelFile);
            }
            let fixed = ["config.json", "condition_encoder.gguf", "vocoder.gguf"];
            let missing = components
                .iter()
                .map(|n| n.as_str())
                .chain(fixed)
                .any(|name| {
                    !std::fs::metadata(model.package.join(name)).is_ok_and(|m| m.is_file())
                });
            if missing {
                tracing::warn!(model = %spec.model, "A file of the music package cannot be read");
                return Err(BackendError::ModelFile);
            }
            let port = self.free_port().ok_or_else(|| {
                tracing::error!(model = %spec.model, "No free port in backend.audiocpp.port_range");
                BackendError::Unavailable
            })?;
            let config = server_config(&model, port, &self.compute, &spec.model);
            let directory = write_config(&config, port).map_err(|error| {
                tracing::error!(model = %spec.model, kind = ?error.kind(), "The configuration of the music server cannot be written");
                BackendError::Unavailable
            })?;
            let mut command = Command::new(&self.program);
            command
                .args(arguments(&directory.join("server.json")))
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
                    tracing::error!(model = %spec.model, kind = ?error.kind(), "The audiocpp_server program cannot be started (backend.audiocpp.program)");
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
            // Wait until the program lists the model as loaded; give up when it dies or takes too long.
            let url = format!("http://127.0.0.1:{port}/v1/models");
            let deadline = tokio::time::Instant::now() + self.ready_timeout;
            loop {
                if let Ok(Some(status)) = child.try_wait() {
                    tracing::error!(model = %spec.model, ?status, "The audiocpp_server program ended while loading");
                    remove_directory(&directory);
                    return Err(BackendError::Crashed);
                }
                if self.loaded(&url).await {
                    break;
                }
                if tokio::time::Instant::now() >= deadline {
                    tracing::error!(model = %spec.model, "The audiocpp_server program did not load the model in time");
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    remove_directory(&directory);
                    return Err(BackendError::Unavailable);
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
            let running = Running::new(spec.clone(), generation);
            let alive = running.alive_flag();
            let instance = Arc::new(Instance {
                port,
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
            tracing::info!(event = "audiocpp_server_ready", model = %spec.model, port, "Music model loaded");
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
                let _ = tokio::time::timeout(Duration::from_secs(15), child.wait()).await;
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
        // A music model does not chat.
        Box::pin(async { Err(BackendError::InvalidRequest) })
    }

    fn music<'a>(
        &'a self,
        running: &'a Running,
        request: MusicRequest,
    ) -> BackendFuture<'a, Vec<u8>> {
        Box::pin(async move {
            let instance = self
                .instance(running.generation)
                .ok_or(BackendError::Crashed)?;
            if request.lyrics.trim().is_empty() || request.description.trim().is_empty() {
                return Err(BackendError::InvalidRequest);
            }
            let body = speech_body(&running.spec.model, &request);
            let response = self
                .http
                .post(format!(
                    "http://127.0.0.1:{}/v1/audio/speech",
                    instance.port
                ))
                .timeout(self.request_timeout)
                .header("Content-Type", "application/json")
                .body(body.to_string())
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
                    "audiocpp_server could not compose the song"
                );
                return Err(BackendError::Crashed);
            }
            // The answer must be a WAV file; anything else (an error body with status 200) is a failure.
            if bytes.len() < 44 || !bytes.starts_with(b"RIFF") || &bytes[8..12] != b"WAVE" {
                tracing::warn!("audiocpp_server answered without a WAV file");
                return Err(BackendError::Crashed);
            }
            Ok(bytes.to_vec())
        })
    }
}

impl AudiocppBackend {
    /// Does the program list its model as loaded?
    async fn loaded(&self, url: &str) -> bool {
        let Ok(response) = self
            .http
            .get(url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        else {
            return false;
        };
        if !response.status().is_success() {
            return false;
        }
        let Ok(bytes) = response.bytes().await else {
            return false;
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return false;
        };
        value["data"]
            .as_array()
            .is_some_and(|models| models.iter().any(|m| m["loaded"] == true))
    }
}

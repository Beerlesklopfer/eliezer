//! Image models through stable-diffusion.cpp. Each loaded model is one `sd-server` child process on
//! a loopback port; the service talks to it over HTTP like to any OpenAI-style image endpoint
//! (`GET /v1/models` to see that it is ready, `POST /v1/images/generations` to draw).
//!
//! Interfaces as probed on 2026-10-19 and 2026-10-20 (docs/Runtimes.md); checked live with Qwen Image 2.1.
//!
//! The child is started without a shell, with a fixed list of arguments built from the configuration (paths,
//! numbers and names from a list), and with a cleared environment: only what a GPU program needs is passed on.

use eliezer_backend::{
    Backend, BackendError, BackendFuture, ChatRequest, Chunk, ImageRequest, LoadSpec, Running,
};
use std::{
    collections::HashMap,
    ffi::OsString,
    path::PathBuf,
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

/// Generations of this backend start here, so a router can tell them from those of the text runtime.
pub const GENERATION_BASE: u64 = 1 << 40;

/// The variables a child process inherits from the service: the program search path, the library path
/// (the bundled ROCm libraries), the cache location of the GPU compiler and the device selection.
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

/// What the configuration says about one image model.
#[derive(Debug, Clone)]
pub struct ImageModel {
    pub diffusion: PathBuf,
    pub text_encoder: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f64>,
    pub sampling_method: Option<String>,
    pub offload_to_cpu: bool,
}

struct Instance {
    port: u16,
    child: AsyncMutex<Child>,
    stopped: AtomicBool,
}

pub struct SdcppBackend {
    program: PathBuf,
    ports: (u16, u16),
    library_path: Vec<PathBuf>,
    models: RwLock<HashMap<String, ImageModel>>,
    instances: Mutex<HashMap<u64, Arc<Instance>>>,
    next_generation: AtomicU64,
    http: reqwest::Client,
    ready_timeout: Duration,
    request_timeout: Duration,
}

impl SdcppBackend {
    /// `program` is the `sd-server` binary, `ports` the loopback ports children may use (inclusive).
    pub fn new(program: PathBuf, ports: (u16, u16)) -> Self {
        Self::with_timeouts(
            program,
            ports,
            Duration::from_secs(300),
            Duration::from_secs(1800),
        )
    }

    /// Like [`SdcppBackend::new`]; `ready` is how long a model may take to load, `request` how long one
    /// generation may take.
    pub fn with_timeouts(
        program: PathBuf,
        ports: (u16, u16),
        ready: Duration,
        request: Duration,
    ) -> Self {
        Self {
            program,
            ports,
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

    /// Directories put in front of the library path of the child process (the ROCm libraries and `libomp`
    /// that `sd-server` needs; it carries no run path of its own).
    pub fn with_library_path(mut self, directories: Vec<PathBuf>) -> Self {
        self.library_path = directories;
        self
    }

    /// Replaces the table of loadable image models (at start and on configuration reload).
    pub fn set_models(&self, models: HashMap<String, ImageModel>) {
        match self.models.write() {
            Ok(mut guard) => *guard = models,
            Err(poisoned) => *poisoned.into_inner() = models,
        }
    }

    fn model(&self, key: &str) -> Option<ImageModel> {
        match self.models.read() {
            Ok(guard) => guard.get(key).cloned(),
            Err(poisoned) => poisoned.into_inner().get(key).cloned(),
        }
    }

    fn instance(&self, generation: u64) -> Option<Arc<Instance>> {
        self.instances.lock().ok()?.get(&generation).cloned()
    }

    /// A loopback port of the range that nothing listens on and no instance of ours uses.
    fn free_port(&self) -> Option<u16> {
        let used: Vec<u16> = self
            .instances
            .lock()
            .map(|i| i.values().map(|i| i.port).collect())
            .unwrap_or_default();
        (self.ports.0..=self.ports.1)
            .filter(|p| !used.contains(p))
            .find(|p| std::net::TcpListener::bind(("127.0.0.1", *p)).is_ok())
    }
}

/// The arguments of `sd-server` for one model. Paths are passed as they are (no shell is involved);
/// everything else comes from numbers and names the configuration validated.
pub fn arguments(model: &ImageModel, port: u16) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec!["--diffusion-model".into(), model.diffusion.clone().into()];
    if let Some(encoder) = &model.text_encoder {
        args.extend(["--llm".into(), encoder.clone().into()]);
    }
    if let Some(vae) = &model.vae {
        args.extend(["--vae".into(), vae.clone().into()]);
    }
    args.extend([
        "--listen-ip".into(),
        "127.0.0.1".into(),
        "--listen-port".into(),
        port.to_string().into(),
    ]);
    if model.offload_to_cpu {
        args.push("--offload-to-cpu".into());
    }
    if let Some(steps) = model.steps {
        args.extend(["--steps".into(), steps.to_string().into()]);
    }
    if let Some(cfg) = model.cfg_scale {
        args.extend(["--cfg-scale".into(), format!("{cfg}").into()]);
    }
    if let Some(method) = &model.sampling_method {
        args.extend(["--sampling-method".into(), method.clone().into()]);
    }
    args.extend(["--log-level".into(), "warn".into()]);
    args
}

/// The prompt as the program gets it. `sd-server` reads native settings from a block
/// `<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` inside the prompt; a user must not be able to set them,
/// so the name of the block is removed wherever it appears.
pub fn clean_prompt(prompt: &str) -> String {
    let mut text = prompt.to_owned();
    // Removing a match can join its neighbours into a new one, so repeat until none is left.
    while let Some(at) = text.to_ascii_lowercase().find("sd_cpp_extra_args") {
        text.replace_range(at..at + "sd_cpp_extra_args".len(), "");
    }
    text
}

/// `WIDTHxHEIGHT`, each between 64 and 2048.
pub fn valid_size(size: &str) -> bool {
    size.split_once('x').is_some_and(|(w, h)| {
        [w, h].iter().all(|part| {
            part.bytes().all(|b| b.is_ascii_digit())
                && part.parse::<u32>().is_ok_and(|n| (64..=2048).contains(&n))
        })
    })
}

async fn pump(model: String, stream: impl tokio::io::AsyncRead + Unpin + Send + 'static) {
    let mut lines = BufReader::new(stream).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let line: String = line.chars().take(300).collect();
        tracing::debug!(event = "sd_server", model = %model, "{line}");
    }
}

impl Backend for SdcppBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        Box::pin(async move {
            let model = self.model(&spec.model).ok_or(BackendError::Unavailable)?;
            for path in [
                Some(&model.diffusion),
                model.text_encoder.as_ref(),
                model.vae.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if !std::fs::metadata(path).is_ok_and(|m| m.is_file()) {
                    tracing::warn!(model = %spec.model, "A file of the image model cannot be read");
                    return Err(BackendError::ModelFile);
                }
            }
            let port = self.free_port().ok_or_else(|| {
                tracing::error!(model = %spec.model, "No free port in backend.sdcpp.port_range");
                BackendError::Unavailable
            })?;
            let mut command = Command::new(&self.program);
            command
                .args(arguments(&model, port))
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
            let mut child = command.spawn().map_err(|error| {
                tracing::error!(model = %spec.model, kind = ?error.kind(), "The sd-server program cannot be started (backend.sdcpp.program)");
                BackendError::Unavailable
            })?;
            if let Some(out) = child.stdout.take() {
                tokio::spawn(pump(spec.model.clone(), out));
            }
            if let Some(err) = child.stderr.take() {
                tokio::spawn(pump(spec.model.clone(), err));
            }
            // Wait until the server answers; give up when it dies or takes too long.
            let url = format!("http://127.0.0.1:{port}/v1/models");
            let deadline = tokio::time::Instant::now() + self.ready_timeout;
            loop {
                if let Ok(Some(status)) = child.try_wait() {
                    tracing::error!(model = %spec.model, ?status, "The sd-server program ended while loading");
                    return Err(BackendError::Crashed);
                }
                let answered = self
                    .http
                    .get(&url)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success());
                if answered {
                    break;
                }
                if tokio::time::Instant::now() >= deadline {
                    tracing::error!(model = %spec.model, "The sd-server program did not become ready in time");
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    return Err(BackendError::Unavailable);
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
            let running = Running::new(spec.clone(), generation);
            let alive = running.alive_flag();
            let instance = Arc::new(Instance {
                port,
                child: AsyncMutex::new(child),
                stopped: AtomicBool::new(false),
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
            tracing::info!(event = "sd_server_ready", model = %spec.model, port, "Image model loaded");
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
        // An image model does not chat.
        Box::pin(async { Err(BackendError::InvalidRequest) })
    }

    fn image<'a>(
        &'a self,
        running: &'a Running,
        request: ImageRequest,
    ) -> BackendFuture<'a, Vec<String>> {
        Box::pin(async move {
            let instance = self
                .instance(running.generation)
                .ok_or(BackendError::Crashed)?;
            if request.size.as_deref().is_some_and(|s| !valid_size(s)) {
                return Err(BackendError::InvalidRequest);
            }
            let prompt = clean_prompt(&request.prompt);
            if prompt.trim().is_empty() {
                return Err(BackendError::InvalidRequest);
            }
            let mut body = serde_json::json!({
                "prompt": prompt,
                "n": request.count.clamp(1, 4),
                "output_format": "png",
            });
            if let Some(size) = &request.size {
                body["size"] = serde_json::json!(size);
            }
            let response = self
                .http
                .post(format!(
                    "http://127.0.0.1:{}/v1/images/generations",
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
                    "sd-server could not draw the image"
                );
                return Err(BackendError::Crashed);
            }
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).map_err(|_| BackendError::Crashed)?;
            let images: Vec<String> = value["data"]
                .as_array()
                .ok_or(BackendError::Crashed)?
                .iter()
                .filter_map(|d| d["b64_json"].as_str().map(str::to_owned))
                .collect();
            if images.is_empty() {
                return Err(BackendError::Crashed);
            }
            Ok(images)
        })
    }
}

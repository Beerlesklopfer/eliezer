//! Interchangeable runtime and test bench without GPU or network.
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::mpsc;

pub type BackendFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, BackendError>> + Send + 'a>>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendError {
    Unavailable,
    Crashed,
    Cancelled,
    InvalidRequest,
    /// The model file is missing or unreadable (moved, deleted, wrong permissions).
    ModelFile,
}
impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unavailable => "Backend is not available",
            Self::Crashed => "Backend terminated unexpectedly",
            Self::Cancelled => "Request was cancelled",
            Self::InvalidRequest => "Invalid backend request",
            Self::ModelFile => "The model file cannot be read",
        })
    }
}
impl std::error::Error for BackendError {}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadSpec {
    pub model: String,
    pub context: u32,
    pub parallel: u32,
    pub bytes: u64,
    /// What was asked for (`auto` leaves the choice to the supervisor). A running instance carries
    /// the device it was actually placed on (`gpu` or `cpu`).
    pub device: eliezer_core::config::DevicePolicy,
}
impl LoadSpec {
    /// Whether an instance loaded with `self` already serves a request for `wanted`: same model,
    /// context, parallelism and size, and the same device unless the request leaves it open.
    pub fn satisfies(&self, wanted: &Self) -> bool {
        use eliezer_core::config::DevicePolicy::Auto;
        self.model == wanted.model
            && self.context == wanted.context
            && self.parallel == wanted.parallel
            && self.bytes == wanted.bytes
            && (wanted.device == Auto || wanted.device == self.device)
    }
}
#[derive(Debug, Clone)]
pub struct Running {
    pub spec: LoadSpec,
    pub generation: u64,
    alive: Arc<AtomicBool>,
}
impl Running {
    /// A freshly started instance. `generation` must be unique per backend.
    pub fn new(spec: LoadSpec, generation: u64) -> Self {
        Self {
            spec,
            generation,
            alive: Arc::new(AtomicBool::new(true)),
        }
    }
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
    /// Shared flag a runtime clears when its worker dies, so the supervisor sees the crash.
    pub fn alive_flag(&self) -> Arc<AtomicBool> {
        self.alive.clone()
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chunk {
    Text(String),
    Thinking(String),
    Tool {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    Done {
        finish_reason: String,
        input_tokens: u64,
        output_tokens: u64,
    },
}
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub input: String,
    pub reasoning: Option<String>,
}
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub input_tokens: u64,
}
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub model: String,
    pub prompt: String,
    pub count: u32,
    pub size: Option<String>,
}
/// Embedded runtimes use handles; the interface does not require an HTTP endpoint.
pub trait Backend: Send + Sync {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running>;
    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()>;
    fn health<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, bool>;
    fn chat<'a>(
        &'a self,
        running: &'a Running,
        request: ChatRequest,
    ) -> BackendFuture<'a, mpsc::Receiver<Result<Chunk, BackendError>>>;
    /// Embeddings; runtimes without support keep this default.
    fn embed<'a>(&'a self, _: &'a Running, _: EmbedRequest) -> BackendFuture<'a, Embeddings> {
        Box::pin(async { Err(BackendError::Unavailable) })
    }
    /// Generated images as base64-encoded PNG data.
    fn image<'a>(&'a self, _: &'a Running, _: ImageRequest) -> BackendFuture<'a, Vec<String>> {
        Box::pin(async { Err(BackendError::Unavailable) })
    }
}
#[derive(Debug, Clone)]
pub struct FakeConfig {
    pub load_delay: Duration,
    pub chunk_delay: Duration,
    pub chunks: Vec<Chunk>,
    pub fail_load: bool,
    /// Loading fails as if the model file had been moved away.
    pub missing_file: bool,
    pub crash_after: Option<usize>,
}
impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            load_delay: Duration::ZERO,
            chunk_delay: Duration::ZERO,
            chunks: vec![
                Chunk::Text("test reply".into()),
                Chunk::Done {
                    finish_reason: "stop".into(),
                    input_tokens: 1,
                    output_tokens: 1,
                },
            ],
            fail_load: false,
            missing_file: false,
            crash_after: None,
        }
    }
}
pub struct FakeBackend {
    pub config: FakeConfig,
    generation: AtomicU64,
}
impl FakeBackend {
    pub fn new(config: FakeConfig) -> Self {
        Self {
            config,
            generation: AtomicU64::new(0),
        }
    }
}
impl FakeBackend {
    /// Number of instances started so far (for tests of reloading after a crash).
    pub fn starts(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }
}
impl Backend for FakeBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        Box::pin(async move {
            if spec.context == 0 || spec.parallel == 0 || spec.parallel > spec.context {
                return Err(BackendError::InvalidRequest);
            }
            tokio::time::sleep(self.config.load_delay).await;
            if self.config.missing_file {
                return Err(BackendError::ModelFile);
            }
            if self.config.fail_load {
                return Err(BackendError::Unavailable);
            }
            Ok(Running {
                spec: spec.clone(),
                generation: self.generation.fetch_add(1, Ordering::Relaxed),
                alive: Arc::new(AtomicBool::new(true)),
            })
        })
    }
    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            running.alive.store(false, Ordering::Release);
            Ok(())
        })
    }
    fn health<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, bool> {
        Box::pin(async move { Ok(running.is_alive()) })
    }
    fn chat<'a>(
        &'a self,
        running: &'a Running,
        request: ChatRequest,
    ) -> BackendFuture<'a, mpsc::Receiver<Result<Chunk, BackendError>>> {
        Box::pin(async move {
            if !running.is_alive() {
                return Err(BackendError::Crashed);
            }
            if request.model != running.spec.model {
                return Err(BackendError::InvalidRequest);
            }
            let (tx, rx) = mpsc::channel(1);
            let config = self.config.clone();
            let alive = running.alive.clone();
            tokio::spawn(async move {
                for (index, chunk) in config.chunks.into_iter().enumerate() {
                    tokio::select! { _ = tx.closed() => return, _ = tokio::time::sleep(config.chunk_delay) => {} }
                    if config.crash_after == Some(index) {
                        alive.store(false, Ordering::Release);
                    }
                    if !alive.load(Ordering::Acquire) {
                        let _ = tx.send(Err(BackendError::Crashed)).await;
                        return;
                    }
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                }
            });
            Ok(rx)
        })
    }
    fn embed<'a>(
        &'a self,
        running: &'a Running,
        request: EmbedRequest,
    ) -> BackendFuture<'a, Embeddings> {
        Box::pin(async move {
            if !running.is_alive() {
                return Err(BackendError::Crashed);
            }
            if request.model != running.spec.model {
                return Err(BackendError::InvalidRequest);
            }
            tokio::time::sleep(self.config.chunk_delay).await;
            let vectors = request
                .input
                .iter()
                .map(|text| {
                    (1..=8_u32)
                        .map(|i| {
                            let sum: u32 = text.bytes().map(|b| u32::from(b) * i).sum();
                            f32::from((sum % 1000) as u16) / 1000.0
                        })
                        .collect()
                })
                .collect();
            let input_tokens = request
                .input
                .iter()
                .map(|t| t.split_whitespace().count() as u64)
                .sum();
            Ok(Embeddings {
                vectors,
                input_tokens,
            })
        })
    }
    fn image<'a>(
        &'a self,
        running: &'a Running,
        request: ImageRequest,
    ) -> BackendFuture<'a, Vec<String>> {
        Box::pin(async move {
            if !running.is_alive() {
                return Err(BackendError::Crashed);
            }
            if request.model != running.spec.model || request.count == 0 {
                return Err(BackendError::InvalidRequest);
            }
            tokio::time::sleep(self.config.chunk_delay).await;
            Ok(vec![FAKE_PNG_BASE64.to_owned(); request.count as usize])
        })
    }
}
/// A valid 1x1 PNG, so clients that decode the fake image succeed.
pub const FAKE_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

pub mod supervisor;

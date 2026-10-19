//! Austauschbare Laufzeit und Prüfstand ohne GPU oder Netzwerk.
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
}
impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unavailable => "Backend ist nicht verfügbar",
            Self::Crashed => "Backend wurde unerwartet beendet",
            Self::Cancelled => "Anfrage wurde abgebrochen",
            Self::InvalidRequest => "Ungültige Backend-Anfrage",
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
}
#[derive(Debug, Clone)]
pub struct Running {
    pub spec: LoadSpec,
    pub generation: u64,
    alive: Arc<AtomicBool>,
}
impl Running {
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
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
/// Eingebettete Laufzeiten verwenden Handles; die Schnittstelle verlangt keinen HTTP-Endpunkt.
pub trait Backend: Send + Sync {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running>;
    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()>;
    fn health<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, bool>;
    fn chat<'a>(
        &'a self,
        running: &'a Running,
        request: ChatRequest,
    ) -> BackendFuture<'a, mpsc::Receiver<Result<Chunk, BackendError>>>;
}
#[derive(Debug, Clone)]
pub struct FakeConfig {
    pub load_delay: Duration,
    pub chunk_delay: Duration,
    pub chunks: Vec<Chunk>,
    pub fail_load: bool,
    pub crash_after: Option<usize>,
}
impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            load_delay: Duration::ZERO,
            chunk_delay: Duration::ZERO,
            chunks: vec![
                Chunk::Text("Prüfantwort".into()),
                Chunk::Done {
                    finish_reason: "stop".into(),
                    input_tokens: 1,
                    output_tokens: 1,
                },
            ],
            fail_load: false,
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
impl Backend for FakeBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        Box::pin(async move {
            if spec.context == 0 || spec.parallel == 0 || spec.parallel > spec.context {
                return Err(BackendError::InvalidRequest);
            }
            tokio::time::sleep(self.config.load_delay).await;
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
}

pub mod supervisor;

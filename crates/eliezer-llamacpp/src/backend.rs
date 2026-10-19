//! `Backend` implementation on top of the llama.cpp engine.
use crate::engine::{self, Cmd, EmbedJob, Job, STREAM_CAPACITY, Setup, Worker};
use eliezer_backend::{
    Backend, BackendError, BackendFuture, ChatRequest, Chunk, EmbedRequest, Embeddings, LoadSpec,
    Running,
};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::sync::{mpsc, oneshot};

/// A model file the backend may load, by model key. Keys come from the configuration only.
#[derive(Debug, Clone)]
pub struct ModelFile {
    pub path: PathBuf,
    pub embedding: bool,
}

pub struct LlamaCppBackend {
    models: RwLock<HashMap<String, ModelFile>>,
    workers: Mutex<HashMap<u64, Arc<Worker>>>,
    next_generation: AtomicU64,
    gpu_layers: u32,
}

impl LlamaCppBackend {
    /// `gpu_layers` is the number of layers offloaded to the GPU (999 means all).
    pub fn new(gpu_layers: u32) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            workers: Mutex::new(HashMap::new()),
            next_generation: AtomicU64::new(1),
            gpu_layers,
        }
    }

    /// Replaces the table of loadable models (called at start and on configuration reload).
    pub fn set_models(&self, models: HashMap<String, ModelFile>) {
        match self.models.write() {
            Ok(mut guard) => *guard = models,
            Err(poisoned) => *poisoned.into_inner() = models,
        }
    }

    fn model(&self, key: &str) -> Option<ModelFile> {
        match self.models.read() {
            Ok(guard) => guard.get(key).cloned(),
            Err(poisoned) => poisoned.into_inner().get(key).cloned(),
        }
    }

    fn worker(&self, running: &Running) -> Result<Arc<Worker>, BackendError> {
        let workers = self.workers.lock().map_err(|_| BackendError::Crashed)?;
        workers
            .get(&running.generation)
            .cloned()
            .ok_or(BackendError::Crashed)
    }
}

impl Backend for LlamaCppBackend {
    fn start<'a>(&'a self, spec: &'a LoadSpec) -> BackendFuture<'a, Running> {
        Box::pin(async move {
            let file = self.model(&spec.model).ok_or(BackendError::Unavailable)?;
            // Say plainly when the file was moved or deleted, before llama.cpp reports it obscurely.
            match std::fs::File::open(&file.path).and_then(|f| f.metadata()) {
                Ok(meta) if meta.is_file() => {}
                other => {
                    tracing::error!(
                        model = %spec.model,
                        path = %file.path.display(),
                        reason = %other.map_or_else(|e| e.to_string(), |_| "not a regular file".to_owned()),
                        "The model file cannot be read"
                    );
                    return Err(BackendError::ModelFile);
                }
            }
            let worker = engine::spawn(Setup {
                key: spec.model.clone(),
                path: file.path,
                context: spec.context,
                parallel: spec.parallel,
                embedding: file.embedding,
                gpu_layers: self.gpu_layers,
            })
            .await?;
            let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
            let running = Running::new(spec.clone(), generation);
            // A dying worker clears the flag the supervisor watches.
            let alive = running.alive_flag();
            let worker_alive = worker.alive.clone();
            tokio::spawn(async move {
                while worker_alive.load(Ordering::Acquire) {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                alive.store(false, Ordering::Release);
            });
            self.workers
                .lock()
                .map_err(|_| BackendError::Crashed)?
                .insert(generation, Arc::new(worker));
            Ok(running)
        })
    }

    fn stop<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            let worker = self
                .workers
                .lock()
                .map_err(|_| BackendError::Crashed)?
                .remove(&running.generation);
            let Some(worker) = worker else {
                return Ok(());
            };
            let _ = worker.cmd.send(Cmd::Stop);
            // The thread ends after it released the model; wait for that off the async workers.
            let worker = Arc::into_inner(worker);
            if let Some(Worker { join, .. }) = worker {
                let _ = tokio::task::spawn_blocking(move || join.join()).await;
            }
            Ok(())
        })
    }

    fn health<'a>(&'a self, running: &'a Running) -> BackendFuture<'a, bool> {
        Box::pin(async move {
            Ok(self
                .worker(running)
                .map(|w| w.alive.load(Ordering::Acquire))
                .unwrap_or(false))
        })
    }

    fn chat<'a>(
        &'a self,
        running: &'a Running,
        request: ChatRequest,
    ) -> BackendFuture<'a, mpsc::Receiver<Result<Chunk, BackendError>>> {
        Box::pin(async move {
            let worker = self.worker(running)?;
            let body: serde_json::Value =
                serde_json::from_str(&request.input).map_err(|_| BackendError::InvalidRequest)?;
            let (tx, rx) = mpsc::channel(STREAM_CAPACITY);
            let (admitted, answer) = oneshot::channel();
            worker
                .cmd
                .send(Cmd::Generate(Job {
                    body,
                    reasoning: request.reasoning,
                    tx,
                    admitted,
                }))
                .map_err(|_| BackendError::Crashed)?;
            answer.await.map_err(|_| BackendError::Crashed)??;
            Ok(rx)
        })
    }

    fn embed<'a>(
        &'a self,
        running: &'a Running,
        request: EmbedRequest,
    ) -> BackendFuture<'a, Embeddings> {
        Box::pin(async move {
            let worker = self.worker(running)?;
            let (reply, answer) = oneshot::channel();
            worker
                .cmd
                .send(Cmd::Embed(EmbedJob {
                    input: request.input,
                    reply,
                }))
                .map_err(|_| BackendError::Crashed)?;
            answer.await.map_err(|_| BackendError::Crashed)?
        })
    }
}

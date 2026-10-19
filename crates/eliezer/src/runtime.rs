//! Connects configuration, supervisor and program interface into the running service.
use eliezer_api::{App, managed_router};
use eliezer_backend::{
    Backend, BackendError, BackendFuture, ChatRequest, Chunk, LoadSpec, Running,
    supervisor::{Model, Settings, Supervisor},
};
use eliezer_core::{
    config::{Config, GpuVendor, ModelKind},
    memory::{self, gb_bytes},
    scheduler::{Clock, Instance, State},
};
use std::{os::unix::fs::PermissionsExt, path::Path, sync::Arc, time::Instant};
use tokio::sync::mpsc;

/// Monotonic clock since service start.
struct SystemClock(Instant);
impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.0.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Placeholder until the real runtimes (M4): loading fails with a clear message.
struct NoBackend;
impl Backend for NoBackend {
    fn start<'a>(&'a self, _: &'a LoadSpec) -> BackendFuture<'a, Running> {
        Box::pin(async { Err(BackendError::Unavailable) })
    }
    fn stop<'a>(&'a self, _: &'a Running) -> BackendFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
    fn health<'a>(&'a self, _: &'a Running) -> BackendFuture<'a, bool> {
        Box::pin(async { Ok(false) })
    }
    fn chat<'a>(
        &'a self,
        _: &'a Running,
        _: ChatRequest,
    ) -> BackendFuture<'a, mpsc::Receiver<Result<Chunk, BackendError>>> {
        Box::pin(async { Err(BackendError::Unavailable) })
    }
}

/// Reads API keys (one line per key, `#` for comments); the file must be 0600.
pub fn read_keys(path: &Path) -> Result<Vec<String>, &'static str> {
    let mode = std::fs::metadata(path)
        .map_err(|_| "Key file cannot be read")?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err("Key file must be readable by the service user only (0600)");
    }
    let text = std::fs::read_to_string(path).map_err(|_| "Key file cannot be read")?;
    let keys: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect();
    if keys.is_empty() {
        return Err("Key file contains no key");
    }
    Ok(keys)
}

fn models(config: &Config) -> Vec<Model> {
    let mut out = Vec::new();
    for m in &config.model {
        let Ok(bytes) = memory::estimate(m, m.default_context, m.parallel, None, None) else {
            tracing::warn!(event = "model_skipped", key = %m.key, "Memory requirement cannot be calculated; model skipped");
            continue;
        };
        out.push(Model {
            instance: Instance {
                key: m.key.clone(),
                state: State::Unloaded,
                bytes,
                priority: m.priority,
                pinned: m.pinned,
                image: m.kind == ModelKind::Image,
                in_flight: 0,
                last_used_ms: 0,
            },
            default_spec: LoadSpec {
                model: m.key.clone(),
                context: m.default_context,
                parallel: m.parallel,
                bytes,
            },
            max_context: m.max_context,
        });
    }
    out
}

fn settings(config: &Config) -> Result<Settings, &'static str> {
    let budget = gb_bytes(config.gpu.vram_budget_gb)
        .and_then(|b| gb_bytes(config.gpu.reserve_gb).map(|r| b.saturating_sub(r)))
        .map_err(|_| "VRAM budget is invalid")?;
    let s = &config.scheduler;
    Ok(Settings {
        budget,
        idle_ms: s.idle_unload_secs.saturating_mul(1000),
        wait_ms: s.queue_wait_secs.saturating_mul(1000),
        queue_max: s.queue_max,
        retry_ms: s.failed_retry_secs.saturating_mul(1000),
    })
}

/// Reads the API keys named in the configuration; without a key file the API is open.
pub fn keys(config: &Config) -> Result<Vec<String>, &'static str> {
    match &config.server.api_keys_file {
        Some(path) => read_keys(path),
        None => {
            tracing::warn!(
                event = "no_auth",
                "No key file configured; the API is open without authentication"
            );
            Ok(Vec::new())
        }
    }
}

/// The running service: router for the HTTP server plus the handles needed to reload it.
pub struct Runtime {
    pub router: axum::Router,
    pub supervisor: Supervisor,
    app: App,
    listen: std::net::SocketAddr,
}
impl Runtime {
    /// Applies a reloaded configuration. Loaded models stay, removed ones are unloaded once idle.
    /// The listen address cannot change without a restart.
    pub async fn reload(&self, config: Arc<Config>, keys: Vec<String>) -> Result<(), &'static str> {
        let settings = settings(&config)?;
        if config.server.listen != self.listen {
            tracing::warn!(
                event = "restart_required",
                "The listen address changed; it takes effect only after a restart"
            );
        }
        self.supervisor
            .reconfigure(models(&config), settings)
            .await
            .map_err(|_| "Supervisor is not running")?;
        self.app.replace(config, keys);
        Ok(())
    }
}

/// Starts the supervisor and builds the router.
pub fn build(config: Arc<Config>, keys: Vec<String>) -> Result<Runtime, &'static str> {
    let supervisor = Supervisor::spawn(
        models(&config),
        Arc::new(NoBackend),
        Arc::new(SystemClock(Instant::now())),
        settings(&config)?,
    );
    let gpu = &config.gpu;
    let root = std::path::PathBuf::from("/sys/class/drm");
    let vendor = match gpu.vendor {
        GpuVendor::Auto => eliezer_gpu::detect_vendor(&root, gpu.index),
        other => other,
    };
    let probe = Arc::new(eliezer_gpu::sources(
        vendor,
        gpu.measure,
        gpu.index,
        gb_bytes(gpu.vram_budget_gb).map_err(|_| "VRAM budget is invalid")?,
        root,
    ));
    tracing::info!(event = "gpu", ?vendor, "GPU vendor determined");
    let listen = config.server.listen;
    let app = App::new(config, supervisor.clone(), Arc::new(NoBackend), probe, keys);
    Ok(Runtime {
        router: managed_router(app.clone()),
        supervisor,
        app,
        listen,
    })
}

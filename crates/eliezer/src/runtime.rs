//! Verbindet Konfiguration, Verwalter und Programmschnittstelle zum laufenden Dienst.
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

/// Monotone Uhr seit Dienststart.
struct SystemClock(Instant);
impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.0.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Platzhalter bis zu den echten Laufzeiten (M4): Laden schlägt mit klarer Meldung fehl.
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

/// Liest API-Schlüssel (eine Zeile je Schlüssel, `#` für Kommentare); die Datei muss 0600 sein.
pub fn read_keys(path: &Path) -> Result<Vec<String>, &'static str> {
    let mode = std::fs::metadata(path)
        .map_err(|_| "Schlüsseldatei kann nicht gelesen werden")?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err("Schlüsseldatei darf nur für den Dienstbenutzer lesbar sein (0600)");
    }
    let text =
        std::fs::read_to_string(path).map_err(|_| "Schlüsseldatei kann nicht gelesen werden")?;
    let keys: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect();
    if keys.is_empty() {
        return Err("Schlüsseldatei enthält keinen Schlüssel");
    }
    Ok(keys)
}

fn models(config: &Config) -> Vec<Model> {
    let mut out = Vec::new();
    for m in &config.model {
        let Ok(bytes) = memory::estimate(m, m.default_context, m.parallel, None, None) else {
            tracing::warn!(event = "model_skipped", key = %m.key, "Speicherbedarf nicht berechenbar; Modell übersprungen");
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

/// Startet den Verwalter und liefert Router und Verwalter-Handle.
pub fn build(
    config: Arc<Config>,
    keys: Vec<String>,
) -> Result<(axum::Router, Supervisor), &'static str> {
    let budget = gb_bytes(config.gpu.vram_budget_gb)
        .and_then(|b| gb_bytes(config.gpu.reserve_gb).map(|r| b.saturating_sub(r)))
        .map_err(|_| "VRAM-Budget ist ungültig")?;
    let s = &config.scheduler;
    let supervisor = Supervisor::spawn(
        models(&config),
        Arc::new(NoBackend),
        Arc::new(SystemClock(Instant::now())),
        Settings {
            budget,
            idle_ms: s.idle_unload_secs.saturating_mul(1000),
            wait_ms: s.queue_wait_secs.saturating_mul(1000),
            queue_max: s.queue_max,
            retry_ms: s.failed_retry_secs.saturating_mul(1000),
        },
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
        gb_bytes(gpu.vram_budget_gb).map_err(|_| "VRAM-Budget ist ungültig")?,
        root,
    ));
    tracing::info!(event = "gpu", ?vendor, "GPU-Hersteller bestimmt");
    let app = App {
        config,
        supervisor: supervisor.clone(),
        backend: Arc::new(NoBackend),
        probe,
        keys: Arc::new(keys),
    };
    Ok((managed_router(app), supervisor))
}

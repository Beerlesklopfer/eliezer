//! Connects configuration, supervisor and program interface into the running service.
use eliezer_api::{App, managed_router};
use eliezer_backend::{
    Backend, LoadSpec,
    routing::RoutingBackend,
    supervisor::{Model, Settings, Supervisor},
};
use eliezer_core::{
    config::{Config, GpuVendor, ModelKind},
    memory::{self, gb_bytes},
    scheduler::{Clock, Instance, State},
};
use std::{os::unix::fs::PermissionsExt, path::Path, sync::Arc, time::Instant};

/// Monotonic clock since service start.
struct SystemClock(Instant);
impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.0.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

mod none {
    use eliezer_backend::{
        Backend, BackendError, BackendFuture, ChatRequest, Chunk, LoadSpec, Running,
    };
    use tokio::sync::mpsc;

    /// Used when no runtime is compiled in: loading fails with a clear message.
    pub struct NoBackend;
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
}
use none::NoBackend;

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
                // Pictures and music are heavy jobs of a program of their own: they take the GPU one at a time.
                image: matches!(m.kind, ModelKind::Image | ModelKind::Music),
                in_flight: 0,
                last_used_ms: 0,
            },
            default_spec: LoadSpec {
                model: m.key.clone(),
                context: m.default_context,
                parallel: m.parallel,
                bytes,
                device: m.device,
            },
            max_context: m.max_context,
        });
    }
    out
}

/// The runtimes that serve models: embedded llama.cpp (when compiled in) for language and embedding models and
/// stable-diffusion.cpp as a child process for image models (when `[backend.sdcpp]` is configured) and audio.cpp as a
/// child process for music models (when `[backend.audiocpp]` is configured).
struct Runtimes {
    backend: Arc<dyn Backend>,
    router: Arc<RoutingBackend>,
    #[cfg(feature = "llamacpp")]
    llamacpp: Arc<eliezer_llamacpp::LlamaCppBackend>,
    sdcpp: Option<Arc<eliezer_sdcpp::SdcppBackend>>,
    audiocpp: Option<Arc<eliezer_audiocpp::AudiocppBackend>>,
}
impl Runtimes {
    fn new(config: &Config) -> Self {
        #[cfg(feature = "llamacpp")]
        let (text, llamacpp): (Arc<dyn Backend>, _) = {
            // 999 offloads every layer to the GPU.
            let llamacpp = Arc::new(eliezer_llamacpp::LlamaCppBackend::new(999));
            llamacpp.set_models(llamacpp_models(config));
            tracing::info!(event = "runtime", "Embedded llama.cpp runtime is available");
            (llamacpp.clone(), llamacpp)
        };
        #[cfg(not(feature = "llamacpp"))]
        let text: Arc<dyn Backend> = {
            tracing::warn!(
                event = "no_runtime",
                "This build contains no model runtime; loading models will fail"
            );
            Arc::new(NoBackend)
        };
        let sdcpp = config.backend.sdcpp.as_ref().map(|sd| {
            tracing::info!(
                event = "runtime",
                "Image models are served by stable-diffusion.cpp"
            );
            Arc::new(
                eliezer_sdcpp::SdcppBackend::new(
                    sd.program.clone(),
                    (sd.port_range[0], sd.port_range[1]),
                )
                .with_library_path(sd.library_path.clone()),
            )
        });
        let image: Arc<dyn Backend> = match &sdcpp {
            Some(backend) => backend.clone(),
            None => Arc::new(NoBackend),
        };
        let audiocpp = config.backend.audiocpp.as_ref().map(|audio| {
            tracing::info!(event = "runtime", "Music models are served by audio.cpp");
            Arc::new(
                eliezer_audiocpp::AudiocppBackend::new(
                    audio.program.clone(),
                    (audio.port_range[0], audio.port_range[1]),
                    &audio.compute,
                )
                .with_library_path(audio.library_path.clone()),
            )
        });
        let mut router = RoutingBackend::new(text, image, eliezer_sdcpp::GENERATION_BASE);
        if let Some(backend) = &audiocpp {
            router = router.with_audio(backend.clone(), eliezer_audiocpp::GENERATION_BASE);
        }
        let router = Arc::new(router);
        let runtimes = Self {
            backend: router.clone(),
            router,
            #[cfg(feature = "llamacpp")]
            llamacpp,
            sdcpp,
            audiocpp,
        };
        runtimes.update(config);
        runtimes
    }
    fn update(&self, config: &Config) {
        #[cfg(feature = "llamacpp")]
        self.llamacpp.set_models(llamacpp_models(config));
        let images = sdcpp_models(config);
        self.router
            .set_image_models(images.keys().cloned().collect());
        if let Some(sdcpp) = &self.sdcpp {
            sdcpp.set_models(images);
        }
        let songs = audiocpp_models(config);
        self.router
            .set_audio_models(songs.keys().cloned().collect());
        if let Some(audiocpp) = &self.audiocpp {
            audiocpp.set_models(songs);
        }
    }
}

/// The image models of the configuration with the files and settings the child process needs, by key.
fn sdcpp_models(config: &Config) -> std::collections::HashMap<String, eliezer_sdcpp::ImageModel> {
    config
        .model
        .iter()
        .filter(|m| {
            m.backend == eliezer_core::config::BackendKind::Sdcpp && m.kind == ModelKind::Image
        })
        .filter_map(|m| {
            let extra = m.extra.as_ref();
            Some((
                m.key.clone(),
                eliezer_sdcpp::ImageModel {
                    diffusion: m.source.file.clone()?,
                    text_encoder: extra.and_then(|e| e.text_encoder.clone()),
                    vae: extra.and_then(|e| e.vae.clone()),
                    tokenizer: extra.and_then(|e| e.tokenizer.clone()),
                    steps: extra.and_then(|e| e.steps),
                    cfg_scale: extra.and_then(|e| e.cfg_scale),
                    sampling_method: extra.and_then(|e| e.sampling_method.clone()),
                    offload_to_cpu: extra.and_then(|e| e.offload_to_cpu).unwrap_or(true),
                },
            ))
        })
        .collect()
}

/// The music models of the configuration with the package and settings the child process needs, by key.
fn audiocpp_models(
    config: &Config,
) -> std::collections::HashMap<String, eliezer_audiocpp::MusicModel> {
    use eliezer_core::config::{
        MUSIC_FLOW_TRANSFORMER, MUSIC_LANGUAGE_MODEL, MUSIC_RVQ_DEPTH_DECODER,
    };
    config
        .model
        .iter()
        .filter(|m| {
            m.backend == eliezer_core::config::BackendKind::Audiocpp && m.kind == ModelKind::Music
        })
        .filter_map(|m| {
            let extra = m.extra.as_ref();
            let name = |chosen: Option<&String>, default: &str| {
                chosen.cloned().unwrap_or_else(|| default.to_owned())
            };
            Some((
                m.key.clone(),
                eliezer_audiocpp::MusicModel {
                    package: m.source.file.clone()?,
                    language_model: name(
                        extra.and_then(|e| e.language_model.as_ref()),
                        MUSIC_LANGUAGE_MODEL,
                    ),
                    rvq_depth_decoder: name(
                        extra.and_then(|e| e.rvq_depth_decoder.as_ref()),
                        MUSIC_RVQ_DEPTH_DECODER,
                    ),
                    flow_transformer: name(
                        extra.and_then(|e| e.flow_transformer.as_ref()),
                        MUSIC_FLOW_TRANSFORMER,
                    ),
                    steps: extra.and_then(|e| e.steps),
                    guidance: extra.and_then(|e| e.cfg_scale),
                },
            ))
        })
        .collect()
}

/// Model files the llama.cpp runtime may load: only configured models of that backend, by key.
#[cfg(feature = "llamacpp")]
fn llamacpp_models(
    config: &Config,
) -> std::collections::HashMap<String, eliezer_llamacpp::ModelFile> {
    config
        .model
        .iter()
        .filter(|m| m.backend == eliezer_core::config::BackendKind::Llamacpp)
        .filter_map(|m| {
            Some((
                m.key.clone(),
                eliezer_llamacpp::ModelFile {
                    path: m.source.file.clone()?,
                    embedding: m.kind == ModelKind::Embedding,
                    projector: m.source.projector.clone(),
                },
            ))
        })
        .collect()
}

fn settings(config: &Config) -> Result<Settings, &'static str> {
    let budget = gb_bytes(config.gpu.vram_budget_gb)
        .and_then(|b| gb_bytes(config.gpu.reserve_gb).map(|r| b.saturating_sub(r)))
        .map_err(|_| "VRAM budget is invalid")?;
    // Without a RAM budget nothing is placed on the CPU.
    let ram_budget = if config.cpu.ram_budget_gb > 0.0 {
        gb_bytes(config.cpu.ram_budget_gb)
            .and_then(|b| gb_bytes(config.cpu.reserve_gb).map(|r| b.saturating_sub(r)))
            .map_err(|_| "RAM budget is invalid")?
    } else {
        0
    };
    let s = &config.scheduler;
    Ok(Settings {
        budget,
        ram_budget,
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

/// The Hugging Face token for downloads: the file `huggingface.token_file` (mode 0600; a file that others
/// can read is not used) or else the environment variable `HF_TOKEN`. The token is never logged; the
/// messages only say where it came from.
pub fn hf_token(config: &Config) -> Option<eliezer_hub::Token> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(path) = &config.huggingface.token_file {
        match std::fs::metadata(path) {
            Ok(meta) if meta.permissions().mode() & 0o077 != 0 => {
                tracing::warn!(
                    event = "token_permissions",
                    mode = format!("{:o}", meta.permissions().mode() & 0o777),
                    "The Hugging Face token file can be read by others and is not used; make it 0600"
                );
            }
            Ok(_) => {
                if let Some(token) = std::fs::read_to_string(path)
                    .ok()
                    .and_then(|text| eliezer_hub::Token::new(&text))
                {
                    return Some(token);
                }
                tracing::warn!(
                    event = "token_empty",
                    "The Hugging Face token file is empty or not readable"
                );
            }
            // A missing file is the normal case: no token configured yet.
            Err(_) => {}
        }
    }
    std::env::var("HF_TOKEN")
        .ok()
        .and_then(|text| eliezer_hub::Token::new(&text))
}

/// The running service: router for the HTTP server plus the handles needed to reload it.
pub struct Runtime {
    pub router: axum::Router,
    pub supervisor: Supervisor,
    app: App,
    runtimes: Runtimes,
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
        self.runtimes.update(&config);
        self.app.set_hf_token(hf_token(&config));
        self.app.replace(config, keys);
        Ok(())
    }
}

/// Starts the supervisor and builds the router.
pub fn build(config: Arc<Config>, keys: Vec<String>) -> Result<Runtime, &'static str> {
    let runtimes = Runtimes::new(&config);
    let supervisor = Supervisor::spawn(
        models(&config),
        runtimes.backend.clone(),
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
    let mut app = App::new(
        config,
        supervisor.clone(),
        runtimes.backend.clone(),
        probe,
        keys,
    );
    app.set_hf_token(hf_token(&app.config.get()));
    // Files that were fetched earlier are checked against their pins: a deviation is reported, not fixed.
    {
        let root = app.config.get().storage.models_dir.clone();
        tokio::task::spawn_blocking(move || {
            let report = eliezer_api::pin_report(&root, false);
            for pin in report["pins"].as_array().into_iter().flatten() {
                for file in pin["files"].as_array().into_iter().flatten() {
                    if file["status"] != "ok" {
                        tracing::warn!(
                            event = "pin_deviation",
                            repo = pin["repo"].as_str().unwrap_or_default(),
                            file = file["file"].as_str().unwrap_or_default(),
                            problem = file["status"].as_str().unwrap_or_default(),
                            "A pinned model file differs from its pin"
                        );
                    }
                }
            }
        });
    }
    // The search of Hugging Face reads public metadata only; the configuration can switch it off.
    match eliezer_hub::HfClient::new("https://huggingface.co", std::time::Duration::from_secs(15)) {
        Ok(client) => app = app.with_hub(std::sync::Arc::new(client)),
        Err(error) => {
            tracing::warn!(event = "hub", %error, "The Hugging Face search is unavailable")
        }
    }
    Ok(Runtime {
        router: managed_router(app.clone()),
        supervisor,
        app,
        runtimes,
        listen,
    })
}

//! Strict configuration; file system access goes through a resolver supplied by the caller.
use serde::Deserialize;
use serde_path_to_error::Segment;
use std::{
    collections::BTreeSet,
    fmt,
    net::SocketAddr,
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    pub gpu: GpuConfig,
    #[serde(default)]
    pub cpu: CpuConfig,
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    #[serde(default)]
    pub huggingface: HuggingFaceConfig,
    #[serde(default)]
    pub backend: BackendConfig,
    #[serde(default)]
    pub model: Vec<ModelConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    pub api_keys_file: Option<PathBuf>,
    /// System user the service runs as; `eliezer --import` hands imported files to it.
    pub run_user: Option<String>,
    /// Group of imported files; defaults to the primary group of `run_user`.
    pub run_group: Option<String>,
    #[serde(default = "request_timeout")]
    pub request_timeout_secs: u64,
}
fn request_timeout() -> u64 {
    600
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    pub models_dir: PathBuf,
}
impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            models_dir: "/var/lib/eliezer/models".into(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuConfig {
    pub vendor: GpuVendor,
    pub index: u32,
    pub vram_budget_gb: f64,
    pub reserve_gb: f64,
    pub measure: ProbeMethod,
}
impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            vendor: GpuVendor::Auto,
            index: 0,
            vram_budget_gb: 8.0,
            reserve_gb: 2.0,
            measure: ProbeMethod::Auto,
        }
    }
}
/// Main memory that language models may use when the GPU is not available to them. Off by default.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CpuConfig {
    /// Memory for models placed on the CPU; 0 turns CPU placement off.
    pub ram_budget_gb: f64,
    /// Kept free for the system out of the budget.
    pub reserve_gb: f64,
}
impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            ram_budget_gb: 0.0,
            reserve_gb: 4.0,
        }
    }
}

/// Where a model may run. Diffusion models cannot use the CPU in a useful time, so they always get
/// the GPU; language and embedding models may fall back to the CPU (`auto`).
#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DevicePolicy {
    /// The GPU, unless an image model has priority on it or the model does not fit there.
    #[default]
    Auto,
    Gpu,
    Cpu,
}
impl DevicePolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    #[default]
    Auto,
    Nvidia,
    Amd,
    None,
}
#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProbeMethod {
    #[default]
    Auto,
    Nvml,
    #[serde(rename = "nvidia-smi")]
    NvidiaSmi,
    Sysfs,
    #[serde(rename = "rocm-smi")]
    RocmSmi,
    None,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    pub idle_unload_secs: u64,
    pub queue_max: usize,
    pub queue_wait_secs: u64,
    pub failed_retry_secs: u64,
}
impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            idle_unload_secs: 1800,
            queue_max: 64,
            queue_wait_secs: 120,
            failed_retry_secs: 30,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HuggingFaceConfig {
    pub token_file: Option<PathBuf>,
    pub cache_dir: PathBuf,
    pub cache_limit_gb: f64,
    pub downloads: DownloadMode,
}
impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            token_file: None,
            cache_dir: "/var/lib/eliezer/models/hf".into(),
            cache_limit_gb: 200.0,
            downloads: DownloadMode::OnRequest,
        }
    }
}
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DownloadMode {
    #[default]
    OnRequest,
    Never,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    pub mistralrs: Option<EmbeddedBackendConfig>,
    pub llamacpp: Option<EmbeddedBackendConfig>,
    pub sdcpp: Option<SdcppConfig>,
}
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddedBackendConfig {
    pub enabled: bool,
    pub max_loaded: usize,
}
impl Default for EmbeddedBackendConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_loaded: 4,
        }
    }
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdcppConfig {
    pub program: PathBuf,
    pub port_range: [u16; 2],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub key: String,
    pub backend: BackendKind,
    pub source: ModelSource,
    pub kind: ModelKind,
    #[serde(default)]
    pub priority: i32,
    #[serde(default = "context_length")]
    pub max_context: u32,
    #[serde(default = "context_length")]
    pub default_context: u32,
    #[serde(default = "one")]
    pub parallel: u32,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub pinned: bool,
    /// `auto` (default), `gpu` or `cpu`; image models always use the GPU.
    #[serde(default)]
    pub device: DevicePolicy,
    pub reasoning: Option<ReasoningConfig>,
    pub arch: Option<Architecture>,
    pub size_gb: Option<f64>,
    pub vram_gb: Option<f64>,
    /// Memory of the vision projector in decimal GB; without it a flat 0.3 GB is assumed.
    pub projector_gb: Option<f64>,
    pub extra: Option<ExtraConfig>,
}
fn context_length() -> u32 {
    32768
}
fn one() -> u32 {
    1
}
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Mistralrs,
    Llamacpp,
    Sdcpp,
}
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Llm,
    Embedding,
    Image,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSource {
    pub hf: Option<String>,
    pub revision: Option<String>,
    pub file: Option<PathBuf>,
    /// The vision projector (`mmproj` GGUF file) of a llama.cpp model that reads images.
    pub projector: Option<PathBuf>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningConfig {
    pub options: Vec<String>,
    pub default: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Architecture {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub swa_layers: Option<u32>,
    pub window: Option<u32>,
    pub full_kv_heads: Option<u32>,
    pub full_head_dim: Option<u32>,
    pub linear_layers: Option<u32>,
    pub state_bytes: Option<u64>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtraConfig {
    pub text_encoder: Option<PathBuf>,
    pub vae: Option<PathBuf>,
}

/// The caller also resolves existing symbolic links within paths that do not exist yet.
pub trait PathResolver {
    fn resolve(&self, path: &Path) -> Result<PathBuf, PathResolutionError>;
}
#[derive(Debug)]
pub struct PathResolutionError;

#[derive(Debug)]
pub struct ConfigError {
    pub line: usize,
    pub field: String,
    pub message: &'static str,
}
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Configuration error in line {}, field {}: {}",
            self.line, self.field, self.message
        )
    }
}
impl std::error::Error for ConfigError {}

const FIELDS: &str = "cpu ram_budget_gb device server listen api_keys_file run_user run_group request_timeout_secs storage models_dir gpu vendor index vram_budget_gb reserve_gb measure scheduler idle_unload_secs queue_max queue_wait_secs failed_retry_secs huggingface token_file cache_dir cache_limit_gb downloads backend mistralrs llamacpp sdcpp enabled max_loaded program port_range model key source hf revision file kind priority max_context default_context parallel vision pinned reasoning options default arch layers kv_heads head_dim swa_layers window full_kv_heads full_head_dim linear_layers state_bytes size_gb vram_gb projector projector_gb extra text_encoder vae";

/// Which section each known field belongs to. Used only to say "this field is in another section";
/// names that are not in this table are never echoed.
const SECTION_FIELDS: &[(&str, &str)] = &[
    (
        "server",
        "listen api_keys_file run_user run_group request_timeout_secs",
    ),
    ("storage", "models_dir"),
    ("gpu", "vendor index vram_budget_gb reserve_gb measure"),
    ("cpu", "ram_budget_gb reserve_gb"),
    (
        "scheduler",
        "idle_unload_secs queue_max queue_wait_secs failed_retry_secs",
    ),
    (
        "huggingface",
        "token_file cache_dir cache_limit_gb downloads",
    ),
];

/// Adds a hint when a known field stands in the wrong section (`storage.cache_dir`).
fn with_section_hint(field: String) -> String {
    let Some((section, key)) = field.split_once('.') else {
        return field;
    };
    let owns = |name: &str| {
        SECTION_FIELDS
            .iter()
            .any(|(s, fields)| *s == name && fields.split_whitespace().any(|f| f == key))
    };
    if owns(section) {
        return field;
    }
    match SECTION_FIELDS
        .iter()
        .find(|(s, fields)| *s != section && fields.split_whitespace().any(|f| f == key))
    {
        Some((right, _)) if SECTION_FIELDS.iter().any(|(s, _)| *s == section) => {
            format!("{field} (a field of [{right}], not of [{section}])")
        }
        _ => field,
    }
}

fn safe_path(path: &serde_path_to_error::Path) -> String {
    let mut result = String::new();
    for segment in path {
        match segment {
            Segment::Map { key } if FIELDS.split_whitespace().any(|name| name == key) => {
                if !result.is_empty() {
                    result.push('.');
                }
                result.push_str(key);
            }
            Segment::Seq { index } => result.push_str(&format!("[{index}]")),
            _ => {
                result.push_str(".<unknown>");
                break;
            }
        }
    }
    if result.is_empty() {
        "<configuration>".into()
    } else {
        result
    }
}

impl ConfigError {
    fn at(input: &str, field: String, message: &'static str) -> Self {
        Self {
            line: field_line(input, &field),
            field,
            message,
        }
    }
}

/// Error position without echoing values; takes tables and model indices into account.
fn field_line(input: &str, field: &str) -> usize {
    let mut table = String::new();
    let mut model_index = 0_usize;
    let mut seen_model = false;
    let mut best = (0, 1);
    for (index, line) in input.lines().enumerate() {
        let line = line.trim();
        let header = line
            .split('#')
            .next()
            .unwrap_or("")
            .trim()
            .replace(['"', '\''], "");
        if header == "[[model]]" {
            if seen_model {
                model_index += 1;
            }
            seen_model = true;
            table = format!("model[{model_index}]");
        } else if line.starts_with('[') {
            let name = header
                .trim_start_matches('[')
                .split(']')
                .next()
                .unwrap_or("");
            table = if let Some(rest) = name.strip_prefix("model.") {
                format!("model[{model_index}].{rest}")
            } else {
                name.to_owned()
            };
        }
        let key = if line.starts_with('[') {
            table.clone()
        } else if let Some((key, _)) = line.split_once('=') {
            let key = key.trim();
            if table.is_empty() {
                key.to_owned()
            } else {
                format!("{table}.{key}")
            }
        } else {
            continue;
        };
        if (field == key
            || field.starts_with(&format!("{key}."))
            || field.starts_with(&format!("{key}[")))
            && key.len() >= best.0
        {
            best = (key.len(), index + 1);
        }
    }
    best.1
}

impl Config {
    /// Problems with the files and directories the configuration names, for warnings (the
    /// configuration itself is fine): a model store or model file that was moved, deleted or made
    /// unreadable. Not an error, because the files may come back and the rest keeps working.
    pub fn file_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let store = &self.storage.models_dir;
        match std::fs::metadata(store) {
            Err(error) => problems.push(format!(
                "storage.models_dir {}: {} (moved or unmounted? set storage.models_dir to the new place)",
                store.display(),
                describe(&error)
            )),
            Ok(meta) if !meta.is_dir() => problems.push(format!(
                "storage.models_dir {} is not a directory",
                store.display()
            )),
            Ok(_) => {}
        }
        for model in &self.model {
            let files = [
                (&model.source.file, "source.file"),
                (&model.source.projector, "source.projector"),
            ];
            for (file, field) in files {
                let Some(file) = file else {
                    continue;
                };
                match std::fs::File::open(file).and_then(|f| f.metadata()) {
                    Err(error) => problems.push(format!(
                        "model '{}': {} {} (moved or deleted? update {field}, or run `eliezer --import`)",
                        model.key,
                        file.display(),
                        describe(&error)
                    )),
                    Ok(meta) if !meta.is_file() => problems.push(format!(
                        "model '{}': {} is not a regular file",
                        model.key,
                        file.display()
                    )),
                    Ok(_) => {}
                }
            }
        }
        problems
    }

    /// Parses everything except the model list, so `eliezer --import` can run even when a
    /// `[[model]]` entry is broken (the import is what writes working entries). Returns the
    /// configuration without models and the text it was parsed from.
    pub fn parse_without_models(input: &str) -> Result<(Self, String), ConfigError> {
        let invalid = |message| ConfigError {
            line: 1,
            field: "<configuration>".into(),
            message,
        };
        let mut table: toml::Table =
            toml::from_str(input).map_err(|_| invalid("Invalid TOML syntax"))?;
        table.remove("model");
        let text = toml::to_string(&table).map_err(|_| invalid("Invalid TOML syntax"))?;
        Ok((Self::parse(&text)?, text))
    }

    pub fn parse(input: &str) -> Result<Self, ConfigError> {
        let de = toml::de::Deserializer::parse(input).map_err(|e| ConfigError {
            line: input.as_bytes()[..e.span().map_or(0, |s| s.start).min(input.len())]
                .iter()
                .filter(|b| **b == b'\n')
                .count()
                + 1,
            field: "<configuration>".into(),
            message: "Invalid TOML syntax",
        })?;
        let config: Self = serde_path_to_error::deserialize(de).map_err(|e| ConfigError {
            line: input.as_bytes()[..e.inner().span().map_or(0, |s| s.start).min(input.len())]
                .iter()
                .filter(|b| **b == b'\n')
                .count()
                + 1,
            field: with_section_hint(safe_path(e.path())),
            message: "Unknown field, missing value or invalid data type",
        })?;
        config.validate(input)?;
        Ok(config)
    }

    fn validate(&self, input: &str) -> Result<(), ConfigError> {
        let check = |ok: bool, field: &str, message| {
            if ok {
                Ok(())
            } else {
                Err(ConfigError::at(input, field.into(), message))
            }
        };
        check(
            self.server.listen.port() != 0,
            "server.listen",
            "Port must be greater than zero",
        )?;
        check(
            self.server.request_timeout_secs > 0,
            "server.request_timeout_secs",
            "Timeout must be greater than zero",
        )?;
        for (name, field) in [
            (&self.server.run_user, "server.run_user"),
            (&self.server.run_group, "server.run_group"),
        ] {
            if let Some(name) = name {
                check(
                    user_name(name),
                    field,
                    "Expected a system user or group name (letters, digits, '_', '-' and '.', starting with a letter or '_', at most 64 characters)",
                )?;
            }
        }
        check(
            positive(self.gpu.vram_budget_gb),
            "gpu.vram_budget_gb",
            "Budget must be finite and greater than zero",
        )?;
        check(
            self.gpu.reserve_gb.is_finite()
                && self.gpu.reserve_gb >= 0.0
                && self.gpu.reserve_gb < self.gpu.vram_budget_gb,
            "gpu.reserve_gb",
            "Reserve must be finite, non-negative and smaller than the budget",
        )?;
        check(
            self.cpu.ram_budget_gb.is_finite() && self.cpu.ram_budget_gb >= 0.0,
            "cpu.ram_budget_gb",
            "Budget must be finite and not negative (0 turns CPU placement off)",
        )?;
        check(
            self.cpu.reserve_gb.is_finite()
                && self.cpu.reserve_gb >= 0.0
                && (self.cpu.ram_budget_gb == 0.0 || self.cpu.reserve_gb < self.cpu.ram_budget_gb),
            "cpu.reserve_gb",
            "Reserve must be finite, not negative and smaller than the budget",
        )?;
        check(
            positive(self.huggingface.cache_limit_gb),
            "huggingface.cache_limit_gb",
            "Cache limit must be finite and greater than zero",
        )?;
        for (value, field) in [
            (self.scheduler.idle_unload_secs, "idle_unload_secs"),
            (self.scheduler.queue_wait_secs, "queue_wait_secs"),
            (self.scheduler.failed_retry_secs, "failed_retry_secs"),
        ] {
            check(
                value > 0,
                &format!("scheduler.{field}"),
                "Timeout must be greater than zero",
            )?;
        }
        check(
            self.scheduler.queue_max > 0,
            "scheduler.queue_max",
            "Queue must allow at least one entry",
        )?;
        for (backend, name) in [
            (&self.backend.mistralrs, "mistralrs"),
            (&self.backend.llamacpp, "llamacpp"),
        ] {
            if let Some(backend) = backend {
                check(
                    backend.max_loaded > 0,
                    &format!("backend.{name}.max_loaded"),
                    "Upper limit must be greater than zero",
                )?;
            }
        }
        if let Some(sd) = &self.backend.sdcpp {
            check(
                clean_absolute(&sd.program),
                "backend.sdcpp.program",
                "Absolute program path without parent components required",
            )?;
            check(
                sd.port_range[0] > 0 && sd.port_range[0] <= sd.port_range[1],
                "backend.sdcpp.port_range",
                "Port range must be ascending and start above zero",
            )?;
            check(
                !(sd.port_range[0]..=sd.port_range[1]).contains(&self.server.listen.port()),
                "backend.sdcpp.port_range",
                "Port range overlaps the service port",
            )?;
        }
        for (path, field) in [
            (Some(&self.storage.models_dir), "storage.models_dir"),
            (Some(&self.huggingface.cache_dir), "huggingface.cache_dir"),
            (self.server.api_keys_file.as_ref(), "server.api_keys_file"),
            (
                self.huggingface.token_file.as_ref(),
                "huggingface.token_file",
            ),
        ] {
            if let Some(path) = path {
                check(
                    clean_absolute(path),
                    field,
                    "Absolute path without parent components required",
                )?;
            }
        }
        let mut keys = BTreeSet::new();
        for (index, model) in self.model.iter().enumerate() {
            let prefix = format!("model[{index}]");
            let check_model = |ok, field, message| check(ok, &format!("{prefix}.{field}"), message);
            check_model(identifier(&model.key), "key", "Invalid model key")?;
            check_model(
                keys.insert(&model.key),
                "key",
                "Model key appears more than once",
            )?;
            check_model(
                model.source.file.is_some() != model.source.hf.is_some(),
                "source",
                "Exactly one source (file or hf) is required",
            )?;
            if let Some(hf) = &model.source.hf {
                check_model(
                    identifier(hf),
                    "source.hf",
                    "Invalid Hugging Face identifier",
                )?;
            }
            check_model(
                model
                    .source
                    .revision
                    .as_ref()
                    .is_none_or(|rev| model.source.hf.is_some() && identifier(rev)),
                "source.revision",
                "Revision is only allowed as a valid identifier of an HF source",
            )?;
            let enabled = match model.backend {
                BackendKind::Mistralrs => {
                    self.backend.mistralrs.as_ref().is_some_and(|b| b.enabled)
                }
                BackendKind::Llamacpp => self.backend.llamacpp.as_ref().is_some_and(|b| b.enabled),
                BackendKind::Sdcpp => self.backend.sdcpp.is_some(),
            };
            check_model(
                enabled,
                "backend",
                "Backend is not configured or is disabled",
            )?;
            check_model(
                model.max_context > 0
                    && model.default_context > 0
                    && model.default_context <= model.max_context,
                "default_context",
                "Context must be positive and must not exceed the upper limit",
            )?;
            check_model(
                model.parallel > 0 && model.parallel <= model.default_context,
                "parallel",
                "Parallelism must be positive and fit into the context",
            )?;
            for (value, field) in [
                (model.size_gb, "size_gb"),
                (model.vram_gb, "vram_gb"),
                (model.projector_gb, "projector_gb"),
            ] {
                check_model(
                    value.is_none_or(positive),
                    field,
                    "Memory size must be finite and greater than zero",
                )?;
            }
            check_model(
                model.source.projector.is_none()
                    || (model.source.file.is_some()
                        && model.kind == ModelKind::Llm
                        && model.backend == BackendKind::Llamacpp),
                "source.projector",
                "A vision projector belongs to a language model served by llamacpp from a file",
            )?;
            check_model(
                model.backend != BackendKind::Llamacpp
                    || model.vision == model.source.projector.is_some(),
                "vision",
                "A llamacpp model reads images exactly when it has source.projector (set both)",
            )?;
            check_model(
                model.kind != ModelKind::Image || model.vram_gb.is_some(),
                "vram_gb",
                "Image models require a measured memory requirement",
            )?;
            check_model(
                model.kind != ModelKind::Image || model.device != DevicePolicy::Cpu,
                "device",
                "Image models need the GPU; diffusion on the CPU is not supported",
            )?;
            check_model(
                (model.backend == BackendKind::Sdcpp) == (model.kind == ModelKind::Image),
                "kind",
                "Image models require sdcpp; sdcpp serves only image models",
            )?;
            if let Some(reasoning) = &model.reasoning {
                let mut options = BTreeSet::new();
                check_model(
                    !reasoning.options.is_empty()
                        && reasoning.options.iter().all(|o| {
                            ["none", "minimal", "low", "medium", "high", "xhigh"]
                                .contains(&o.as_str())
                                && options.insert(o)
                        })
                        && reasoning.options.contains(&reasoning.default),
                    "reasoning",
                    "Reasoning options must be unique and known and must include the default",
                )?;
            }
            if let Some(arch) = &model.arch {
                check_model(
                    arch.layers > 0 && arch.kv_heads > 0 && arch.head_dim > 0,
                    "arch",
                    "Architecture sizes must be positive",
                )?;
                let swa = arch.swa_layers.unwrap_or(0);
                let linear = arch.linear_layers.unwrap_or(0);
                check_model(
                    u64::from(swa) + u64::from(linear) <= u64::from(arch.layers),
                    "arch",
                    "Special layers exceed the total count",
                )?;
                check_model(
                    swa == 0 || arch.window.is_some_and(|v| v > 0),
                    "arch.window",
                    "Sliding-window layers require a positive window",
                )?;
                check_model(
                    linear == 0 || arch.state_bytes.is_some_and(|v| v > 0),
                    "arch.state_bytes",
                    "Linear layers require a positive state size",
                )?;
                check_model(
                    arch.full_kv_heads.is_none_or(|v| v > 0)
                        && arch.full_head_dim.is_none_or(|v| v > 0),
                    "arch",
                    "Optional head sizes must be positive",
                )?;
            }
            for (path, field) in model_paths(model) {
                check_model(
                    clean_absolute(path),
                    field,
                    "Absolute model path without parent components required",
                )?;
            }
        }
        Ok(())
    }

    pub fn validate_paths(
        &self,
        input: &str,
        resolver: &impl PathResolver,
    ) -> Result<(), ConfigError> {
        let resolve = |path: &Path, field: &str| {
            resolver
                .resolve(path)
                .map_err(|_| ConfigError::at(input, field.into(), "Path cannot be resolved safely"))
        };
        let root = resolve(&self.storage.models_dir, "storage.models_dir")?;
        let mut paths = vec![(
            &self.huggingface.cache_dir as &Path,
            "huggingface.cache_dir".to_owned(),
        )];
        for (index, model) in self.model.iter().enumerate() {
            paths.extend(
                model_paths(model)
                    .into_iter()
                    .map(|(path, field)| (path, format!("model[{index}].{field}"))),
            );
        }
        for (path, field) in paths {
            let resolved = resolve(path, &field)?;
            if !resolved.starts_with(&root) {
                return Err(ConfigError::at(
                    input,
                    field,
                    "Model path lies outside the central model store (a symbolic link that leads out of it is not followed): name the real directory in storage.models_dir, or move the file into the store (eliezer --import)",
                ));
            }
        }
        for (path, field) in [
            (self.server.api_keys_file.as_ref(), "server.api_keys_file"),
            (
                self.huggingface.token_file.as_ref(),
                "huggingface.token_file",
            ),
        ] {
            if let Some(path) = path
                && resolve(path, field)?.starts_with(&root)
            {
                return Err(ConfigError::at(
                    input,
                    field.into(),
                    "Secret files must lie outside the model store",
                ));
            }
        }
        Ok(())
    }
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}
fn describe(error: &std::io::Error) -> &'static str {
    match error.kind() {
        std::io::ErrorKind::NotFound => "not found",
        std::io::ErrorKind::PermissionDenied => "cannot be read (permission denied)",
        _ => "cannot be read",
    }
}

/// A Unix user or group name as found on local and domain-joined machines (`joerg.bernau`,
/// `svc_eliezer`). It is only ever looked up in /etc/passwd and /etc/group, never passed to a
/// shell or used as a path, so the rule just keeps out spaces, slashes and odd characters.
fn user_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        && value.len() <= 64
        && bytes.all(|c| c.is_ascii_alphanumeric() || b"_-.".contains(&c))
}
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 200
        && value.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        })
}
fn clean_absolute(path: &Path) -> bool {
    path.is_absolute()
        && !path.components().any(|c| matches!(c, Component::ParentDir))
        && !path
            .as_os_str()
            .as_encoded_bytes()
            .iter()
            .any(|b| b.is_ascii_control())
}
fn model_paths(model: &ModelConfig) -> Vec<(&Path, &'static str)> {
    let mut paths = Vec::new();
    if let Some(path) = &model.source.file {
        paths.push((path.as_path(), "source.file"));
    }
    if let Some(path) = &model.source.projector {
        paths.push((path.as_path(), "source.projector"));
    }
    if let Some(extra) = &model.extra {
        if let Some(path) = &extra.text_encoder {
            paths.push((path.as_path(), "extra.text_encoder"));
        }
        if let Some(path) = &extra.vae {
            paths.push((path.as_path(), "extra.vae"));
        }
    }
    paths
}

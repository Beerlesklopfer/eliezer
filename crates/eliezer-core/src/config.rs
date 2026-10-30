//! Strict configuration; file system access goes through a resolver supplied by the caller.
use serde::Deserialize;
use serde_path_to_error::Segment;
use std::{
    collections::{BTreeMap, BTreeSet},
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
    /// Allows the search of Hugging Face (`/api/v1/hub/...`); it only reads public metadata.
    pub search: bool,
    /// The repositories the service may download from: `Organisation/Repository` or `Organisation/*`.
    /// Empty means nothing can be downloaded, whatever `downloads` says.
    pub allow: Vec<String>,
}
impl HuggingFaceConfig {
    /// Whether a download from `repo` is on the allowlist. An entry is `Organisation/Repository`; in each of the two
    /// parts `*` stands for any run of characters (`Qwen/*`, `*/Qwen3-VL*`).
    pub fn allows(&self, repo: &str) -> bool {
        let Some((org, name)) = repo.split_once('/') else {
            return false;
        };
        self.allow.iter().any(|entry| {
            entry
                .split_once('/')
                .is_some_and(|(o, n)| wildcard(o, org) && wildcard(n, name))
        })
    }
}
/// Matches `text` against `pattern`, where `*` stands for any run of characters (also none).
pub fn wildcard(pattern: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}
impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            token_file: None,
            cache_dir: "/var/lib/eliezer/models/hf".into(),
            cache_limit_gb: 200.0,
            downloads: DownloadMode::OnRequest,
            search: true,
            allow: Vec::new(),
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
    pub audiocpp: Option<AudiocppConfig>,
    pub whispercpp: Option<WhispercppConfig>,
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
/// The audio.cpp server for music and speech synthesis models: one `audiocpp_server` child process per loaded model.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudiocppConfig {
    pub program: PathBuf,
    pub port_range: [u16; 2],
    #[serde(default)]
    pub library_path: Vec<PathBuf>,
    /// The compute backend of the program: `hip` (AMD, the default), `cuda`, `vulkan` or `cpu`.
    #[serde(default = "compute_default")]
    pub compute: String,
    /// Speech synthesis: the directory with the voice samples (`<name>.wav`, optionally `<name>.txt`, the words spoken
    /// in the sample). Default: `voices` inside `storage.models_dir`. Never taken from a request.
    pub voices_dir: Option<PathBuf>,
    /// Speech synthesis: the longest text of one request in characters; a model can set its own.
    #[serde(default = "max_input_chars_default")]
    pub max_input_chars: u32,
    /// Speech synthesis: a text is cut into passages of at most this many characters (at sentence boundaries) that are
    /// synthesized one after the other; a model can set its own.
    #[serde(default = "max_passage_chars_default")]
    pub max_passage_chars: u32,
    /// Speech synthesis: the pause between two passages of a paragraph (milliseconds); a model can set its own.
    #[serde(default = "pause_ms_default")]
    pub pause_ms: u32,
    /// Speech synthesis: the pause after the last passage of a paragraph (milliseconds); a model can set its own.
    #[serde(default = "paragraph_pause_ms_default")]
    pub paragraph_pause_ms: u32,
    /// Voice upload: the size limit of the uploaded file in MB.
    #[serde(default = "voice_max_file_mb_default")]
    pub voice_max_file_mb: u32,
    /// Voice upload: the length limit of the recording in seconds.
    #[serde(default = "voice_max_seconds_default")]
    pub voice_max_seconds: u32,
    /// Voice upload: the longest voice name in characters.
    #[serde(default = "voice_name_max_chars_default")]
    pub voice_name_max_chars: u32,
    /// Voice upload: the sample rate (Hz) the samples are stored at (mono, 16 bit).
    #[serde(default = "voice_sample_rate_default")]
    pub voice_sample_rate: u32,
    /// Voice upload: the largest number of voices in the directory.
    #[serde(default = "max_voices_default")]
    pub max_voices: u32,
}
fn max_input_chars_default() -> u32 {
    100_000
}
fn max_passage_chars_default() -> u32 {
    400
}
fn pause_ms_default() -> u32 {
    400
}
fn paragraph_pause_ms_default() -> u32 {
    800
}
fn voice_max_file_mb_default() -> u32 {
    20
}
fn voice_max_seconds_default() -> u32 {
    120
}
fn voice_name_max_chars_default() -> u32 {
    64
}
fn voice_sample_rate_default() -> u32 {
    24_000
}
fn max_voices_default() -> u32 {
    100
}
/// The whisper.cpp server for speech recognition: one `whisper-server` child process per loaded model.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhispercppConfig {
    pub program: PathBuf,
    pub port_range: [u16; 2],
    #[serde(default)]
    pub library_path: Vec<PathBuf>,
    /// Flash attention of the program (`--flash-attn` / `--no-flash-attn`); the program's own default without.
    pub flash_attn: Option<bool>,
    /// The GPU the program uses (`--device`); the program's default (0) without.
    pub gpu_device: Option<u32>,
    /// The length of one piece (seconds) a long recording is cut into before it goes to the program.
    #[serde(default = "chunk_secs_default")]
    pub chunk_secs: u32,
    /// How much two neighbouring pieces overlap (seconds), so no word is cut in two.
    #[serde(default = "overlap_secs_default")]
    pub overlap_secs: u32,
    /// The size limit of an uploaded file in MB; a model can set its own (`extra.max_file_mb`).
    #[serde(default = "max_file_mb_default")]
    pub max_file_mb: u32,
    /// The length limit of a recording in minutes; a model can set its own (`extra.max_audio_minutes`).
    #[serde(default = "max_audio_minutes_default")]
    pub max_audio_minutes: u32,
    /// Ask the program to break segments only between words (`split_on_word`); a model can set its own.
    #[serde(default = "split_on_word_default")]
    pub split_on_word: bool,
    /// Wrap segments after this many characters (`max_len`, 20 to 400); 0 leaves the program's own wrapping. A model can set its own.
    #[serde(default)]
    pub max_segment_chars: u32,
    /// How long a model may take to load (seconds).
    #[serde(default = "load_timeout_secs_default")]
    pub load_timeout_secs: u32,
    /// How long the program may take for one piece of a recording (seconds).
    #[serde(default = "request_timeout_secs_default")]
    pub request_timeout_secs: u32,
    /// How long the program gets to end after it is told to stop, before it is killed (seconds).
    #[serde(default = "stop_grace_secs_default")]
    pub stop_grace_secs: u32,
}
fn split_on_word_default() -> bool {
    true
}
fn load_timeout_secs_default() -> u32 {
    300
}
fn request_timeout_secs_default() -> u32 {
    3600
}
fn stop_grace_secs_default() -> u32 {
    15
}
fn chunk_secs_default() -> u32 {
    600
}
fn overlap_secs_default() -> u32 {
    5
}
fn max_file_mb_default() -> u32 {
    100
}
fn max_audio_minutes_default() -> u32 {
    120
}
fn compute_default() -> String {
    "hip".to_owned()
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdcppConfig {
    pub program: PathBuf,
    pub port_range: [u16; 2],
    /// Directories with the shared libraries the program needs (the bundled ROCm libraries, `libomp`); they
    /// are put in front of the library path of the child process.
    #[serde(default)]
    pub library_path: Vec<PathBuf>,
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
    /// The model is for text recognition in images (`capabilities.ocr` in the model list); needs `vision`.
    #[serde(default)]
    pub ocr: bool,
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
    Audiocpp,
    Whispercpp,
}
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Llm,
    Embedding,
    Image,
    /// Music from lyrics and a description (`POST /v1/audio/speech`); served by audio.cpp.
    Music,
    /// Speech recognition (`POST /v1/audio/transcriptions`); served by whisper.cpp.
    Stt,
    /// Speech synthesis with voice cloning (`POST /v1/audio/speech`); served by audio.cpp.
    Tts,
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
/// The sampling methods stable-diffusion.cpp offers (checked against `sd-server --help`).
pub const SAMPLING_METHODS: &[&str] = &[
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
    "dpm++2m_sde",
    "dpm++2m_sde_bt",
    "lms",
];
/// The component files of a music package that are used when `extra` names none (the balanced mix of audio.cpp).
pub const MUSIC_LANGUAGE_MODEL: &str = "language_model_q4_0.gguf";
pub const MUSIC_RVQ_DEPTH_DECODER: &str = "rvq_depth_decoder_q8_0.gguf";
pub const MUSIC_FLOW_TRANSFORMER: &str = "transformer_q4_0.gguf";
/// The model families of audio.cpp that `kind = "tts"` models may name in `extra.family` (the first is the default).
/// Both clone a voice from a reference recording.
pub const TTS_FAMILIES: [&str; 2] = ["chatterbox", "cosyvoice3"];
/// The numeric request options of a speech family that `extra.tts_options` may set (checked against the `model_specs`
/// of audio.cpp at the pinned commit, docs/Runtimes.md).
pub fn tts_option_names(family: &str) -> &'static [&'static str] {
    match family {
        "chatterbox" => &[
            "exaggeration",
            "guidance_scale",
            "temperature",
            "repetition_penalty",
            "min_p",
            "top_p",
            "flow_guidance_scale",
            "max_tokens",
            "text_chunk_size",
        ],
        "cosyvoice3" => &[
            "text_chunk_size",
            "max_tokens",
            "min_tokens",
            "top_k",
            "num_inference_steps",
        ],
        _ => &[],
    }
}
/// A voice name: letters and digits of ASCII, `-` and `_`, from 1 to `max_chars` characters. The name is a file name
/// stem, so nothing else may pass.
pub fn valid_voice_name(name: &str, max_chars: u32) -> bool {
    !name.is_empty()
        && name.len() <= max_chars as usize
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
/// The parts of a music package that every mix needs.
pub const MUSIC_FIXED_COMPONENTS: [&str; 2] = ["condition_encoder.gguf", "vocoder.gguf"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtraConfig {
    pub text_encoder: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    /// The `tokenizer.json` of a model whose text encoder has no tokenizer of its own in the program (MOSS-Image).
    pub tokenizer: Option<PathBuf>,
    /// Sampling steps of an image model (default of the program: 20).
    pub steps: Option<u32>,
    /// Guidance scale (default of the program: 7.0).
    pub cfg_scale: Option<f64>,
    /// One of [`SAMPLING_METHODS`] (default: the program's choice for the model).
    pub sampling_method: Option<String>,
    /// Keep the weights in RAM and move them to the GPU when needed (default: yes, which saves VRAM).
    pub offload_to_cpu: Option<bool>,
    /// The key of a language model that writes a detailed description from a short wish before the picture is drawn.
    pub prompt_helper: Option<String>,
    /// The model edits pictures: `POST /v1/images/edits` accepts it and `capabilities.image_edit` is reported.
    pub image_edit: Option<bool>,
    /// Music models: the component files of the package, by name inside the package directory (the defaults of the
    /// program are `language_model_q4_0.gguf`, `rvq_depth_decoder_q8_0.gguf` and `transformer_q4_0.gguf`).
    pub language_model: Option<String>,
    pub rvq_depth_decoder: Option<String>,
    pub flow_transformer: Option<String>,
    /// Music models: the length drawn when a request does not name one (seconds, 1 to 300; default 30).
    pub duration_sec: Option<u32>,
    /// Speech models: the language used when a request names none (`auto` detects it; default `auto`). Speech
    /// synthesis models: a language code (no `auto`); without one the program's own default (English) applies.
    pub language: Option<String>,
    /// Speech models: threads of the program (its own default without).
    pub threads: Option<u32>,
    /// Speech models: the size limit of an uploaded file in MB (default: `backend.whispercpp.max_file_mb`).
    pub max_file_mb: Option<u32>,
    /// Speech models: the length limit of a recording in minutes (default: `backend.whispercpp.max_audio_minutes`).
    pub max_audio_minutes: Option<u32>,
    /// Speech models: `split_on_word` of the program (default: `backend.whispercpp.split_on_word`).
    pub split_on_word: Option<bool>,
    /// Speech models: `max_len` of the program, 0 or 20 to 400 (default: `backend.whispercpp.max_segment_chars`).
    pub max_segment_chars: Option<u32>,
    /// Speech synthesis models: the audio.cpp family (one of [`TTS_FAMILIES`]; default the first).
    pub family: Option<String>,
    /// Speech synthesis models: the speaking rate when a request names none (0.25 to 4; the model must support it).
    pub speed: Option<f64>,
    /// Speech synthesis models: like `backend.audiocpp.max_input_chars`, `max_passage_chars`, `pause_ms` and
    /// `paragraph_pause_ms`, for this model.
    pub max_input_chars: Option<u32>,
    pub max_passage_chars: Option<u32>,
    pub pause_ms: Option<u32>,
    pub paragraph_pause_ms: Option<u32>,
    /// Speech synthesis models: the voice (a name in the voice directory) used when a request names none.
    pub default_voice: Option<String>,
    /// Speech synthesis models: the model has no speaker of its own, so a request without a voice (and without
    /// `default_voice`) is refused with 400 before anything is read (default: false).
    pub requires_voice: Option<bool>,
    /// Speech synthesis models: numeric request options of the family (see [`tts_option_names`]).
    pub tts_options: Option<BTreeMap<String, f64>>,
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

const FIELDS: &str = "cpu ram_budget_gb device server listen api_keys_file run_user run_group request_timeout_secs storage models_dir gpu vendor index vram_budget_gb reserve_gb measure scheduler idle_unload_secs queue_max queue_wait_secs failed_retry_secs huggingface token_file cache_dir cache_limit_gb downloads search allow backend mistralrs llamacpp sdcpp audiocpp whispercpp enabled max_loaded program port_range library_path compute model key source hf revision file kind priority max_context default_context parallel vision ocr pinned reasoning options default arch layers kv_heads head_dim swa_layers window full_kv_heads full_head_dim linear_layers state_bytes size_gb vram_gb projector projector_gb extra text_encoder vae tokenizer steps cfg_scale sampling_method offload_to_cpu prompt_helper image_edit language_model rvq_depth_decoder flow_transformer duration_sec language threads max_file_mb max_audio_minutes flash_attn gpu_device chunk_secs overlap_secs split_on_word max_segment_chars load_timeout_secs request_timeout_secs stop_grace_secs voices_dir max_input_chars max_passage_chars pause_ms paragraph_pause_ms voice_max_file_mb voice_max_seconds voice_name_max_chars voice_sample_rate max_voices family speed default_voice requires_voice tts_options";

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
        "token_file cache_dir cache_limit_gb downloads search allow",
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
    /// The directory with the voice samples: `backend.audiocpp.voices_dir`, else `voices` inside the model store.
    pub fn voices_dir(&self) -> PathBuf {
        self.backend
            .audiocpp
            .as_ref()
            .and_then(|a| a.voices_dir.clone())
            .unwrap_or_else(|| self.storage.models_dir.join("voices"))
    }

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
                // A music model is a package directory with several component files.
                let package = model.kind == ModelKind::Music && field == "source.file";
                match std::fs::File::open(file).and_then(|f| f.metadata()) {
                    Err(error) => problems.push(format!(
                        "model '{}': {} {} (moved or deleted? update {field}, or run `eliezer --import`)",
                        model.key,
                        file.display(),
                        describe(&error)
                    )),
                    Ok(meta) if package && !meta.is_dir() => problems.push(format!(
                        "model '{}': {} is not a directory (a music model names its package directory)",
                        model.key,
                        file.display()
                    )),
                    Ok(meta) if package && !file.join("config.json").is_file() => {
                        problems.push(format!(
                            "model '{}': {} has no config.json (is it a complete audio.cpp package?)",
                            model.key,
                            file.display()
                        ));
                        let _ = meta;
                    }
                    Ok(meta) if !package && !meta.is_file() => problems.push(format!(
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
        let config: Self = serde_path_to_error::deserialize(de).map_err(|e| {
            // A missing field is named (the name is a field of the schema, never a value of the file).
            let missing = e
                .inner()
                .message()
                .strip_prefix("missing field `")
                .and_then(|rest| rest.strip_suffix('`'))
                .filter(|name| FIELDS.split_whitespace().any(|f| f == *name))
                .map(str::to_owned);
            let path = safe_path(e.path());
            ConfigError {
                line: input.as_bytes()[..e.inner().span().map_or(0, |s| s.start).min(input.len())]
                    .iter()
                    .filter(|b| **b == b'\n')
                    .count()
                    + 1,
                field: match &missing {
                    Some(name) if path.is_empty() || path == "<configuration>" => name.clone(),
                    Some(name) => format!("{path}.{name}"),
                    None => with_section_hint(path),
                },
                message: if missing.is_some() {
                    "A required section or field is missing"
                } else {
                    "Unknown field, missing value or invalid data type"
                },
            }
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
        for entry in &self.huggingface.allow {
            let part_ok = |p: &str| {
                !p.is_empty()
                    && p.len() <= 96
                    && !p.starts_with('.')
                    && p.bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"._-*".contains(&c))
            };
            let ok = entry.split_once('/').is_some_and(|(org, repo)| {
                part_ok(org)
                    && part_ok(repo)
                    // `*/*` would allow every repository of Hugging Face.
                    && !(org.bytes().all(|c| c == b'*') && repo.bytes().all(|c| c == b'*'))
            });
            check(
                ok,
                "huggingface.allow",
                "Entries are `Organisation/Repository`; `*` is a wildcard in a part, but not in both parts only",
            )?;
        }
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
        let programs = [
            (
                "sdcpp",
                self.backend
                    .sdcpp
                    .as_ref()
                    .map(|c| (&c.program, &c.library_path, c.port_range)),
            ),
            (
                "audiocpp",
                self.backend
                    .audiocpp
                    .as_ref()
                    .map(|c| (&c.program, &c.library_path, c.port_range)),
            ),
            (
                "whispercpp",
                self.backend
                    .whispercpp
                    .as_ref()
                    .map(|c| (&c.program, &c.library_path, c.port_range)),
            ),
        ];
        for (name, program) in programs {
            let Some((program, library_path, port_range)) = program else {
                continue;
            };
            check(
                clean_absolute(program),
                &format!("backend.{name}.program"),
                "Absolute program path without parent components required",
            )?;
            for dir in library_path {
                check(
                    clean_absolute(dir) && !dir.as_os_str().as_encoded_bytes().contains(&b':'),
                    &format!("backend.{name}.library_path"),
                    "Absolute directories without parent components and without ':' required",
                )?;
            }
            check(
                port_range[0] > 0 && port_range[0] <= port_range[1],
                &format!("backend.{name}.port_range"),
                "Port range must be ascending and start above zero",
            )?;
            check(
                !(port_range[0]..=port_range[1]).contains(&self.server.listen.port()),
                &format!("backend.{name}.port_range"),
                "Port range overlaps the service port",
            )?;
        }
        if let Some(music) = &self.backend.audiocpp {
            check(
                ["hip", "cuda", "vulkan", "cpu"].contains(&music.compute.as_str()),
                "backend.audiocpp.compute",
                "The compute backend is hip, cuda, vulkan or cpu",
            )?;
        }
        if let Some(audio) = &self.backend.audiocpp {
            if let Some(dir) = &audio.voices_dir {
                check(
                    clean_absolute(dir),
                    "backend.audiocpp.voices_dir",
                    "Absolute directory without parent components required",
                )?;
            }
            check(
                (1..=400_000).contains(&audio.max_input_chars),
                "backend.audiocpp.max_input_chars",
                "The text limit is 1 to 400000 characters",
            )?;
            check(
                (40..=4000).contains(&audio.max_passage_chars),
                "backend.audiocpp.max_passage_chars",
                "A passage is 40 to 4000 characters long",
            )?;
            check(
                audio.pause_ms <= 10_000 && audio.paragraph_pause_ms <= 10_000,
                "backend.audiocpp.pause_ms",
                "A pause is at most 10000 milliseconds",
            )?;
            check(
                (1..=500).contains(&audio.voice_max_file_mb)
                    && (1..=600).contains(&audio.voice_max_seconds),
                "backend.audiocpp.voice_max_file_mb",
                "The voice upload limits are 1 to 500 MB and 1 to 600 seconds",
            )?;
            check(
                (1..=100).contains(&audio.voice_name_max_chars),
                "backend.audiocpp.voice_name_max_chars",
                "A voice name is 1 to 100 characters long",
            )?;
            check(
                (8000..=48_000).contains(&audio.voice_sample_rate),
                "backend.audiocpp.voice_sample_rate",
                "The sample rate of the voices is 8000 to 48000 Hz",
            )?;
            check(
                (1..=10_000).contains(&audio.max_voices),
                "backend.audiocpp.max_voices",
                "The number of voices is 1 to 10000",
            )?;
        }
        if let (Some(images), Some(music)) = (&self.backend.sdcpp, &self.backend.audiocpp) {
            check(
                images.port_range[1] < music.port_range[0]
                    || music.port_range[1] < images.port_range[0],
                "backend.audiocpp.port_range",
                "Port range overlaps the one of backend.sdcpp",
            )?;
        }
        if let Some(speech) = &self.backend.whispercpp {
            check(
                (30..=3600).contains(&speech.chunk_secs),
                "backend.whispercpp.chunk_secs",
                "A piece is between 30 and 3600 seconds long",
            )?;
            check(
                speech.overlap_secs <= 60 && speech.overlap_secs * 2 < speech.chunk_secs,
                "backend.whispercpp.overlap_secs",
                "The overlap is at most 60 seconds and less than half a piece",
            )?;
            check(
                speech.max_segment_chars == 0 || (20..=400).contains(&speech.max_segment_chars),
                "backend.whispercpp.max_segment_chars",
                "The segment length is 0 (the program's own) or 20 to 400 characters",
            )?;
            check(
                (1..=3600).contains(&speech.load_timeout_secs),
                "backend.whispercpp.load_timeout_secs",
                "The load time out is 1 to 3600 seconds",
            )?;
            check(
                (1..=86_400).contains(&speech.request_timeout_secs),
                "backend.whispercpp.request_timeout_secs",
                "The request time out is 1 to 86400 seconds",
            )?;
            check(
                (1..=300).contains(&speech.stop_grace_secs),
                "backend.whispercpp.stop_grace_secs",
                "The stop grace time is 1 to 300 seconds",
            )?;
            check(
                speech.max_file_mb > 0 && speech.max_audio_minutes > 0,
                "backend.whispercpp.max_file_mb",
                "The limits must be greater than zero",
            )?;
        }
        let mut taken: Vec<(&str, [u16; 2])> = Vec::new();
        for (name, range) in [
            ("sdcpp", self.backend.sdcpp.as_ref().map(|c| c.port_range)),
            (
                "audiocpp",
                self.backend.audiocpp.as_ref().map(|c| c.port_range),
            ),
            (
                "whispercpp",
                self.backend.whispercpp.as_ref().map(|c| c.port_range),
            ),
        ] {
            let Some(range) = range else { continue };
            // The pair sdcpp/audiocpp is checked above; every other pair here.
            if name == "whispercpp" {
                for (other, taken_range) in &taken {
                    check(
                        range[1] < taken_range[0] || taken_range[1] < range[0],
                        "backend.whispercpp.port_range",
                        if *other == "sdcpp" {
                            "Port range overlaps the one of backend.sdcpp"
                        } else {
                            "Port range overlaps the one of backend.audiocpp"
                        },
                    )?;
                }
            }
            taken.push((name, range));
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
        for (index, model) in self.model.iter().enumerate() {
            if model.extra.as_ref().and_then(|e| e.image_edit) == Some(true) {
                check(
                    model.kind == ModelKind::Image,
                    &format!("model[{index}].extra.image_edit"),
                    "Only an image model can edit pictures",
                )?;
            }
            if let Some(helper) = model.extra.as_ref().and_then(|e| e.prompt_helper.as_ref()) {
                let named = self.model.iter().find(|m| m.key == *helper);
                check(
                    model.kind == ModelKind::Image
                        && named.is_some_and(|m| m.kind == ModelKind::Llm),
                    &format!("model[{index}].extra.prompt_helper"),
                    "A prompt helper belongs to an image model and names a configured language model",
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
            check_model(
                !model.ocr || model.vision,
                "ocr",
                "A text recognition model reads images: it needs vision = true (and a projector)",
            )?;
            if let Some(extra) = &model.extra {
                check_model(
                    extra.steps.is_none_or(|s| (1..=150).contains(&s)),
                    "extra.steps",
                    "Steps are a number from 1 to 150",
                )?;
                check_model(
                    extra
                        .cfg_scale
                        .is_none_or(|c| c.is_finite() && (0.0..=30.0).contains(&c)),
                    "extra.cfg_scale",
                    "The guidance scale is a number from 0 to 30",
                )?;
                check_model(
                    extra
                        .sampling_method
                        .as_ref()
                        .is_none_or(|m| SAMPLING_METHODS.contains(&m.as_str())),
                    "extra.sampling_method",
                    "Unknown sampling method",
                )?;
            }
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
                BackendKind::Audiocpp => self.backend.audiocpp.is_some(),
                BackendKind::Whispercpp => self.backend.whispercpp.is_some(),
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
            let generation = matches!(
                model.kind,
                ModelKind::Image | ModelKind::Music | ModelKind::Tts
            );
            // Speech recognition needs a measured size too (it is placed by it), but may run on the CPU.
            check_model(
                (!generation && model.kind != ModelKind::Stt) || model.vram_gb.is_some(),
                "vram_gb",
                "Image, music and speech models require a measured memory requirement",
            )?;
            check_model(
                !generation || model.device != DevicePolicy::Cpu,
                "device",
                "Image, music and speech synthesis models need the GPU; generation on the CPU is not supported",
            )?;
            check_model(
                (model.backend == BackendKind::Sdcpp) == (model.kind == ModelKind::Image),
                "kind",
                "Image models require sdcpp; sdcpp serves only image models",
            )?;
            check_model(
                (model.backend == BackendKind::Audiocpp)
                    == matches!(model.kind, ModelKind::Music | ModelKind::Tts),
                "kind",
                "Music and speech synthesis models require audiocpp; audiocpp serves only these",
            )?;
            check_model(
                (model.backend == BackendKind::Whispercpp) == (model.kind == ModelKind::Stt),
                "kind",
                "Speech models require whispercpp; whispercpp serves only speech models",
            )?;
            check_model(
                model.kind != ModelKind::Stt || model.source.file.is_some(),
                "source.file",
                "A speech model names its ggml file in source.file (download it with `eliezer download`)",
            )?;
            check_model(
                model.kind != ModelKind::Tts || model.source.file.is_some(),
                "source.file",
                "A speech synthesis model names its GGUF file in source.file (download it with `eliezer download`)",
            )?;
            if let Some(extra) = &model.extra {
                let synthesis = extra.family.is_some()
                    || extra.speed.is_some()
                    || extra.max_input_chars.is_some()
                    || extra.max_passage_chars.is_some()
                    || extra.pause_ms.is_some()
                    || extra.paragraph_pause_ms.is_some()
                    || extra.default_voice.is_some()
                    || extra.requires_voice.is_some()
                    || extra.tts_options.is_some();
                check_model(
                    model.kind == ModelKind::Tts || !synthesis,
                    "extra",
                    "family, speed, the passage and pause settings, default_voice and tts_options belong to speech synthesis models",
                )?;
                if model.kind == ModelKind::Tts {
                    let family = extra.family.as_deref().unwrap_or(TTS_FAMILIES[0]);
                    check_model(
                        TTS_FAMILIES.contains(&family),
                        "extra.family",
                        "Unknown speech synthesis family (chatterbox or cosyvoice3)",
                    )?;
                    check_model(
                        extra.language.as_ref().is_none_or(|l| l != "auto"),
                        "extra.language",
                        "A speech synthesis model needs a language code such as de, not auto",
                    )?;
                    check_model(
                        extra.speed.is_none_or(|s| (0.25..=4.0).contains(&s)),
                        "extra.speed",
                        "The speed is between 0.25 and 4",
                    )?;
                    check_model(
                        extra
                            .max_input_chars
                            .is_none_or(|n| (1..=400_000).contains(&n))
                            && extra
                                .max_passage_chars
                                .is_none_or(|n| (40..=4000).contains(&n))
                            && extra.pause_ms.is_none_or(|n| n <= 10_000)
                            && extra.paragraph_pause_ms.is_none_or(|n| n <= 10_000),
                        "extra",
                        "Text limit 1 to 400000, passage 40 to 4000 characters, pauses at most 10000 milliseconds",
                    )?;
                    check_model(
                        extra
                            .default_voice
                            .as_ref()
                            .is_none_or(|v| valid_voice_name(v, 100)),
                        "extra.default_voice",
                        "A voice name has letters, digits, - and _ only",
                    )?;
                    let allowed = tts_option_names(family);
                    check_model(
                        extra.tts_options.as_ref().is_none_or(|o| {
                            o.iter()
                                .all(|(k, v)| allowed.contains(&k.as_str()) && v.is_finite())
                        }),
                        "extra.tts_options",
                        "Unknown or non-numeric option of the speech synthesis family",
                    )?;
                }
                let speech = (extra.language.is_some() && model.kind != ModelKind::Tts)
                    || extra.threads.is_some()
                    || extra.max_file_mb.is_some()
                    || extra.max_audio_minutes.is_some()
                    || extra.split_on_word.is_some()
                    || extra.max_segment_chars.is_some();
                check_model(
                    model.kind == ModelKind::Stt || !speech,
                    "extra",
                    "language, threads, the limits, split_on_word and max_segment_chars belong to speech models",
                )?;
                check_model(
                    extra.language.as_ref().is_none_or(|l| {
                        l == "auto"
                            || ((2..=3).contains(&l.len())
                                && l.bytes().all(|b| b.is_ascii_lowercase()))
                    }),
                    "extra.language",
                    "The language is auto or a lowercase language code such as de or en",
                )?;
                check_model(
                    extra.threads.is_none_or(|t| (1..=256).contains(&t))
                        && extra.max_file_mb.is_none_or(|m| m > 0)
                        && extra.max_audio_minutes.is_none_or(|m| m > 0),
                    "extra",
                    "threads are 1 to 256, the limits are greater than zero",
                )?;
                check_model(
                    extra
                        .max_segment_chars
                        .is_none_or(|c| c == 0 || (20..=400).contains(&c)),
                    "extra.max_segment_chars",
                    "The segment length is 0 (the program's own) or 20 to 400 characters",
                )?;
                let component = |name: &Option<String>| {
                    name.as_ref().is_none_or(|n| {
                        n.ends_with(".gguf")
                            && n.len() > 5
                            && n.bytes()
                                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                            && !n.starts_with('.')
                    })
                };
                check_model(
                    component(&extra.language_model)
                        && component(&extra.rvq_depth_decoder)
                        && component(&extra.flow_transformer),
                    "extra",
                    "Music components are file names of .gguf files inside the package directory",
                )?;
                check_model(
                    model.kind == ModelKind::Music
                        || (extra.language_model.is_none()
                            && extra.rvq_depth_decoder.is_none()
                            && extra.flow_transformer.is_none()
                            && extra.duration_sec.is_none()),
                    "extra",
                    "language_model, rvq_depth_decoder, flow_transformer and duration_sec belong to music models",
                )?;
                check_model(
                    extra.duration_sec.is_none_or(|d| (1..=300).contains(&d)),
                    "extra.duration_sec",
                    "The duration is between 1 and 300 seconds",
                )?;
            }
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
        if let Some(path) = &extra.tokenizer {
            paths.push((path.as_path(), "extra.tokenizer"));
        }
    }
    paths
}

//! Judging a repository: which weight files it offers, whether a runtime can serve them and whether
//! they fit the memory budgets. The figures are estimates from file sizes; the exact calculation
//! needs the GGUF header and happens at import (`eliezer --suggest`).

use crate::{Detail, RepoFile};
use std::collections::BTreeMap;

/// The memory the service may use, per pool, after the reserve (bytes). 0 means the pool is off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Budgets {
    pub gpu: u64,
    pub ram: u64,
}

/// Whether a runtime of Eliezer can serve the architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Served and checked on the reference card (see docs/Status.md).
    Verified,
    /// A runtime lists the architecture, but it has not been checked here.
    Expected,
    /// The architecture is not known to any runtime.
    Unsupported,
    /// The hub did not report an architecture.
    Unknown,
}
impl Support {
    pub fn name(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Expected => "expected",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Fits the GPU budget.
    Gpu,
    /// Does not fit the GPU, but the RAM budget (CPU placement).
    Cpu,
    TooLarge,
}
impl Fit {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
            Self::TooLarge => "too_large",
        }
    }
}

/// One downloadable variant: the weight file (all parts of a split file together).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    pub name: String,
    pub quantization: Option<&'static str>,
    pub bytes: u64,
    pub parts: usize,
    /// File size plus 20 % for the cache and the runtime, plus the projector (a language model) or the
    /// smallest text encoder and VAE (an image model).
    pub estimate_bytes: u64,
    pub fit: Fit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assessment {
    pub support: Support,
    /// The runtime that would serve it (`llamacpp`, `sdcpp`), when one is known.
    pub backend: Option<&'static str>,
    pub variants: Vec<Variant>,
    /// The vision projector of a language model.
    pub projector: Option<RepoFile>,
    /// Further files a model needs (text encoders and VAE of image models).
    pub required: Vec<RepoFile>,
}

const QUANTIZATIONS: &[&str] = &[
    "IQ2_XXS", "IQ3_XXS", "IQ1_S", "IQ1_M", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S",
    "Q5_K_M", "Q6_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "MXFP4", "NVFP4", "BF16", "F16",
    "F32", "FP8", "INT8",
];

/// The quantization named in a file name (`...-Q4_K_M.gguf` → `Q4_K_M`).
pub fn quantization(file: &str) -> Option<&'static str> {
    let upper = file.to_ascii_uppercase();
    QUANTIZATIONS.iter().copied().find(|q| {
        upper.match_indices(q).any(|(at, _)| {
            let before = upper[..at].chars().next_back();
            let after = upper[at + q.len()..].chars().next();
            let edge = |c: Option<char>| c.is_none_or(|c| !c.is_ascii_alphanumeric());
            edge(before) && edge(after)
        })
    })
}

/// Architectures that were served and checked with llama.cpp on the reference card, and image
/// architectures checked with stable-diffusion.cpp.
const VERIFIED_LLAMACPP: &[&str] = &[
    "llama", "qwen3moe", "qwen35", "gemma4", "gpt-oss", "qwen2vl",
];
/// Architectures llama.cpp lists; not checked here.
const EXPECTED_LLAMACPP: &[&str] = &[
    "qwen2",
    "qwen2moe",
    "qwen3",
    "qwen3vl",
    "gemma",
    "gemma2",
    "gemma3",
    "mistral3",
    "phi3",
    "phi4",
    "falcon",
    "granite",
    "olmo",
    "olmo2",
    "deepseek2",
    "bert",
    "nomic-bert",
    "command-r",
    "starcoder2",
    "smollm3",
    "glm4",
    "exaone",
];
const VERIFIED_SDCPP: &[&str] = &["qwen_image21"];
const EXPECTED_SDCPP: &[&str] = &["qwen_image", "flux", "sd3", "sdxl", "sd1"];

fn support(architecture: Option<&str>) -> (Support, Option<&'static str>) {
    let Some(architecture) = architecture else {
        return (Support::Unknown, None);
    };
    if VERIFIED_LLAMACPP.contains(&architecture) {
        (Support::Verified, Some("llamacpp"))
    } else if VERIFIED_SDCPP.contains(&architecture) {
        (Support::Verified, Some("sdcpp"))
    } else if EXPECTED_LLAMACPP.contains(&architecture) {
        (Support::Expected, Some("llamacpp"))
    } else if EXPECTED_SDCPP.contains(&architecture) {
        (Support::Expected, Some("sdcpp"))
    } else {
        (Support::Unsupported, None)
    }
}

/// `model-00001-of-00003.gguf` → `model.gguf`: the name all parts of a split file share.
fn group_name(file: &str) -> String {
    let Some(stem) = file.strip_suffix(".gguf") else {
        return file.to_owned();
    };
    if let Some(at) = stem.rfind("-of-")
        && stem[at + 4..].bytes().all(|c| c.is_ascii_digit())
        && let Some(start) = stem[..at].rfind('-')
        && stem[start + 1..at].bytes().all(|c| c.is_ascii_digit())
    {
        return format!("{}.gguf", &stem[..start]);
    }
    file.to_owned()
}

pub fn assess(detail: &Detail, budgets: Budgets) -> Assessment {
    let (support, backend) = support(detail.architecture.as_deref());
    let is_projector = |f: &&RepoFile| {
        let base = f.name.rsplit('/').next().unwrap_or(&f.name);
        base.starts_with("mmproj") && base.ends_with(".gguf")
    };
    let projector = detail
        .files
        .iter()
        .filter(is_projector)
        .max_by_key(|f| f.bytes)
        .cloned();
    let required: Vec<RepoFile> = detail
        .files
        .iter()
        .filter(|f| f.name.starts_with("text_encoders/") || f.name.starts_with("vae/"))
        .cloned()
        .collect();
    let mut groups: BTreeMap<String, (u64, usize)> = BTreeMap::new();
    for file in detail
        .files
        .iter()
        .filter(|f| f.name.ends_with(".gguf") && !f.name.contains('/') && !is_projector(f))
    {
        let entry = groups.entry(group_name(&file.name)).or_default();
        entry.0 = entry.0.saturating_add(file.bytes);
        entry.1 += 1;
    }
    // An image model needs its text encoder and VAE as well; the smallest of each counts, which
    // is conservative (stable-diffusion.cpp can offload the encoder).
    let smallest = |prefix: &str| {
        required
            .iter()
            .filter(|f| f.name.starts_with(prefix))
            .map(|f| f.bytes)
            .min()
            .unwrap_or(0)
    };
    let extra = projector.as_ref().map_or(0, |p| p.bytes)
        + if backend == Some("sdcpp") {
            smallest("text_encoders/") + smallest("vae/")
        } else {
            0
        };
    let variants = groups
        .into_iter()
        .map(|(name, (bytes, parts))| {
            let estimate_bytes = bytes.saturating_add(bytes / 5).saturating_add(extra);
            let fit = if estimate_bytes <= budgets.gpu {
                Fit::Gpu
            } else if estimate_bytes <= budgets.ram {
                Fit::Cpu
            } else {
                Fit::TooLarge
            };
            Variant {
                quantization: quantization(&name),
                name,
                bytes,
                parts,
                estimate_bytes,
                fit,
            }
        })
        .collect();
    Assessment {
        support,
        backend,
        variants,
        projector,
        required,
    }
}

//! Search of Hugging Face for model candidates (Specification 9.1).
//!
//! The module only looks: it lists repositories and their files and judges them against the
//! runtimes and the memory budgets. Nothing is downloaded, nothing is added to the allowlist.
//! The service never sends a token here; public metadata needs none.

use std::{future::Future, pin::Pin};

mod assess;
#[cfg(feature = "http")]
mod client;
mod fake;

pub use assess::{Assessment, Budgets, Fit, Support, Variant, assess, quantization};
#[cfg(feature = "http")]
pub use client::HfClient;
pub use fake::FakeHub;

/// What a model is for. Maps to the pipeline tag of Hugging Face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    #[default]
    Any,
    /// Text generation (chat).
    Llm,
    /// Chat with image input.
    Vision,
    Embedding,
    /// Text to image.
    Image,
}
impl Task {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "any" => Self::Any,
            "llm" => Self::Llm,
            "vision" => Self::Vision,
            "embedding" => Self::Embedding,
            "image" => Self::Image,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Llm => "llm",
            Self::Vision => "vision",
            Self::Embedding => "embedding",
            Self::Image => "image",
        }
    }
    /// The `pipeline_tag` values Hugging Face uses for this task. Embedding models carry either of
    /// two (the official nomic-embed-text GGUF is a `sentence-similarity` model); `Any` has none.
    pub fn pipeline_tags(self) -> &'static [&'static str] {
        match self {
            Self::Any => &[],
            Self::Llm => &["text-generation"],
            Self::Vision => &["image-text-to-text"],
            Self::Embedding => &["feature-extraction", "sentence-similarity"],
            Self::Image => &["text-to-image"],
        }
    }
}

/// The file format of the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Gguf,
    Safetensors,
}
impl Format {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "gguf" => Some(Self::Gguf),
            "safetensors" => Some(Self::Safetensors),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub text: String,
    pub task: Task,
    pub format: Format,
    /// A license tag as Hugging Face writes it (`apache-2.0`, `mit`, ...).
    pub license: Option<String>,
    pub limit: usize,
}

/// One search hit: what the list of Hugging Face knows without opening the repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub repo: String,
    pub downloads: u64,
    pub likes: u64,
    pub task: Option<String>,
    pub license: Option<String>,
    pub gguf: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub name: String,
    pub bytes: u64,
}

/// A repository with its files, as the detail page of Hugging Face lists them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Detail {
    pub repo: String,
    /// The commit of the listing; a download would pin it.
    pub sha: String,
    pub task: Option<String>,
    pub license: Option<String>,
    pub gated: bool,
    pub downloads: u64,
    /// The architecture named in the GGUF metadata, when the hub has read it.
    pub architecture: Option<String>,
    pub files: Vec<RepoFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubError {
    /// The hub cannot be reached or did not answer in time.
    Unreachable,
    NotFound,
    /// The answer is not what the interface promises.
    BadResponse,
    /// Rate limited by the hub.
    Busy,
}
impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unreachable => "Hugging Face cannot be reached",
            Self::NotFound => "The repository was not found on Hugging Face",
            Self::BadResponse => "Hugging Face answered in an unexpected form",
            Self::Busy => "Hugging Face is limiting requests; try again later",
        })
    }
}
impl std::error::Error for HubError {}

pub type HubFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, HubError>> + Send + 'a>>;

/// The source of candidates. The real one talks to huggingface.co, the test bench answers from memory.
pub trait Hub: Send + Sync {
    fn search<'a>(&'a self, query: &'a Query) -> HubFuture<'a, Vec<Summary>>;
    fn detail<'a>(&'a self, repo: &'a str) -> HubFuture<'a, Detail>;
}

/// `Organisation/Repository` with the characters Hugging Face allows. Anything else never reaches a URL.
pub fn valid_repo(repo: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && p.len() <= 96
            && p.bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
            && !p.starts_with('.')
    };
    repo.split_once('/')
        .is_some_and(|(org, name)| part(org) && part(name) && !name.contains('/'))
}

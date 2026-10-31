//! Search of Hugging Face for model candidates (Specification 9.1).
//!
//! The module only looks: it lists repositories and their files and judges them against the
//! runtimes and the memory budgets. Nothing is downloaded, nothing is added to the allowlist.
//! The service never sends a token here; public metadata needs none.

use std::{future::Future, pin::Pin};

mod assess;
#[cfg(feature = "http")]
mod client;
pub mod download;
mod fake;
pub mod store;

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
    /// The SHA-256 of the content, as Hugging Face lists it for files stored with LFS (all large files).
    pub sha256: Option<String>,
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
    /// The hub refuses access: the repository is gated (a license must be accepted) or private.
    Forbidden,
}
impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unreachable => "Hugging Face cannot be reached",
            Self::NotFound => "The repository was not found on Hugging Face",
            Self::BadResponse => "Hugging Face answered in an unexpected form",
            Self::Busy => "Hugging Face is limiting requests; try again later",
            Self::Forbidden => {
                "Hugging Face refuses access: accept the license of the repository on huggingface.co and provide a token"
            }
        })
    }
}
impl std::error::Error for HubError {}

pub type HubFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, HubError>> + Send + 'a>>;

/// An access token for Hugging Face. It never prints: `Debug` shows a placeholder, and the only way to
/// the text is [`Token::expose`], used where the request header is made.
#[derive(Clone)]
pub struct Token(String);
impl Token {
    /// `None` for an empty text or one with characters a header cannot carry.
    pub fn new(text: &str) -> Option<Self> {
        let text = text.trim();
        (!text.is_empty() && text.bytes().all(|c| c.is_ascii_graphic()))
            .then(|| Self(text.to_owned()))
    }
    pub fn expose(&self) -> &str {
        &self.0
    }
}
impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Token(<hidden>)")
    }
}

/// The environment a child process gets for Hugging Face: the token only as `HF_TOKEN`, never as an
/// argument (arguments show up in `ps`).
pub fn child_environment(token: Option<&Token>) -> Vec<(&'static str, String)> {
    token
        .map(|t| vec![("HF_TOKEN", t.expose().to_owned())])
        .unwrap_or_default()
}

/// The content of one file, delivered in pieces.
pub trait ChunkSource: Send {
    /// The next piece; `None` at the end.
    fn next_chunk(&mut self) -> HubFuture<'_, Option<Vec<u8>>>;
}

/// An opened download.
pub struct FileStream {
    /// False when the hub ignored the requested start and sends the file from its beginning.
    pub resumed: bool,
    /// The bytes still to come, when the hub says.
    pub remaining: Option<u64>,
    pub body: Box<dyn ChunkSource>,
}

/// The source of candidates. The real one talks to huggingface.co, the test bench answers from memory.
pub trait Hub: Send + Sync {
    fn search<'a>(&'a self, query: &'a Query) -> HubFuture<'a, Vec<Summary>>;
    fn detail<'a>(&'a self, repo: &'a str) -> HubFuture<'a, Detail>;
    /// The listing of one commit. The default only knows the current one.
    fn detail_at<'a>(&'a self, repo: &'a str, revision: &'a str) -> HubFuture<'a, Detail> {
        Box::pin(async move {
            let detail = self.detail(repo).await?;
            if detail.sha == revision {
                Ok(detail)
            } else {
                Err(HubError::NotFound)
            }
        })
    }
    /// Opens one file of one commit from byte `offset` on. Public repositories need no token.
    fn open<'a>(
        &'a self,
        repo: &'a str,
        revision: &'a str,
        file: &'a str,
        offset: u64,
        token: Option<&'a Token>,
    ) -> HubFuture<'a, FileStream>;
}

/// A file name inside a repository: relative, at most three levels, no `..`, nothing hidden, only the
/// characters Hugging Face file names use in practice. Anything else never reaches a path or a URL.
pub fn valid_file(name: &str) -> bool {
    let segments: Vec<&str> = name.split('/').collect();
    name.len() <= 200
        && (1..=3).contains(&segments.len())
        && segments.iter().all(|s| {
            !s.is_empty()
                && !s.starts_with('.')
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-+".contains(&c))
        })
}

/// A full commit hash.
pub fn valid_revision(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
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

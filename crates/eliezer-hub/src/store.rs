//! The model store as the downloads see it: where a file of a repository goes, the pin file that
//! records what was fetched (commit, sizes, hashes), and the check of the files against it.
//!
//! Layout: `<models_dir>/<Organisation>/<Repository>/<file>`, the same as an import of a Hugging Face
//! download. Each repository directory holds `.eliezer-pin.json`.

use crate::{valid_file, valid_repo};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    io::Read,
    path::{Path, PathBuf},
};

pub const PIN_FILE: &str = ".eliezer-pin.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: Option<String>,
    /// The commit this file was fetched from.
    pub revision: String,
    pub downloaded_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub repo: String,
    pub files: Vec<PinFile>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StoreError {
    /// A name that is not a repository or file name, or a path that leaves the store.
    Invalid,
    Io(String),
}
impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid => f.write_str("The path is not inside the model store"),
            Self::Io(reason) => write!(f, "{reason}"),
        }
    }
}
impl From<io::Error> for StoreError {
    fn from(error: io::Error) -> Self {
        // The text names the kind of failure, never a path.
        Self::Io(match error.kind() {
            io::ErrorKind::PermissionDenied => {
                "The model store is not writable for the service (a store under /home needs the systemd drop-in with ReadWritePaths)".to_owned()
            }
            io::ErrorKind::StorageFull => "The disk is full".to_owned(),
            _ => format!("File system error: {}", error.kind()),
        })
    }
}

/// `<root>/<Organisation>/<Repository>`
pub fn repo_dir(root: &Path, repo: &str) -> Result<PathBuf, StoreError> {
    if !valid_repo(repo) {
        return Err(StoreError::Invalid);
    }
    Ok(root.join(repo))
}

/// Where a file of a repository goes. The directories are created; the result is checked to lie inside
/// the store after resolving symbolic links, and an existing symbolic link as the target is refused.
pub fn target_path(root: &Path, repo: &str, file: &str) -> Result<PathBuf, StoreError> {
    if !valid_file(file) {
        return Err(StoreError::Invalid);
    }
    let target = repo_dir(root, repo)?.join(file);
    let parent = target.parent().ok_or(StoreError::Invalid)?;
    fs::create_dir_all(parent)?;
    let inside = parent
        .canonicalize()?
        .starts_with(root.canonicalize().map_err(StoreError::from)?);
    if !inside {
        return Err(StoreError::Invalid);
    }
    if fs::symlink_metadata(&target).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(StoreError::Invalid);
    }
    Ok(target)
}

pub fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// Hashes a file in 1 MiB pieces; blocking.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(sha256_hex(hasher.finalize()))
}

pub fn read_pin(dir: &Path, repo: &str) -> Option<Pin> {
    let value: Value = serde_json::from_slice(&fs::read(dir.join(PIN_FILE)).ok()?).ok()?;
    let files = value["files"]
        .as_array()?
        .iter()
        .filter_map(|f| {
            let name = f["name"].as_str().filter(|n| valid_file(n))?.to_owned();
            Some(PinFile {
                name,
                bytes: f["bytes"].as_u64()?,
                sha256: f["sha256"].as_str().map(str::to_owned),
                revision: f["revision"].as_str()?.to_owned(),
                downloaded_at: f["downloaded_at"].as_u64().unwrap_or(0),
            })
        })
        .collect();
    Some(Pin {
        repo: repo.to_owned(),
        files,
    })
}

/// Writes the pin file atomically (a temporary file, then a rename).
pub fn write_pin(dir: &Path, pin: &Pin) -> io::Result<()> {
    let value = json!({
        "repo": pin.repo,
        "files": pin.files.iter().map(|f| json!({
            "name": f.name, "bytes": f.bytes, "sha256": f.sha256,
            "revision": f.revision, "downloaded_at": f.downloaded_at,
        })).collect::<Vec<_>>(),
    });
    let temporary = dir.join(format!("{PIN_FILE}.tmp"));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&value).unwrap_or_default(),
    )?;
    fs::rename(temporary, dir.join(PIN_FILE))
}

/// Adds the files to the pin of a repository, replacing entries of the same name.
pub fn record(root: &Path, repo: &str, files: Vec<PinFile>) -> Result<(), StoreError> {
    let dir = repo_dir(root, repo)?;
    fs::create_dir_all(&dir)?;
    let mut pin = read_pin(&dir, repo).unwrap_or(Pin {
        repo: repo.to_owned(),
        files: Vec::new(),
    });
    for file in files {
        pin.files.retain(|f| f.name != file.name);
        pin.files.push(file);
    }
    pin.files.sort_by(|a, b| a.name.cmp(&b.name));
    write_pin(&dir, &pin)?;
    Ok(())
}

/// Every pin in the store: `<root>/<Organisation>/<Repository>/.eliezer-pin.json`.
pub fn scan(root: &Path) -> Vec<Pin> {
    let mut pins = Vec::new();
    let Ok(orgs) = fs::read_dir(root) else {
        return pins;
    };
    for org in orgs.flatten() {
        let Some(org_name) = org.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(repos) = fs::read_dir(org.path()) else {
            continue;
        };
        for repo in repos.flatten() {
            let Some(name) = repo.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let repo_id = format!("{org_name}/{name}");
            if valid_repo(&repo_id)
                && let Some(pin) = read_pin(&repo.path(), &repo_id)
            {
                pins.push(pin);
            }
        }
    }
    pins.sort_by(|a, b| a.repo.cmp(&b.repo));
    pins
}

/// Deletes a repository fetched with `eliezer download`: its directory with all files, including
/// unfinished `.part` files. Only a directory with a pin file is touched, a symbolic link is refused, and
/// the result is the number of bytes freed.
pub fn remove(root: &Path, repo: &str) -> Result<u64, StoreError> {
    let dir = repo_dir(root, repo)?;
    let meta = fs::symlink_metadata(&dir).map_err(|_| StoreError::Invalid)?;
    if !meta.is_dir() || read_pin(&dir, repo).is_none() {
        return Err(StoreError::Invalid);
    }
    if !dir.canonicalize()?.starts_with(root.canonicalize()?) {
        return Err(StoreError::Invalid);
    }
    let freed = dir_bytes(&dir);
    fs::remove_dir_all(&dir)?;
    // The organisation directory goes too when it is now empty; failing here is harmless.
    if let Some(parent) = dir.parent() {
        let _ = fs::remove_dir(parent);
    }
    Ok(freed)
}

fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries.flatten().fold(0_u64, |sum, entry| {
        let size = match entry.metadata() {
            Ok(m) if m.is_dir() => dir_bytes(&entry.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        };
        sum.saturating_add(size)
    })
}

/// The bytes of all pinned files; what counts against `cache_limit_gb`.
pub fn pinned_bytes(root: &Path) -> u64 {
    scan(root)
        .iter()
        .flat_map(|p| p.files.iter())
        .fold(0_u64, |sum, f| sum.saturating_add(f.bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    Missing,
    SizeMismatch {
        found: u64,
    },
    /// The content hashes differently from the pin (only found with `verify`).
    Modified,
}
impl Problem {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::SizeMismatch { .. } => "size_mismatch",
            Self::Modified => "modified",
        }
    }
}

/// Checks the files of a pin: presence and size always, the hash when `verify` is set. Blocking.
pub fn check(root: &Path, pin: &Pin, verify: bool) -> Vec<(PinFile, Option<Problem>)> {
    let Ok(dir) = repo_dir(root, &pin.repo) else {
        return Vec::new();
    };
    pin.files
        .iter()
        .map(|file| {
            let path = dir.join(&file.name);
            let problem = match fs::metadata(&path) {
                Err(_) => Some(Problem::Missing),
                Ok(meta) if meta.len() != file.bytes => {
                    Some(Problem::SizeMismatch { found: meta.len() })
                }
                Ok(_) if verify => match (&file.sha256, sha256_file(&path)) {
                    (Some(expected), Ok(found)) if *expected != found => Some(Problem::Modified),
                    (_, Err(_)) => Some(Problem::Missing),
                    _ => None,
                },
                Ok(_) => None,
            };
            (file.clone(), problem)
        })
        .collect()
}

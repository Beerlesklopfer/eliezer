//! Import of GGUF model files into the model store (`eliezer --import`).
//!
//! An administrator moves (or copies) the files of another program, for example LM Studio, into
//! `storage.models_dir` and hands them to the service user. The directory structure below the
//! source directory is kept. Only regular `.gguf` files are imported: symbolic links are never
//! followed, existing files are never overwritten and nothing is written outside the model store.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufReader, Read},
    os::unix::fs::{MetadataExt, PermissionsExt, chown},
    path::{Component, Path, PathBuf},
};

#[derive(Debug)]
pub enum ImportError {
    /// The source directory cannot be used (missing, not a directory, overlaps the store).
    Source(String),
    Io {
        path: PathBuf,
        error: io::Error,
    },
    /// The copy does not have the size of the original; the partial file was removed.
    SizeMismatch(PathBuf),
}
impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(message) => f.write_str(message),
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::SizeMismatch(path) => write!(f, "{}: copy has a different size", path.display()),
        }
    }
}
impl std::error::Error for ImportError {}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> ImportError + '_ {
    move |error| ImportError::Io {
        path: path.to_owned(),
        error,
    }
}

/// User and group the imported files are handed to; `None` leaves that part unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Owner {
    pub uid: Option<u32>,
    pub gid: Option<u32>,
}

/// Finds a user in the text of /etc/passwd (`name:x:uid:gid:...`) and returns uid and primary gid.
/// Only local users are known; malformed lines are ignored.
pub fn lookup_user(passwd: &str, name: &str) -> Option<(u32, u32)> {
    passwd.lines().find_map(|line| {
        let mut fields = line.split(':');
        if fields.next()? != name {
            return None;
        }
        let uid = fields.nth(1)?.parse().ok()?;
        let gid = fields.next()?.parse().ok()?;
        Some((uid, gid))
    })
}

/// Finds a group in the text of /etc/group (`name:x:gid:members`).
pub fn lookup_group(group: &str, name: &str) -> Option<u32> {
    group.lines().find_map(|line| {
        let mut fields = line.split(':');
        if fields.next()? != name {
            return None;
        }
        fields.nth(1)?.parse().ok()
    })
}

/// Asks the system's name services (local files, SSSD, LDAP...) via `getent` and returns the
/// matching line. The name was validated by the configuration; `--` keeps it from being an option
/// and no shell is involved.
fn getent(database: &str, name: &str) -> Option<String> {
    let output = std::process::Command::new("getent")
        .args([database, "--", name])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Like [`resolve_owner`], but looks the accounts up on the running system.
pub fn resolve_owner_system(
    run_user: Option<&str>,
    run_group: Option<&str>,
) -> Result<Option<Owner>, String> {
    let passwd = run_user
        .and_then(|n| getent("passwd", n))
        .unwrap_or_default();
    let group = run_group
        .and_then(|n| getent("group", n))
        .unwrap_or_default();
    resolve_owner(&passwd, &group, run_user, run_group)
}

/// Combines the configured `run_user` and `run_group` (text of /etc/passwd and /etc/group).
/// A group without a user changes only the group; a user alone gets its primary group.
pub fn resolve_owner(
    passwd: &str,
    group: &str,
    run_user: Option<&str>,
    run_group: Option<&str>,
) -> Result<Option<Owner>, String> {
    let user = match run_user {
        Some(name) => {
            Some(lookup_user(passwd, name).ok_or_else(|| format!("user '{name}' not found"))?)
        }
        None => None,
    };
    let gid = match run_group {
        Some(name) => {
            Some(lookup_group(group, name).ok_or_else(|| format!("group '{name}' not found"))?)
        }
        None => user.map(|(_, gid)| gid),
    };
    Ok(match (user, gid) {
        (None, None) => None,
        (user, gid) => Some(Owner {
            uid: user.map(|(uid, _)| uid),
            gid,
        }),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Skip {
    /// The target file already exists; it is left untouched.
    Exists,
    /// A symbolic link or another file that is not a regular file.
    NotRegular,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub source: PathBuf,
    pub target: PathBuf,
    pub size: u64,
}

#[derive(Debug, Default)]
pub struct Plan {
    pub items: Vec<Item>,
    pub skipped: Vec<(PathBuf, Skip)>,
}

fn is_gguf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

fn overlaps(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// Collects the `.gguf` files below `source` and where they go below `models_dir`.
pub fn plan(source: &Path, models_dir: &Path) -> Result<Plan, ImportError> {
    let source_real = fs::canonicalize(source).map_err(io_error(source))?;
    if !source_real.is_dir() {
        return Err(ImportError::Source(format!(
            "{} is not a directory",
            source.display()
        )));
    }
    // The store may not exist yet; resolve its existing part.
    let mut store = models_dir.to_owned();
    let mut tail = Vec::new();
    while !store.exists() {
        match (store.file_name().map(ToOwned::to_owned), store.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                store = parent.to_owned();
            }
            _ => break,
        }
    }
    let mut store = fs::canonicalize(&store).map_err(io_error(models_dir))?;
    store.extend(tail.iter().rev());
    if overlaps(&source_real, &store) {
        return Err(ImportError::Source(
            "the source directory and the model store must not contain each other".into(),
        ));
    }
    let mut plan = Plan::default();
    let mut pending = vec![source_real.clone()];
    while let Some(dir) = pending.pop() {
        let mut entries: Vec<_> = fs::read_dir(&dir)
            .map_err(io_error(&dir))?
            .collect::<Result<_, _>>()
            .map_err(io_error(&dir))?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(io_error(&path))?;
            if metadata.is_dir() {
                pending.push(path);
            } else if is_gguf(&path) {
                if !metadata.is_file() {
                    plan.skipped.push((path, Skip::NotRegular));
                    continue;
                }
                // Only normal path components can occur below the source directory.
                let relative = path.strip_prefix(&source_real).map_err(|_| {
                    ImportError::Source("a file escaped the source directory".into())
                })?;
                if !relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
                {
                    return Err(ImportError::Source("unexpected path component".into()));
                }
                let target = store.join(relative);
                if target.exists() {
                    plan.skipped.push((path, Skip::Exists));
                } else {
                    plan.items.push(Item {
                        source: path,
                        target,
                        size: metadata.len(),
                    });
                }
            }
        }
    }
    plan.items.sort_by(|a, b| a.source.cmp(&b.source));
    Ok(plan)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Move,
    Copy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum How {
    /// Renamed within one file system.
    Renamed,
    /// Copied (by request, or because the files are on different file systems).
    Copied,
}

fn hand_over(path: &Path, owner: Option<Owner>, mode: u32) -> Result<(), ImportError> {
    if let Some(owner) = owner {
        chown(path, owner.uid, owner.gid).map_err(io_error(path))?;
    }
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(io_error(path))
}

/// Creates missing parent directories of `target` below `store_root`, owned like the files.
fn create_parents(target: &Path, owner: Option<Owner>) -> Result<(), ImportError> {
    let Some(parent) = target.parent() else {
        return Ok(());
    };
    let mut missing = Vec::new();
    let mut current = parent;
    while !current.exists() {
        missing.push(current.to_owned());
        match current.parent() {
            Some(p) => current = p,
            None => break,
        }
    }
    for dir in missing.iter().rev() {
        fs::create_dir(dir).map_err(io_error(dir))?;
        hand_over(dir, owner, 0o755)?;
    }
    Ok(())
}

fn copy_verified(source: &Path, target: &Path) -> Result<(), ImportError> {
    let mut partial = target.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let result = (|| {
        let copied = fs::copy(source, &partial).map_err(io_error(source))?;
        File::open(&partial)
            .and_then(|f| f.sync_all())
            .map_err(io_error(&partial))?;
        if copied != fs::metadata(source).map_err(io_error(source))?.len() {
            return Err(ImportError::SizeMismatch(target.to_owned()));
        }
        fs::rename(&partial, target).map_err(io_error(target))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

/// Moves or copies one planned file. Never overwrites: a file that appeared in the meantime is
/// reported as an error and left alone.
pub fn execute_item(item: &Item, mode: Mode, owner: Option<Owner>) -> Result<How, ImportError> {
    if item.target.exists() {
        return Err(ImportError::Source(format!(
            "{} already exists",
            item.target.display()
        )));
    }
    create_parents(&item.target, owner)?;
    let how = match mode {
        Mode::Copy => {
            copy_verified(&item.source, &item.target)?;
            How::Copied
        }
        Mode::Move => match fs::rename(&item.source, &item.target) {
            Ok(()) => How::Renamed,
            Err(error) if error.kind() == io::ErrorKind::CrossesDevices => {
                copy_verified(&item.source, &item.target)?;
                fs::remove_file(&item.source).map_err(io_error(&item.source))?;
                How::Copied
            }
            Err(error) => return Err(io_error(&item.source)(error)),
        },
    };
    hand_over(&item.target, owner, 0o644)?;
    Ok(how)
}

/// The owner of a directory, used by callers that import without a configured user.
pub fn owner_of(path: &Path) -> Option<Owner> {
    fs::metadata(path).ok().map(|m| Owner {
        uid: Some(m.uid()),
        gid: Some(m.gid()),
    })
}

// ---------------------------------------------------------------------------------------------
// GGUF header
// ---------------------------------------------------------------------------------------------

/// The facts about a model that the memory estimate needs, read from the GGUF header only.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufInfo {
    pub architecture: String,
    pub name: Option<String>,
    pub context_length: Option<u64>,
    pub layers: Option<u64>,
    pub kv_heads: Option<u64>,
    pub head_dim: Option<u64>,
    /// Sliding-window or recurrent layers: the estimate needs extra fields for those.
    pub hybrid: bool,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    NotGguf,
    Malformed,
}
impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::NotGguf => f.write_str("not a GGUF file"),
            Self::Malformed => f.write_str("malformed GGUF header"),
        }
    }
}
impl std::error::Error for GgufError {}
impl From<io::Error> for GgufError {
    fn from(error: io::Error) -> Self {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            Self::Malformed
        } else {
            Self::Io(error)
        }
    }
}

enum Value {
    Number(u64),
    Text(String),
    Other,
}

struct Reader<R: Read>(R);
impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut buffer = [0; N];
        self.0.read_exact(&mut buffer)?;
        Ok(buffer)
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn skip(&mut self, count: u64) -> Result<(), GgufError> {
        let copied = io::copy(&mut (&mut self.0).take(count), &mut io::sink())?;
        if copied == count {
            Ok(())
        } else {
            Err(GgufError::Malformed)
        }
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let length = self.u64()?;
        if length > 1 << 20 {
            return Err(GgufError::Malformed);
        }
        let mut buffer = vec![0; length as usize];
        self.0.read_exact(&mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }
    /// Reads a value of the given GGUF type; large arrays (the vocabulary) are skipped.
    fn value(&mut self, kind: u32, depth: u32) -> Result<Value, GgufError> {
        let size = |kind| match kind {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        Ok(match kind {
            0 | 2 | 4 | 10 => {
                let width = size(kind).ok_or(GgufError::Malformed)?;
                let mut raw = [0_u8; 8];
                self.0.read_exact(&mut raw[..width])?;
                Value::Number(u64::from_le_bytes(raw))
            }
            1 | 3 | 5 | 6 | 7 | 11 | 12 => {
                self.skip(size(kind).ok_or(GgufError::Malformed)? as u64)?;
                Value::Other
            }
            8 => Value::Text(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(GgufError::Malformed);
                }
                let inner = self.u32()?;
                let count = self.u64()?;
                match inner {
                    8 => {
                        for _ in 0..count {
                            let length = self.u64()?;
                            self.skip(length)?;
                        }
                        Value::Other
                    }
                    9 => {
                        for _ in 0..count {
                            self.value(9, depth + 1)?;
                        }
                        Value::Other
                    }
                    _ => {
                        let width = size(inner).ok_or(GgufError::Malformed)? as u64;
                        // Small numeric arrays (per-layer head counts) are kept as their maximum.
                        if count > 0 && count <= 4096 && matches!(inner, 0 | 2 | 4 | 10) {
                            let mut largest = 0;
                            for _ in 0..count {
                                if let Value::Number(n) = self.value(inner, depth + 1)? {
                                    largest = largest.max(n);
                                }
                            }
                            Value::Number(largest)
                        } else {
                            self.skip(width.checked_mul(count).ok_or(GgufError::Malformed)?)?;
                            Value::Other
                        }
                    }
                }
            }
            _ => return Err(GgufError::Malformed),
        })
    }
}

/// Reads architecture, context length, layer count and attention shape from a GGUF header.
/// The tensor data is never read.
pub fn read_gguf(path: &Path) -> Result<GgufInfo, GgufError> {
    let mut reader = Reader(BufReader::new(File::open(path)?));
    if reader.bytes::<4>()? != *b"GGUF" {
        return Err(GgufError::NotGguf);
    }
    let _version = reader.u32()?;
    let _tensors = reader.u64()?;
    let count = reader.u64()?;
    if count > 100_000 {
        return Err(GgufError::Malformed);
    }
    let mut numbers = BTreeMap::new();
    let mut texts = BTreeMap::new();
    let mut hybrid = false;
    for _ in 0..count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        if key.contains("sliding_window") || key.contains(".ssm.") || key.contains("linear") {
            hybrid = true;
        }
        match reader.value(kind, 0)? {
            Value::Number(n) => {
                numbers.insert(key, n);
            }
            Value::Text(t) => {
                texts.insert(key, t);
            }
            Value::Other => {}
        }
    }
    let architecture = texts
        .get("general.architecture")
        .cloned()
        .ok_or(GgufError::Malformed)?;
    let number = |suffix: &str| numbers.get(&format!("{architecture}.{suffix}")).copied();
    let heads = number("attention.head_count");
    let embedding = number("embedding_length");
    let head_dim = number("attention.key_length").or_else(|| Some(embedding? / heads?.max(1)));
    Ok(GgufInfo {
        name: texts.get("general.name").cloned(),
        context_length: number("context_length"),
        layers: number("block_count"),
        kv_heads: number("attention.head_count_kv").or(heads),
        head_dim,
        hybrid,
        architecture,
    })
}

/// A model key derived from a file name: lowercase, only characters the configuration accepts.
pub fn key_from(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let key: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    key.trim_matches(|c| c == '-' || c == '.').to_owned()
}

/// A `[[model]]` block to paste into the configuration. It is a suggestion: the memory estimate
/// of sliding-window and hybrid models needs more fields, which the block points out.
pub fn suggest(target: &Path, size: u64, info: Option<&GgufInfo>) -> String {
    let embedding =
        info.is_some_and(|i| i.architecture.contains("bert") || i.architecture.contains("embed"));
    let mut text = format!(
        "[[model]]\nkey = \"{}\"\nbackend = \"llamacpp\"\nsource = {{ file = \"{}\" }}\nkind = \"{}\"\n",
        key_from(target),
        target.display(),
        if embedding { "embedding" } else { "llm" }
    );
    match info {
        Some(info) => {
            if let Some(context) = info.context_length {
                text += &format!(
                    "max_context = {context}\ndefault_context = {}\n",
                    context.min(32768)
                );
            }
            text += &format!("size_gb = {:.1}\n", size as f64 / 1e9);
            if let (Some(layers), Some(kv_heads), Some(head_dim)) =
                (info.layers, info.kv_heads, info.head_dim)
            {
                text += &format!(
                    "arch = {{ layers = {layers}, kv_heads = {kv_heads}, head_dim = {head_dim} }}\n"
                );
            } else {
                text += "# arch = { layers = ?, kv_heads = ?, head_dim = ? }  (not found in the file)\n";
            }
            if info.hybrid {
                text += "# Sliding-window or recurrent layers: add swa_layers/window or linear_layers/state_bytes to arch.\n";
            }
        }
        None => {
            text += &format!(
                "size_gb = {:.1}\n# The GGUF header could not be read; add arch = {{ ... }} or vram_gb.\n",
                size as f64 / 1e9
            );
        }
    }
    text
}

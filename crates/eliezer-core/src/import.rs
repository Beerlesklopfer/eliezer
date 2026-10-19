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

/// A local or directory account: numeric ids and the home directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Account {
    pub uid: u32,
    pub gid: u32,
    pub home: PathBuf,
}

/// Parses one line of `getent passwd` (`name:x:uid:gid:gecos:home:shell`).
pub fn parse_account(line: &str) -> Option<Account> {
    let fields: Vec<&str> = line.trim().split(':').collect();
    Some(Account {
        uid: fields.get(2)?.parse().ok()?,
        gid: fields.get(3)?.parse().ok()?,
        home: PathBuf::from(fields.get(5).filter(|h| h.starts_with('/'))?),
    })
}

/// Looks an account up on the running system (local files, SSSD, LDAP).
pub fn account(name: &str) -> Option<Account> {
    parse_account(&getent("passwd", name)?)
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
    /// Sliding-window or recurrent layers were announced in the header.
    pub hybrid: bool,
    /// Layers with a sliding attention window, and the window size (tokens).
    pub swa_layers: Option<u64>,
    pub window: Option<u64>,
    /// KV heads and head size of the full-attention layers, when they differ from the others.
    pub full_kv_heads: Option<u64>,
    pub full_head_dim: Option<u64>,
    /// Recurrent (linear attention) layers and the bytes of state each keeps per sequence.
    pub linear_layers: Option<u64>,
    pub state_bytes: Option<u64>,
    /// The file carries a tokenizer (language models do, diffusion models do not).
    pub has_tokenizer: bool,
    /// How the chat template of the model takes a thinking setting.
    pub thinking: Thinking,
}

/// What the chat template does with a thinking setting; it decides the `reasoning` line of the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Thinking {
    /// The template switches thinking on or off (`enable_thinking`).
    pub toggle: bool,
    /// The template takes an effort level (`reasoning_effort`), with `xhigh` among them.
    pub effort: bool,
    pub xhigh: bool,
    /// Thinking is on when the request says nothing.
    pub default_on: bool,
}
impl Thinking {
    fn from_template(template: &str) -> Self {
        let compact: String = template.split_whitespace().collect();
        let toggle = compact.contains("enable_thinking");
        Self {
            toggle,
            effort: compact.contains("reasoning_effort"),
            xhigh: compact.contains("'xhigh'") || compact.contains("\"xhigh\""),
            // Gemma 4 defaults to off, Qwen to on.
            default_on: toggle && !compact.contains("enable_thinking|default(false)"),
        }
    }
    /// The `reasoning` line for the configuration, if the template takes a setting at all.
    fn line(self) -> Option<String> {
        if !self.toggle && !self.effort {
            return None;
        }
        let mut options: Vec<&str> = Vec::new();
        if self.toggle {
            options.push("none");
        }
        if self.effort {
            options.push("low");
            options.push("medium");
            options.push(if self.xhigh { "xhigh" } else { "high" });
        } else {
            options.push("medium");
        }
        let default = if self.toggle && !self.default_on {
            "none"
        } else if self.effort {
            if self.xhigh { "xhigh" } else { "medium" }
        } else {
            "medium"
        };
        let list: Vec<String> = options.iter().map(|o| format!("\"{o}\"")).collect();
        Some(format!(
            "reasoning = {{ options = [{}], default = \"{default}\" }}\n",
            list.join(", ")
        ))
    }
}

/// What a model file is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Llm,
    Embedding,
    /// Diffusion models (stable-diffusion.cpp): no tokenizer and no transformer layers in the header.
    Image,
}
impl GgufInfo {
    /// The header announces sliding-window or recurrent layers, but the layout could not be worked
    /// out, so the memory estimate treats every layer as full attention (too high).
    pub fn hybrid_unresolved(&self) -> bool {
        self.hybrid && self.swa_layers.is_none() && self.linear_layers.is_none()
    }

    pub fn kind(&self) -> Kind {
        if !self.has_tokenizer && self.layers.is_none() {
            Kind::Image
        } else if self.architecture.contains("bert") || self.architecture.contains("embed") {
            Kind::Embedding
        } else {
            Kind::Llm
        }
    }
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
    /// A small numeric or boolean array (per-layer head counts, the sliding-window pattern).
    List(Vec<u64>),
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
            // Signed integers (the real Gemma 4 file stores its per-layer head counts as int32).
            1 | 3 | 5 | 11 => {
                let width = size(kind).ok_or(GgufError::Malformed)?;
                let mut raw = [0_u8; 8];
                self.0.read_exact(&mut raw[..width])?;
                if raw[width - 1] & 0x80 != 0 {
                    raw[width..].fill(0xff);
                }
                Value::Number(i64::from_le_bytes(raw).max(0) as u64)
            }
            6 | 7 | 12 => {
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
                        // Small numeric and boolean arrays (per-layer head counts, the
                        // sliding-window pattern) are kept; anything else is skipped.
                        if count > 0
                            && count <= 4096
                            && matches!(inner, 0 | 1 | 2 | 3 | 4 | 5 | 7 | 10 | 11)
                        {
                            let mut items = Vec::with_capacity(count as usize);
                            for _ in 0..count {
                                if inner == 7 {
                                    items.push(u64::from(self.bytes::<1>()?[0] != 0));
                                } else if let Value::Number(n) = self.value(inner, depth + 1)? {
                                    items.push(n);
                                }
                            }
                            Value::List(items)
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
    let mut lists: BTreeMap<String, Vec<u64>> = BTreeMap::new();
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
            Value::List(items) => {
                lists.insert(key, items);
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
    let list = |suffix: &str| lists.get(&format!("{architecture}.{suffix}"));
    let layers = number("block_count");
    let heads = number("attention.head_count");
    let embedding = number("embedding_length");
    let key_length = number("attention.key_length");
    let default_dim = key_length.or_else(|| Some(embedding? / heads?.max(1)));
    let kv_scalar = number("attention.head_count_kv");
    let kv_list = list("attention.head_count_kv");

    let mut info = GgufInfo {
        name: texts.get("general.name").cloned(),
        context_length: number("context_length"),
        layers,
        kv_heads: kv_scalar
            .or_else(|| kv_list.and_then(|l| l.iter().copied().max()))
            .or(heads),
        head_dim: default_dim,
        hybrid,
        has_tokenizer: texts.contains_key("tokenizer.ggml.model"),
        thinking: texts
            .get("tokenizer.chat_template")
            .map_or_else(Thinking::default, |t| Thinking::from_template(t)),
        architecture: architecture.clone(),
        ..GgufInfo::default()
    };

    // Sliding-window models (Gemma): a per-layer pattern says which layers use the window; the
    // others attend to everything and often have their own head count and size.
    if let (Some(pattern), Some(window)) = (
        list("attention.sliding_window_pattern"),
        number("attention.sliding_window"),
    ) {
        let swa = pattern.iter().filter(|p| **p == 1).count() as u64;
        let per_layer = |wanted: bool| -> Option<u64> {
            match kv_list {
                Some(l) if l.len() == pattern.len() => l
                    .iter()
                    .zip(pattern)
                    .find(|(_, p)| (**p == 1) == wanted)
                    .map(|(k, _)| *k),
                _ => kv_scalar,
            }
        };
        info.swa_layers = Some(swa);
        info.window = Some(window);
        info.kv_heads = per_layer(true).or(info.kv_heads);
        info.head_dim = number("attention.key_length_swa").or(default_dim);
        let full_kv = per_layer(false);
        if full_kv.is_some() && full_kv != info.kv_heads {
            info.full_kv_heads = full_kv;
        }
        if key_length.is_some() && key_length != info.head_dim {
            info.full_head_dim = key_length;
        }
    }

    // Hybrid models with linear attention (Qwen3.5/3.8): only every n-th layer keeps a KV cache,
    // the others keep a fixed recurrent state per sequence.
    if let (Some(interval), Some(total)) = (number("full_attention_interval"), layers)
        && let Some(full) = total.checked_div(interval)
    {
        info.linear_layers = Some(total - full);
        if let (Some(kernel), Some(state), Some(groups), Some(rank), Some(inner)) = (
            number("ssm.conv_kernel"),
            number("ssm.state_size"),
            number("ssm.group_count"),
            number("ssm.time_step_rank"),
            number("ssm.inner_size"),
        ) && rank > 0
        {
            // 4-byte values: the recurrent matrix per head plus the convolution state.
            let recurrent = rank * state * (inner / rank);
            let conv = kernel.saturating_sub(1) * (inner + 2 * groups * state);
            info.state_bytes = Some(4 * (recurrent + conv));
        }
    }
    Ok(info)
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

/// The `[[model]]` block for one file under the given key. It is a suggestion: the memory estimate
/// of sliding-window and hybrid models needs more fields, which the block points out. Image models
/// get a commented block only, because their runtime (stable-diffusion.cpp) is not connected yet.
pub fn block(key: &str, target: &Path, size: u64, info: Option<&GgufInfo>) -> String {
    block_with(key, target, size, info, sibling_projector(target).as_ref())
}

/// The vision projector next to a model file: the only `mmproj*.gguf` file in the same directory
/// (the layout of LM Studio and Hugging Face downloads), when its header says it is a projector.
/// With several candidates nothing is guessed.
pub fn sibling_projector(target: &Path) -> Option<(PathBuf, u64)> {
    let mut found = None;
    for entry in std::fs::read_dir(target.parent()?).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_str()?;
        if name.starts_with("mmproj") && name.ends_with(".gguf") {
            if found.is_some() {
                return None;
            }
            found = Some(entry.path());
        }
    }
    let path = found?;
    let meta = std::fs::metadata(&path).ok()?;
    let info = read_gguf(&path).ok()?;
    (meta.is_file() && info.architecture == "clip").then_some((path, meta.len()))
}

/// Like [`block`], with an explicit vision projector (path and size in bytes).
pub fn block_with(
    key: &str,
    target: &Path,
    size: u64,
    info: Option<&GgufInfo>,
    projector: Option<&(PathBuf, u64)>,
) -> String {
    // Never below 0.1 GB: the configuration rejects zero, and a tiny file is still a model.
    let gb = (size as f64 / 1e9).max(0.1);
    let kind = info.map_or(Kind::Llm, GgufInfo::kind);
    if kind == Kind::Image {
        return format!(
            "# Image model (needs the stable-diffusion.cpp runtime, which is not connected yet):\n\
             # [[model]]\n# key = \"{key}\"\n# backend = \"sdcpp\"\n# source = {{ file = \"{}\" }}\n\
             # kind = \"image\"\n# vram_gb = {:.1}  (measure it; image models need the figure)\n\
             # extra = {{ text_encoder = \"...\", vae = \"...\" }}\n",
            target.display(),
            gb * 1.4
        );
    }
    let projector = projector.filter(|_| kind == Kind::Llm);
    let source = match projector {
        Some((path, _)) => format!(
            "{{ file = \"{}\", projector = \"{}\" }}",
            target.display(),
            path.display()
        ),
        None => format!("{{ file = \"{}\" }}", target.display()),
    };
    let mut text = format!(
        "[[model]]\nkey = \"{key}\"\nbackend = \"llamacpp\"\nsource = {source}\nkind = \"{}\"\n",
        if kind == Kind::Embedding {
            "embedding"
        } else {
            "llm"
        }
    );
    if let Some((_, bytes)) = projector {
        text += &format!(
            "vision = true\nprojector_gb = {:.1}\n",
            (*bytes as f64 / 1e9).max(0.1)
        );
    }
    match info {
        Some(info) => {
            if kind == Kind::Llm
                && let Some(line) = info.thinking.line()
            {
                text += &line;
            }
            if let Some(context) = info.context_length {
                text += &format!(
                    "max_context = {context}\ndefault_context = {}\n",
                    context.min(32768)
                );
            }
            if let (Some(layers), Some(kv_heads), Some(head_dim)) =
                (info.layers, info.kv_heads, info.head_dim)
            {
                text += &format!("size_gb = {gb:.1}\n");
                let mut arch =
                    format!("layers = {layers}, kv_heads = {kv_heads}, head_dim = {head_dim}");
                if let (Some(swa), Some(window)) = (info.swa_layers, info.window) {
                    arch += &format!(", swa_layers = {swa}, window = {window}");
                }
                if let Some(n) = info.full_kv_heads {
                    arch += &format!(", full_kv_heads = {n}");
                }
                if let Some(n) = info.full_head_dim {
                    arch += &format!(", full_head_dim = {n}");
                }
                if let (Some(linear), Some(bytes)) = (info.linear_layers, info.state_bytes) {
                    arch += &format!(", linear_layers = {linear}, state_bytes = {bytes}");
                }
                text += &format!("arch = {{ {arch} }}\n");
            } else {
                // Without the attention shape the estimate cannot be computed; state a plain figure.
                text += &format!("vram_gb = {:.1}  # estimate: file size + 20%\n", gb * 1.2);
            }
            if info.hybrid_unresolved() {
                text += "# Sliding-window or recurrent layers were announced but not understood: the estimate treats every layer as full attention and is too high.\n";
            }
        }
        None => {
            text += &format!(
                "vram_gb = {:.1}  # estimate: file size + 20% (the GGUF header could not be read)\n",
                gb * 1.2
            );
        }
    }
    text
}

/// A block under the key derived from the file name.
pub fn suggest(target: &Path, size: u64, info: Option<&GgufInfo>) -> String {
    block(&key_from(target), target, size, info)
}

/// One imported file, as far as the configuration is concerned.
#[derive(Debug, Clone)]
pub struct Imported {
    pub target: PathBuf,
    pub size: u64,
    pub info: Option<GgufInfo>,
}

/// The configuration text after adding the imported models.
#[derive(Debug, Default)]
pub struct ConfigUpdate {
    /// The complete new text (equal to the old one when nothing was added).
    pub text: String,
    pub added: Vec<String>,
    /// Files that were not added, with the reason (already configured, projector, image model).
    pub skipped: Vec<(PathBuf, &'static str)>,
    /// Commented blocks for models that cannot be served yet.
    pub notes: Vec<String>,
    pub backend_enabled: bool,
}

fn string_at<'a>(table: &'a toml::Table, path: &[&str]) -> Option<&'a str> {
    let mut value = table.get(path[0])?;
    for part in &path[1..] {
        value = value.get(part)?;
    }
    value.as_str()
}

/// Appends the language and embedding models to the configuration text. Files that are already
/// configured (same `source.file`) are skipped, keys are made unique, and `[backend.llamacpp]`
/// is added when missing. Comments and the rest of the text stay as they are. The caller validates
/// the result before writing it.
pub fn update_config(existing: &str, imported: &[Imported]) -> Result<ConfigUpdate, String> {
    let table: toml::Table =
        toml::from_str(existing).map_err(|_| "the configuration is not valid TOML".to_owned())?;
    let models = table
        .get("model")
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut keys: Vec<String> = models
        .iter()
        .filter_map(|m| {
            m.get("key")
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        })
        .collect();
    let files: Vec<String> = models
        .iter()
        .filter_map(|m| m.get("source")?.get("file")?.as_str().map(str::to_owned))
        .collect();
    let backend = table.get("backend").and_then(|b| b.get("llamacpp"));
    let backend_present = backend.is_some();
    let backend_enabled = backend
        .and_then(|b| b.get("enabled"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(false);
    if backend_present && !backend_enabled {
        return Err(
            "[backend.llamacpp] is present but disabled; enable it or add the models yourself"
                .into(),
        );
    }
    let _ = string_at(&table, &["storage", "models_dir"]);

    let mut update = ConfigUpdate {
        text: existing.to_owned(),
        ..ConfigUpdate::default()
    };
    let mut appended = String::new();
    for item in imported {
        let name = item
            .target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name.starts_with("mmproj") {
            update.skipped.push((
                item.target.clone(),
                "vision projector (attached to its model)",
            ));
            continue;
        }
        if files.iter().any(|f| Path::new(f) == item.target) {
            update
                .skipped
                .push((item.target.clone(), "already configured"));
            continue;
        }
        let mut key = key_from(&item.target);
        if key.is_empty() {
            key = "model".into();
        }
        let base = key.clone();
        let mut n = 2;
        while keys.contains(&key) {
            key = format!("{base}-{n}");
            n += 1;
        }
        let text = block(&key, &item.target, item.size, item.info.as_ref());
        if item.info.as_ref().map(GgufInfo::kind) == Some(Kind::Image) {
            update.skipped.push((
                item.target.clone(),
                "image model (runtime not connected yet)",
            ));
            update.notes.push(text);
            continue;
        }
        keys.push(key.clone());
        update.added.push(key);
        appended += &text;
        appended.push('\n');
    }
    if !update.added.is_empty() {
        let mut text = existing.trim_end().to_owned();
        text.push_str("\n\n# Added by `eliezer --import`; review the values.\n");
        if !backend_present {
            text.push_str("[backend.llamacpp]\nenabled = true\n\n");
            update.backend_enabled = true;
        }
        text.push_str(&appended);
        update.text = text;
    }
    Ok(update)
}

/// A configured model whose file is not what its entry says.
#[derive(Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub key: String,
    pub file: PathBuf,
    pub problem: &'static str,
}

/// Finds entries that point at a diffusion model but are not configured as image models (for
/// example a block pasted by hand with `kind = "llm"`). Entries whose file cannot be read are
/// left alone.
pub fn mismatches(existing: &str) -> Vec<Mismatch> {
    let Ok(table) = toml::from_str::<toml::Table>(existing) else {
        return Vec::new();
    };
    let Some(models) = table.get("model").and_then(toml::Value::as_array) else {
        return Vec::new();
    };
    models
        .iter()
        .filter_map(|model| {
            let key = model.get("key")?.as_str()?.to_owned();
            let file = PathBuf::from(model.get("source")?.get("file")?.as_str()?);
            let kind = model.get("kind").and_then(toml::Value::as_str).unwrap_or("llm");
            let info = read_gguf(&file).ok()?;
            (info.kind() == Kind::Image && kind != "image").then_some(Mismatch {
                key,
                file,
                problem: "the file is an image (diffusion) model, but the entry is not an image model",
            })
        })
        .collect()
}

/// Replaces the configuration file atomically (temporary file in the same directory, same mode and
/// owner), after keeping the old one as `<name>.bak`.
pub fn write_config(path: &Path, text: &str) -> Result<(), ImportError> {
    let metadata = fs::metadata(path).map_err(io_error(path))?;
    let mut backup = path.as_os_str().to_owned();
    backup.push(".bak");
    fs::copy(path, &backup).map_err(io_error(path))?;
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".new");
    let temporary = PathBuf::from(temporary);
    let result = (|| {
        fs::write(&temporary, text).map_err(io_error(&temporary))?;
        fs::set_permissions(&temporary, metadata.permissions()).map_err(io_error(&temporary))?;
        // Keep the owner when running as root; as another user the file stays with that user.
        let _ = chown(&temporary, Some(metadata.uid()), Some(metadata.gid()));
        fs::rename(&temporary, path).map_err(io_error(path))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

//! The API key file: one key per line, each optionally preceded by a comment `# name: NAME`.
//!
//! The service reads the keys (blank lines and other `#` lines are ignored); `eliezer --new-key`,
//! `--list-keys` and `--revoke-key` edit the file. Keys are random tokens in the style of OpenAI
//! (`sk-` and 64 hex digits). A key is shown once when it is created and never written to a log.

use std::{fs::File, io, io::Read};

const NAME_PREFIX: &str = "# name:";
const HEADER: &str =
    "# Eliezer API keys, one per line (mode 0600). Managed by `eliezer --new-key`.\n";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: Option<String>,
    pub key: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum KeyError {
    /// Names are 1 to 40 characters of letters, digits, `.`, `_` and `-`.
    InvalidName,
    NameTaken,
    NotFound,
}
impl std::fmt::Display for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidName => {
                "A key name has 1 to 40 characters: letters, digits, '.', '_' and '-'"
            }
            Self::NameTaken => "A key with this name exists already",
            Self::NotFound => "No key has this name",
        })
    }
}
impl std::error::Error for KeyError {}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 40
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
}

fn name_of(line: &str) -> Option<&str> {
    line.trim().strip_prefix(NAME_PREFIX).map(str::trim)
}

/// The keys of a key file text, with the names from the comments in front of them.
pub fn parse(text: &str) -> Vec<Entry> {
    let mut entries = Vec::new();
    let mut pending: Option<String> = None;
    for line in text.lines().map(str::trim) {
        if line.is_empty() {
            continue;
        }
        if let Some(name) = name_of(line) {
            pending = valid_name(name).then(|| name.to_owned());
        } else if !line.starts_with('#') {
            entries.push(Entry {
                name: pending.take(),
                key: line.to_owned(),
            });
        }
    }
    entries
}

/// A key for display: the start and the last four characters, never the whole key.
pub fn mask(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 12 {
        return "*".repeat(chars.len());
    }
    let head: String = chars[..3].iter().collect();
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{head}…{tail}")
}

/// A new random key: `sk-` and 64 hex digits from the system's random source (256 bits).
pub fn generate() -> io::Result<String> {
    let mut bytes = [0_u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!("sk-{hex}"))
}

/// The first name `key-1`, `key-2`... that no entry uses.
pub fn free_name(entries: &[Entry]) -> String {
    (1..)
        .map(|n| format!("key-{n}"))
        .find(|candidate| !entries.iter().any(|e| e.name.as_deref() == Some(candidate)))
        .unwrap_or_else(|| "key".to_owned())
}

/// The text with one more key. A new file gets a short header.
pub fn add(text: &str, name: &str, key: &str) -> Result<String, KeyError> {
    if !valid_name(name) {
        return Err(KeyError::InvalidName);
    }
    if parse(text).iter().any(|e| e.name.as_deref() == Some(name)) {
        return Err(KeyError::NameTaken);
    }
    let mut out = if text.trim().is_empty() {
        HEADER.to_owned()
    } else {
        let mut t = text.trim_end().to_owned();
        t.push('\n');
        t
    };
    out.push_str(&format!("{NAME_PREFIX} {name}\n{key}\n"));
    Ok(out)
}

/// The text without the key of this name and the comment in front of it.
pub fn remove(text: &str, name: &str) -> Result<String, KeyError> {
    let lines: Vec<&str> = text.lines().collect();
    let mut keep = vec![true; lines.len()];
    let mut found = false;
    let mut pending: Option<(usize, String)> = None;
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(n) = name_of(trimmed) {
            pending = Some((index, n.to_owned()));
        } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
            if let Some((comment, n)) = pending.take()
                && n == name
            {
                keep[comment] = false;
                keep[index] = false;
                found = true;
            }
        } else if trimmed.is_empty() {
            // a blank line between the comment and the key does not detach it
        } else if name_of(trimmed).is_none() {
            pending = None;
        }
    }
    if !found {
        return Err(KeyError::NotFound);
    }
    let mut out: String = lines
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(l, _)| format!("{l}\n"))
        .collect();
    if out.trim().is_empty() {
        out = HEADER.to_owned();
    }
    Ok(out)
}

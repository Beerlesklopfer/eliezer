//! Fetching one file into the model store: resumable, checked against the hash Hugging Face lists,
//! and only then moved to its final name. Nothing here knows about HTTP.

use crate::{Hub, HubError, RepoFile, Token, store};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};
use tokio::io::AsyncWriteExt;

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Downloaded,
    /// The file was there already and matches.
    AlreadyPresent,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FetchError {
    Hub(HubError),
    Io(String),
    /// The hash of the received bytes differs from the one Hugging Face lists.
    Checksum,
    /// More or fewer bytes than the listing says.
    Size,
}
impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hub(error) => write!(f, "{error}"),
            Self::Io(reason) => write!(f, "{reason}"),
            Self::Checksum => {
                f.write_str("The checksum of the download does not match; the file was discarded")
            }
            Self::Size => f.write_str("The size of the download does not match the listing"),
        }
    }
}
impl From<HubError> for FetchError {
    fn from(error: HubError) -> Self {
        Self::Hub(error)
    }
}
impl From<store::StoreError> for FetchError {
    fn from(error: store::StoreError) -> Self {
        Self::Io(error.to_string())
    }
}
fn io_error(error: std::io::Error) -> FetchError {
    FetchError::Io(store::StoreError::from(error).to_string())
}

/// Hashes the bytes already in a partial file; blocking.
fn hash_part(path: &Path) -> std::io::Result<Sha256> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            return Ok(hasher);
        }
        hasher.update(&buffer[..n]);
    }
}

/// Fetches `file` of `repo` at `revision` to `dest`. A partial file `dest.part` from an earlier try
/// is continued. `progress` gets the bytes of the file that are in place.
pub async fn fetch(
    hub: &dyn Hub,
    repo: &str,
    revision: &str,
    file: &RepoFile,
    dest: &Path,
    token: Option<&Token>,
    progress: &(dyn Fn(u64) + Send + Sync),
) -> Result<Outcome, FetchError> {
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = std::path::PathBuf::from(part);

    if let Ok(meta) = std::fs::metadata(dest)
        && meta.len() == file.bytes
    {
        let matches = match &file.sha256 {
            None => true,
            Some(expected) => {
                let path = dest.to_owned();
                tokio::task::spawn_blocking(move || store::sha256_file(&path))
                    .await
                    .is_ok_and(|r| r.is_ok_and(|found| found == *expected))
            }
        };
        if matches {
            progress(file.bytes);
            return Ok(Outcome::AlreadyPresent);
        }
    }

    let mut offset = std::fs::metadata(&part).map_or(0, |m| m.len());
    if file.bytes > 0 && offset > file.bytes {
        let _ = std::fs::remove_file(&part);
        offset = 0;
    }
    let mut hasher = Sha256::new();
    if offset > 0 {
        let path = part.clone();
        hasher = tokio::task::spawn_blocking(move || hash_part(&path))
            .await
            .map_err(|_| FetchError::Io("The hashing task failed".to_owned()))?
            .map_err(io_error)?;
    }
    let mut stream = hub.open(repo, revision, &file.name, offset, token).await?;
    if offset > 0 && !stream.resumed {
        // The hub sends the file from its start: begin again.
        offset = 0;
        hasher = Sha256::new();
    }
    let mut out = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(offset == 0)
        .append(offset > 0)
        .open(&part)
        .await
        .map_err(io_error)?;
    let mut done = offset;
    progress(done);
    // A tokio file hands a write to a helper thread and `write_all` returns before it has landed. Whatever ends the
    // loop, the file is flushed before it is dropped: otherwise a stream that breaks leaves a partial file whose
    // length is still changing, and the next try resumes at a length that is not the one the data ends at.
    let streamed: Result<(), FetchError> = async {
        while let Some(chunk) = stream.body.next_chunk().await? {
            done += chunk.len() as u64;
            if file.bytes > 0 && done > file.bytes {
                return Err(FetchError::Size);
            }
            hasher.update(&chunk);
            out.write_all(&chunk).await.map_err(io_error)?;
            progress(done);
        }
        Ok(())
    }
    .await;
    let flushed = out.flush().await.map_err(io_error);
    streamed?;
    flushed?;
    drop(out);
    if file.bytes > 0 && done != file.bytes {
        return Err(FetchError::Size);
    }
    if let Some(expected) = &file.sha256
        && store::sha256_hex(hasher.finalize()) != *expected
    {
        let _ = std::fs::remove_file(&part);
        return Err(FetchError::Checksum);
    }
    std::fs::rename(&part, dest).map_err(io_error)?;
    Ok(Outcome::Downloaded)
}

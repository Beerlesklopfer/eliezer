//! The test bench: answers from memory, like the fake backend does for the runtimes.

use crate::{ChunkSource, Detail, FileStream, Hub, HubError, HubFuture, Query, Summary, Token};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// One opened download: repository, revision, file, start offset, and whether a token was given.
pub type Opened = (String, String, String, u64, bool);

#[derive(Default)]
pub struct FakeHub {
    pub summaries: Vec<Summary>,
    pub details: Vec<Detail>,
    /// File contents by `(repo, file)`, for the downloads.
    pub contents: HashMap<(String, String), Vec<u8>>,
    /// Size of the pieces a download is delivered in (0 means 64 KiB).
    pub chunk_size: usize,
    /// A download breaks off with `Unreachable` after this many bytes (to test resuming).
    pub break_after: Option<u64>,
    /// Ignore the requested start and send the whole file (a server without range support).
    pub ignore_range: bool,
    /// Every query seen, to check what the caller asked.
    pub queries: Mutex<Vec<Query>>,
    /// Every download opened: `(repo, revision, file, offset, token was given)`.
    pub opened: Mutex<Vec<Opened>>,
    /// Answer every call with this error.
    pub fail: Option<HubError>,
}

struct Pieces {
    data: Arc<Vec<u8>>,
    at: usize,
    chunk: usize,
    break_at: Option<usize>,
}
impl ChunkSource for Pieces {
    fn next_chunk(&mut self) -> HubFuture<'_, Option<Vec<u8>>> {
        Box::pin(async move {
            if self.break_at.is_some_and(|b| self.at >= b) {
                return Err(HubError::Unreachable);
            }
            if self.at >= self.data.len() {
                return Ok(None);
            }
            let mut end = (self.at + self.chunk).min(self.data.len());
            if let Some(b) = self.break_at {
                end = end.min(b);
            }
            let piece = self.data[self.at..end].to_vec();
            self.at = end;
            // Let other tasks run between pieces, like a network would.
            tokio::task::yield_now().await;
            Ok(Some(piece))
        })
    }
}

impl Hub for FakeHub {
    fn search<'a>(&'a self, query: &'a Query) -> HubFuture<'a, Vec<Summary>> {
        Box::pin(async move {
            if let Some(error) = &self.fail {
                return Err(error.clone());
            }
            if let Ok(mut seen) = self.queries.lock() {
                seen.push(query.clone());
            }
            let text = query.text.to_ascii_lowercase();
            Ok(self
                .summaries
                .iter()
                .filter(|s| s.repo.to_ascii_lowercase().contains(&text))
                .filter(|s| {
                    query
                        .license
                        .as_ref()
                        .is_none_or(|l| s.license.as_ref() == Some(l))
                })
                .filter(|s| {
                    let tags = query.task.pipeline_tags();
                    tags.is_empty() || tags.iter().any(|t| s.task.as_deref() == Some(*t))
                })
                .take(query.limit)
                .cloned()
                .collect())
        })
    }
    fn detail<'a>(&'a self, repo: &'a str) -> HubFuture<'a, Detail> {
        Box::pin(async move {
            if let Some(error) = &self.fail {
                return Err(error.clone());
            }
            self.details
                .iter()
                .find(|d| d.repo == repo)
                .cloned()
                .ok_or(HubError::NotFound)
        })
    }
    fn open<'a>(
        &'a self,
        repo: &'a str,
        revision: &'a str,
        file: &'a str,
        offset: u64,
        token: Option<&'a Token>,
    ) -> HubFuture<'a, FileStream> {
        Box::pin(async move {
            if let Some(error) = &self.fail {
                return Err(error.clone());
            }
            if let Ok(mut seen) = self.opened.lock() {
                seen.push((
                    repo.to_owned(),
                    revision.to_owned(),
                    file.to_owned(),
                    offset,
                    token.is_some(),
                ));
            }
            let gated = self.details.iter().any(|d| d.repo == repo && d.gated);
            if gated && token.is_none() {
                return Err(HubError::Forbidden);
            }
            let data = self
                .contents
                .get(&(repo.to_owned(), file.to_owned()))
                .ok_or(HubError::NotFound)?;
            let start = if self.ignore_range {
                0
            } else {
                offset as usize
            };
            let resumed = start > 0;
            let data = Arc::new(data.clone());
            Ok(FileStream {
                resumed,
                remaining: Some(data.len().saturating_sub(start) as u64),
                body: Box::new(Pieces {
                    at: start.min(data.len()),
                    chunk: if self.chunk_size == 0 {
                        64 * 1024
                    } else {
                        self.chunk_size
                    },
                    break_at: self.break_after.map(|b| start + b as usize),
                    data,
                }),
            })
        })
    }
}

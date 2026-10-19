//! The test bench: answers from memory, like the fake backend does for the runtimes.

use crate::{Detail, Hub, HubError, HubFuture, Query, Summary};
use std::sync::Mutex;

#[derive(Default)]
pub struct FakeHub {
    pub summaries: Vec<Summary>,
    pub details: Vec<Detail>,
    /// Every query seen, to check what the caller asked.
    pub queries: Mutex<Vec<Query>>,
    /// Answer every call with this error.
    pub fail: Option<HubError>,
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
}

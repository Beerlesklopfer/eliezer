//! The real client: the public metadata API of Hugging Face, without a token.
//!
//! Interfaces as probed on 2026-10-20 (docs/Runtimes.md):
//! `GET /api/models?search=&filter=&pipeline_tag=&sort=downloads&direction=-1&limit=` lists
//! repositories (`id`, `downloads`, `likes`, `pipeline_tag`, `tags`), and
//! `GET /api/models/{org}/{repo}?blobs=true` gives `sha`, `gated`, `gguf.architecture` and
//! `siblings[]` with `rfilename` and `size`.

use crate::{Detail, Hub, HubError, HubFuture, Query, RepoFile, Summary, valid_repo};
use serde_json::Value;
use std::time::Duration;

/// An answer larger than this is not a metadata answer.
const MAX_BODY: usize = 8 * 1024 * 1024;

pub struct HfClient {
    http: reqwest::Client,
    base: String,
}

impl HfClient {
    /// `base` is `https://huggingface.co`; tests point it at a local server.
    pub fn new(base: &str, timeout: Duration) -> Result<Self, HubError> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::limited(3))
            .user_agent(concat!("eliezer/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|_| HubError::Unreachable)?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_owned(),
        })
    }

    async fn get(&self, url: String, params: &[(&str, String)]) -> Result<Value, HubError> {
        let mut response = self
            .http
            .get(url)
            .query(params)
            .send()
            .await
            .map_err(|_| HubError::Unreachable)?;
        match response.status().as_u16() {
            200..=299 => {}
            401 | 403 | 404 => return Err(HubError::NotFound),
            429 => return Err(HubError::Busy),
            _ => return Err(HubError::BadResponse),
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| HubError::Unreachable)? {
            if body.len() + chunk.len() > MAX_BODY {
                return Err(HubError::BadResponse);
            }
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&body).map_err(|_| HubError::BadResponse)
    }
}

fn license_of(tags: &Value) -> Option<String> {
    tags.as_array()?
        .iter()
        .filter_map(Value::as_str)
        .find_map(|t| t.strip_prefix("license:").map(str::to_owned))
}

impl Hub for HfClient {
    fn search<'a>(&'a self, query: &'a Query) -> HubFuture<'a, Vec<Summary>> {
        Box::pin(async move {
            let mut params = vec![
                ("search", query.text.clone()),
                ("filter", query.format.name().to_owned()),
                ("sort", "downloads".to_owned()),
                ("direction", "-1".to_owned()),
                ("limit", query.limit.to_string()),
            ];
            if let Some(license) = &query.license {
                params.push(("filter", format!("license:{license}")));
            }
            if let Some(tag) = query.task.pipeline_tag() {
                params.push(("pipeline_tag", tag.to_owned()));
            }
            let list = self
                .get(format!("{}/api/models", self.base), &params)
                .await?;
            let items = list.as_array().ok_or(HubError::BadResponse)?;
            Ok(items
                .iter()
                .filter_map(|item| {
                    let repo = item["id"].as_str().filter(|r| valid_repo(r))?;
                    Some(Summary {
                        repo: repo.to_owned(),
                        downloads: item["downloads"].as_u64().unwrap_or(0),
                        likes: item["likes"].as_u64().unwrap_or(0),
                        task: item["pipeline_tag"].as_str().map(str::to_owned),
                        license: license_of(&item["tags"]),
                        gguf: item["tags"]
                            .as_array()
                            .is_some_and(|t| t.iter().any(|t| t == "gguf")),
                    })
                })
                .collect())
        })
    }

    fn detail<'a>(&'a self, repo: &'a str) -> HubFuture<'a, Detail> {
        Box::pin(async move {
            if !valid_repo(repo) {
                return Err(HubError::NotFound);
            }
            let value = self
                .get(
                    format!("{}/api/models/{repo}", self.base),
                    &[("blobs", "true".to_owned())],
                )
                .await?;
            let sha = value["sha"].as_str().ok_or(HubError::BadResponse)?;
            let files = value["siblings"]
                .as_array()
                .ok_or(HubError::BadResponse)?
                .iter()
                .filter_map(|f| {
                    Some(RepoFile {
                        name: f["rfilename"].as_str()?.to_owned(),
                        bytes: f["size"].as_u64().unwrap_or(0),
                    })
                })
                .collect();
            Ok(Detail {
                repo: repo.to_owned(),
                sha: sha.to_owned(),
                task: value["pipeline_tag"].as_str().map(str::to_owned),
                license: license_of(&value["tags"]),
                gated: !matches!(value["gated"], Value::Bool(false) | Value::Null),
                downloads: value["downloads"].as_u64().unwrap_or(0),
                architecture: value["gguf"]["architecture"].as_str().map(str::to_owned),
                files,
            })
        })
    }
}

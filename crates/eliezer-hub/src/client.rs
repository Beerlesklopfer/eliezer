//! The real client: the public metadata API of Hugging Face, without a token.
//!
//! Interfaces as probed on 2026-10-20 (docs/Runtimes.md):
//! `GET /api/models?search=&filter=&pipeline_tag=&sort=downloads&direction=-1&limit=` lists
//! repositories (`id`, `downloads`, `likes`, `pipeline_tag`, `tags`), and
//! `GET /api/models/{org}/{repo}?blobs=true` gives `sha`, `gated`, `gguf.architecture` and
//! `siblings[]` with `rfilename` and `size`.

use crate::{
    ChunkSource, Detail, FileStream, Hub, HubError, HubFuture, Query, RepoFile, Summary, Token,
    valid_file, valid_repo, valid_revision,
};
use serde_json::Value;
use std::time::Duration;

/// An answer larger than this is not a metadata answer.
const MAX_BODY: usize = 8 * 1024 * 1024;

pub struct HfClient {
    http: reqwest::Client,
    base: String,
    /// The time a metadata request may take in all. A file download has no such limit (it takes as long
    /// as the file is big); it only fails when no data arrives for `read_timeout`.
    request_timeout: Duration,
}

impl HfClient {
    /// `base` is `https://huggingface.co`; tests point it at a local server.
    pub fn new(base: &str, timeout: Duration) -> Result<Self, HubError> {
        Self::with_read_timeout(base, timeout, Duration::from_secs(60))
    }

    /// Like [`HfClient::new`]; `read_timeout` is how long a download may wait for the next piece of data.
    pub fn with_read_timeout(
        base: &str,
        timeout: Duration,
        read_timeout: Duration,
    ) -> Result<Self, HubError> {
        let http = reqwest::Client::builder()
            .connect_timeout(timeout)
            .read_timeout(read_timeout)
            // The file download goes through a few redirects to the storage host; the token is
            // not forwarded to another host (reqwest drops the Authorization header then).
            .redirect(reqwest::redirect::Policy::limited(5))
            .user_agent(concat!("eliezer/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|_| HubError::Unreachable)?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_owned(),
            request_timeout: timeout,
        })
    }

    async fn get(&self, url: String, params: &[(&str, String)]) -> Result<Value, HubError> {
        let mut response = self
            .http
            .get(url)
            .timeout(self.request_timeout)
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
            // One request per pipeline tag of the task (embedding models have two); no tag asks once.
            let tags: Vec<Option<&str>> = match query.task.pipeline_tags() {
                [] => vec![None],
                tags => tags.iter().map(|t| Some(*t)).collect(),
            };
            let mut hits: Vec<Summary> = Vec::new();
            for tag in tags {
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
                if let Some(tag) = tag {
                    params.push(("pipeline_tag", tag.to_owned()));
                }
                let list = self
                    .get(format!("{}/api/models", self.base), &params)
                    .await?;
                let items = list.as_array().ok_or(HubError::BadResponse)?;
                for item in items {
                    let Some(repo) = item["id"].as_str().filter(|r| valid_repo(r)) else {
                        continue;
                    };
                    if hits.iter().any(|h| h.repo == repo) {
                        continue;
                    }
                    hits.push(Summary {
                        repo: repo.to_owned(),
                        downloads: item["downloads"].as_u64().unwrap_or(0),
                        likes: item["likes"].as_u64().unwrap_or(0),
                        task: item["pipeline_tag"].as_str().map(str::to_owned),
                        license: license_of(&item["tags"]),
                        gguf: item["tags"]
                            .as_array()
                            .is_some_and(|t| t.iter().any(|t| t == "gguf")),
                    });
                }
            }
            // The merged lists keep the order of Hugging Face: most downloaded first.
            hits.sort_by_key(|h| std::cmp::Reverse(h.downloads));
            hits.truncate(query.limit);
            Ok(hits)
        })
    }

    fn detail<'a>(&'a self, repo: &'a str) -> HubFuture<'a, Detail> {
        Box::pin(self.detail_of(repo, None))
    }

    fn detail_at<'a>(&'a self, repo: &'a str, revision: &'a str) -> HubFuture<'a, Detail> {
        Box::pin(async move {
            if !valid_revision(revision) {
                return Err(HubError::NotFound);
            }
            self.detail_of(repo, Some(revision)).await
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
            if !valid_repo(repo) || !valid_revision(revision) || !valid_file(file) {
                return Err(HubError::NotFound);
            }
            let mut request = self
                .http
                .get(format!("{}/{repo}/resolve/{revision}/{file}", self.base));
            if offset > 0 {
                request = request.header("Range", format!("bytes={offset}-"));
            }
            if let Some(token) = token {
                request = request.bearer_auth(token.expose());
            }
            let response = request.send().await.map_err(|_| HubError::Unreachable)?;
            let resumed = match response.status().as_u16() {
                200 => false,
                206 => true,
                401 | 403 => return Err(HubError::Forbidden),
                404 => return Err(HubError::NotFound),
                429 => return Err(HubError::Busy),
                _ => return Err(HubError::BadResponse),
            };
            let remaining = response.content_length();
            Ok(FileStream {
                resumed,
                remaining,
                body: Box::new(Body(response)),
            })
        })
    }
}

struct Body(reqwest::Response);
impl ChunkSource for Body {
    fn next_chunk(&mut self) -> HubFuture<'_, Option<Vec<u8>>> {
        Box::pin(async move {
            self.0
                .chunk()
                .await
                .map(|chunk| chunk.map(|bytes| bytes.to_vec()))
                .map_err(|_| HubError::Unreachable)
        })
    }
}

impl HfClient {
    /// The detail page of a repository, at a commit or at the current head.
    async fn detail_of(&self, repo: &str, revision: Option<&str>) -> Result<Detail, HubError> {
        if !valid_repo(repo) {
            return Err(HubError::NotFound);
        }
        let url = match revision {
            Some(revision) => format!("{}/api/models/{repo}/revision/{revision}", self.base),
            None => format!("{}/api/models/{repo}", self.base),
        };
        let value = self.get(url, &[("blobs", "true".to_owned())]).await?;
        let sha = value["sha"].as_str().ok_or(HubError::BadResponse)?;
        let files = value["siblings"]
            .as_array()
            .ok_or(HubError::BadResponse)?
            .iter()
            .filter_map(|f| {
                Some(RepoFile {
                    name: f["rfilename"].as_str()?.to_owned(),
                    bytes: f["size"].as_u64().unwrap_or(0),
                    sha256: f["lfs"]["sha256"].as_str().map(str::to_owned),
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
    }
}

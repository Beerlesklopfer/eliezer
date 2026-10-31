//! `/api/v1/models/download` and `/api/v1/models/pins`: fetching models from Hugging Face into the
//! model store (Specification 9.3). Only repositories on `[huggingface] allow` are fetched, names are
//! checked before they become a path or a URL, and every file is verified against the listed hash.

use super::{App, failure};
use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use eliezer_core::config::DownloadMode;
use eliezer_hub::{
    Hub, HubError, RepoFile, Token,
    download::{Outcome, fetch},
    store::{self, PinFile},
    valid_file, valid_repo, valid_revision,
};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// Finished jobs kept for the status report.
const KEEP_JOBS: usize = 20;
const MAX_FILES: usize = 16;

pub struct JobFile {
    pub name: String,
    pub total: u64,
    /// Bytes of the file in place; written by the download, read by the status.
    pub done: Arc<AtomicU64>,
    pub state: &'static str,
    pub error: Option<String>,
}
pub struct Job {
    pub id: String,
    pub repo: String,
    pub revision: String,
    /// The directory in the model store the files go to.
    pub dir: String,
    pub state: &'static str,
    pub files: Vec<JobFile>,
    pub error: Option<String>,
    pub started_ms: u64,
    pub finished_ms: Option<u64>,
}
impl Job {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "repo": self.repo,
            "revision": self.revision,
            "directory": self.dir,
            "state": self.state,
            "error": self.error,
            "started_ms": self.started_ms,
            "finished_ms": self.finished_ms,
            "total_bytes": self.files.iter().map(|f| f.total).sum::<u64>(),
            "done_bytes": self.files.iter().map(|f| f.done.load(Ordering::Relaxed)).sum::<u64>(),
            "files": self.files.iter().map(|f| json!({
                "file": f.name,
                "total_bytes": f.total,
                "done_bytes": f.done.load(Ordering::Relaxed),
                "state": f.state,
                "error": f.error,
            })).collect::<Vec<_>>(),
        })
    }
}

/// The download jobs of the service. One job runs at a time; the others wait in order.
#[derive(Default)]
pub struct Downloads {
    jobs: Mutex<VecDeque<Job>>,
    next: AtomicU64,
    running: tokio::sync::Mutex<()>,
}
impl Downloads {
    fn insert(&self, job: Job) {
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.push_front(job);
            while jobs.len() > KEEP_JOBS {
                if let Some(at) = jobs
                    .iter()
                    .rposition(|j| j.state == "done" || j.state == "failed")
                {
                    jobs.remove(at);
                } else {
                    break;
                }
            }
        }
    }
    fn update(&self, id: &str, change: impl FnOnce(&mut Job)) {
        if let Ok(mut jobs) = self.jobs.lock()
            && let Some(job) = jobs.iter_mut().find(|j| j.id == id)
        {
            change(job);
        }
    }
    fn active(&self, repo: &str) -> bool {
        self.jobs.lock().is_ok_and(|jobs| {
            jobs.iter()
                .any(|j| j.repo == repo && (j.state == "queued" || j.state == "running"))
        })
    }
    /// The jobs as JSON, newest first.
    pub fn list(&self, limit: usize) -> Vec<Value> {
        self.jobs
            .lock()
            .map(|jobs| jobs.iter().take(limit).map(Job::to_json).collect())
            .unwrap_or_default()
    }
    fn get(&self, id: &str) -> Option<Value> {
        self.jobs
            .lock()
            .ok()
            .and_then(|jobs| jobs.iter().find(|j| j.id == id).map(Job::to_json))
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn refuse(status: StatusCode, code: &str, message: &str) -> Response {
    failure(status, code, message, false)
}

fn hub_refusal(error: HubError) -> Response {
    match error {
        HubError::Forbidden => refuse(
            StatusCode::FORBIDDEN,
            "license_required",
            &HubError::Forbidden.to_string(),
        ),
        HubError::NotFound => refuse(
            StatusCode::NOT_FOUND,
            "not_found",
            "The repository or commit was not found on Hugging Face",
        ),
        HubError::Unreachable => refuse(
            StatusCode::BAD_GATEWAY,
            "hub_unreachable",
            &error.to_string(),
        ),
        HubError::BadResponse => refuse(
            StatusCode::BAD_GATEWAY,
            "hub_bad_response",
            &error.to_string(),
        ),
        HubError::Busy => refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded",
            &error.to_string(),
        ),
    }
}

/// `POST /api/v1/models/download` with `{"repo", "files": [...], "revision"?}` → 202 and the job.
pub async fn start(
    State(app): State<App>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Ok(Json(body)) = body else {
        return refuse(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Invalid JSON request body",
        );
    };
    let config = app.config.get();
    if matches!(config.huggingface.downloads, DownloadMode::Never) {
        return refuse(
            StatusCode::FORBIDDEN,
            "downloads_disabled",
            "Downloads are switched off ([huggingface] downloads)",
        );
    }
    let Some(hub) = app.hub.clone() else {
        return refuse(
            StatusCode::FORBIDDEN,
            "downloads_disabled",
            "No connection to Hugging Face is configured",
        );
    };
    let Some(repo) = body["repo"].as_str().filter(|r| valid_repo(r)) else {
        return refuse(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "repo is written Organisation/Repository",
        );
    };
    if !config.huggingface.allows(repo) {
        return refuse(
            StatusCode::FORBIDDEN,
            "not_allowed",
            "This repository is not on the allowlist ([huggingface] allow)",
        );
    }
    let mut names: Vec<String> = Vec::new();
    match body["files"].as_array() {
        Some(list) if !list.is_empty() && list.len() <= MAX_FILES => {
            for item in list {
                match item.as_str().filter(|f| valid_file(f)) {
                    Some(name) if !names.iter().any(|n| n == name) => names.push(name.to_owned()),
                    Some(_) => {}
                    None => {
                        return refuse(
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            "A file name is not valid (relative, at most three levels, letters, digits and . _ - +)",
                        );
                    }
                }
            }
        }
        _ => {
            return refuse(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "files is a list of 1 to 16 file names",
            );
        }
    }
    let revision = match body.get("revision") {
        None | Some(Value::Null) => None,
        Some(Value::String(r)) if valid_revision(r) => Some(r.clone()),
        Some(_) => {
            return refuse(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "revision is a full commit hash (40 lowercase hex digits)",
            );
        }
    };
    let detail = match &revision {
        Some(r) => hub.detail_at(repo, r).await,
        None => hub.detail(repo).await,
    };
    let detail = match detail {
        Ok(detail) => detail,
        Err(error) => return hub_refusal(error),
    };
    let token = app.hf_token.get();
    let token: Option<Token> = (*token).clone();
    if detail.gated && token.is_none() {
        return hub_refusal(HubError::Forbidden);
    }
    let mut files: Vec<RepoFile> = Vec::new();
    for name in &names {
        match detail.files.iter().find(|f| f.name == *name) {
            Some(file) => files.push(file.clone()),
            None => {
                return refuse(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    &format!("The file {name} is not in the repository at this commit"),
                );
            }
        }
    }
    let root = config.storage.models_dir.clone();
    // What is already pinned with the same commit, name and size is not fetched again.
    let pinned = store::scan(&root);
    let known = |file: &RepoFile| {
        pinned
            .iter()
            .filter(|p| p.repo == repo)
            .flat_map(|p| p.files.iter())
            .any(|f| f.name == file.name && f.bytes == file.bytes && f.revision == detail.sha)
    };
    let new_bytes: u64 = files.iter().filter(|f| !known(f)).map(|f| f.bytes).sum();
    let limit =
        eliezer_core::memory::gb_bytes(config.huggingface.cache_limit_gb).unwrap_or(u64::MAX);
    let used = pinned
        .iter()
        .flat_map(|p| p.files.iter())
        .fold(0_u64, |sum, f| sum.saturating_add(f.bytes));
    if used.saturating_add(new_bytes) > limit {
        return refuse(
            StatusCode::INSUFFICIENT_STORAGE,
            "insufficient_storage",
            &format!(
                "The download needs {:.1} GB; {:.1} GB of the cache limit ({:.1} GB, [huggingface] cache_limit_gb) are used",
                new_bytes as f64 / 1e9,
                used as f64 / 1e9,
                limit as f64 / 1e9
            ),
        );
    }
    if app.downloads.active(repo) {
        return refuse(
            StatusCode::CONFLICT,
            "busy",
            "A download from this repository is already running or waiting",
        );
    }
    let id = format!(
        "dl-{}",
        app.downloads.next.fetch_add(1, Ordering::Relaxed) + 1
    );
    let job = Job {
        id: id.clone(),
        repo: repo.to_owned(),
        revision: detail.sha.clone(),
        dir: root.join(repo).display().to_string(),
        state: "queued",
        files: files
            .iter()
            .map(|f| JobFile {
                name: f.name.clone(),
                total: f.bytes,
                done: Arc::new(AtomicU64::new(0)),
                state: "queued",
                error: None,
            })
            .collect(),
        error: None,
        started_ms: now_ms(),
        finished_ms: None,
    };
    let answer = job.to_json();
    app.downloads.insert(job);
    tokio::spawn(run_job(
        app.downloads.clone(),
        hub,
        root,
        id,
        repo.to_owned(),
        detail.sha,
        files,
        token,
    ));
    (StatusCode::ACCEPTED, Json(answer)).into_response()
}

#[allow(clippy::too_many_arguments)]
async fn run_job(
    downloads: Arc<Downloads>,
    hub: Arc<dyn Hub>,
    root: std::path::PathBuf,
    id: String,
    repo: String,
    revision: String,
    files: Vec<RepoFile>,
    token: Option<Token>,
) {
    let _turn = downloads.running.lock().await;
    downloads.update(&id, |j| j.state = "running");
    tracing::info!(event = "download_start", repo = %repo, files = files.len(), "Download started");
    let mut pins: Vec<PinFile> = Vec::new();
    let mut failure_text: Option<String> = None;
    for (index, file) in files.iter().enumerate() {
        downloads.update(&id, |j| j.files[index].state = "running");
        let progress = downloads.jobs.lock().ok().and_then(|jobs| {
            jobs.iter()
                .find(|j| j.id == id)
                .map(|j| j.files[index].done.clone())
        });
        let target = {
            let (root, repo, name) = (root.clone(), repo.clone(), file.name.clone());
            tokio::task::spawn_blocking(move || store::target_path(&root, &repo, &name)).await
        };
        let result = match target {
            Ok(Ok(target)) => {
                let report = |done: u64| {
                    if let Some(counter) = &progress {
                        counter.store(done, Ordering::Relaxed);
                    }
                };
                fetch(
                    &*hub,
                    &repo,
                    &revision,
                    file,
                    &target,
                    token.as_ref(),
                    &report,
                )
                .await
                .map_err(|e| e.to_string())
            }
            Ok(Err(error)) => Err(error.to_string()),
            Err(_) => Err("The download task failed".to_owned()),
        };
        match result {
            Ok(outcome) => {
                downloads.update(&id, |j| {
                    j.files[index].state = if outcome == Outcome::AlreadyPresent {
                        "present"
                    } else {
                        "done"
                    };
                });
                pins.push(PinFile {
                    name: file.name.clone(),
                    bytes: file.bytes,
                    sha256: file.sha256.clone(),
                    revision: revision.clone(),
                    downloaded_at: now_ms() / 1000,
                });
            }
            Err(text) => {
                // A refusal of the hub becomes the license notice; other texts are already free of secrets.
                downloads.update(&id, |j| {
                    j.files[index].state = "failed";
                    j.files[index].error = Some(text.clone());
                });
                failure_text = Some(text);
                break;
            }
        }
    }
    if !pins.is_empty() {
        let (root, repo) = (root.clone(), repo.clone());
        let _ = tokio::task::spawn_blocking(move || store::record(&root, &repo, pins)).await;
    }
    let failed = failure_text.is_some();
    downloads.update(&id, |j| {
        j.state = if failed { "failed" } else { "done" };
        j.error = failure_text.clone();
        j.finished_ms = Some(now_ms());
    });
    tracing::info!(event = "download_end", repo = %repo, failed, "Download ended");
}

/// `GET /api/v1/models/download`: the jobs, newest first.
pub async fn list(State(app): State<App>) -> Response {
    Json(json!({"object": "list", "data": app.downloads.list(KEEP_JOBS)})).into_response()
}

/// `GET /api/v1/models/download/{id}`.
pub async fn job(State(app): State<App>, Path(id): Path<String>) -> Response {
    match app.downloads.get(&id) {
        Some(job) => Json(job).into_response(),
        None => refuse(StatusCode::NOT_FOUND, "not_found", "No such download"),
    }
}

/// `GET /api/v1/models/pins[?verify=true]`: what was fetched and whether the files still match.
pub async fn pins(
    State(app): State<App>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let verify = match params.get("verify").map(String::as_str) {
        None | Some("false") => false,
        Some("true") => true,
        Some(_) => {
            return refuse(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "verify is true or false",
            );
        }
    };
    let root = app.config.get().storage.models_dir.clone();
    let report = tokio::task::spawn_blocking(move || report(&root, verify)).await;
    match report {
        Ok(report) => Json(report).into_response(),
        Err(_) => refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "backend_error",
            "The check failed",
        ),
    }
}

/// The pin report of the store; blocking (with `verify` it hashes every pinned file).
pub fn report(root: &std::path::Path, verify: bool) -> Value {
    let mut deviations = 0_usize;
    let pins: Vec<Value> = store::scan(root)
        .iter()
        .map(|pin| {
            let files: Vec<Value> = store::check(root, pin, verify)
                .into_iter()
                .map(|(file, problem)| {
                    if problem.is_some() {
                        deviations += 1;
                    }
                    json!({
                        "file": file.name,
                        "bytes": file.bytes,
                        "revision": file.revision,
                        "sha256": file.sha256,
                        "status": problem.as_ref().map_or("ok", |p| p.name()),
                        "found_bytes": match problem {
                            Some(store::Problem::SizeMismatch { found }) => Some(found),
                            _ => None,
                        },
                    })
                })
                .collect();
            json!({"repo": pin.repo, "files": files})
        })
        .collect();
    json!({"pins": pins, "verified": verify, "deviations": deviations})
}

/// `DELETE /api/v1/models/pins/{org}/{repo}`: removes a repository fetched earlier from the model store.
/// Refused while a download of it is active or a configured model still uses one of its files.
pub async fn remove(State(app): State<App>, Path((org, name)): Path<(String, String)>) -> Response {
    let repo = format!("{org}/{name}");
    if !valid_repo(&repo) {
        return refuse(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "The repository must be written ORG/REPO",
        );
    }
    if app.downloads.active(&repo) {
        return refuse(
            StatusCode::CONFLICT,
            "download_active",
            "A download of this repository is running or queued",
        );
    }
    let config = app.config.get();
    let root = config.storage.models_dir.clone();
    let Ok(dir) = store::repo_dir(&root, &repo) else {
        return refuse(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Invalid repository",
        );
    };
    let dir = dir.canonicalize().unwrap_or(dir);
    let uses = |path: &Option<std::path::PathBuf>| {
        path.as_ref().is_some_and(|p| {
            p.canonicalize()
                .unwrap_or_else(|_| p.clone())
                .starts_with(&dir)
        })
    };
    let users: Vec<&str> = config
        .model
        .iter()
        .filter(|m| {
            let extra = m.extra.as_ref();
            uses(&m.source.file)
                || uses(&m.source.projector)
                || uses(&extra.and_then(|e| e.text_encoder.clone()))
                || uses(&extra.and_then(|e| e.vae.clone()))
                || uses(&extra.and_then(|e| e.tokenizer.clone()))
        })
        .map(|m| m.key.as_str())
        .collect();
    if !users.is_empty() {
        return refuse(
            StatusCode::CONFLICT,
            "in_use",
            &format!(
                "The models {} use files of this repository; remove them from the configuration first",
                users.join(", ")
            ),
        );
    }
    let removed =
        tokio::task::spawn_blocking(move || store::remove(&root, &repo).map(|b| (repo, b))).await;
    match removed {
        Ok(Ok((repo, bytes))) => Json(json!({"repo": repo, "freed_bytes": bytes})).into_response(),
        Ok(Err(store::StoreError::Invalid)) => refuse(
            StatusCode::NOT_FOUND,
            "not_found",
            "No repository fetched with `eliezer download` has this name",
        ),
        _ => refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "backend_error",
            "Deleting the files failed",
        ),
    }
}

//! `/api/v1/hub/...`: the search of Hugging Face for model candidates (Specification 9.1).
//! Read only: nothing is downloaded and the allowlist does not change.

use super::{App, failure};
use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use eliezer_hub::{
    Assessment, Budgets, Detail, Format, Hub, HubError, Query as HubQuery, Task, assess, valid_repo,
};
use serde_json::{Value, json};
use std::{collections::HashMap, sync::Arc};

const DEFAULT_LIMIT: usize = 10;
const MAX_LIMIT: usize = 20;
/// Details cost one request to Hugging Face per hit, so they are limited further.
const MAX_DETAILED: usize = 10;

fn hub_error(error: HubError) -> Response {
    let (status, code) = match error {
        HubError::Unreachable => (StatusCode::BAD_GATEWAY, "hub_unreachable"),
        HubError::NotFound => (StatusCode::NOT_FOUND, "not_found"),
        HubError::BadResponse => (StatusCode::BAD_GATEWAY, "hub_bad_response"),
        HubError::Busy => (StatusCode::SERVICE_UNAVAILABLE, "overloaded"),
        HubError::Forbidden => (StatusCode::FORBIDDEN, "license_required"),
    };
    failure(status, code, &error.to_string(), false)
}

/// The hub, unless the search is switched off in the configuration.
fn hub_of(app: &App) -> Result<Arc<dyn Hub>, Box<Response>> {
    match &app.hub {
        Some(hub) if app.config.get().huggingface.search => Ok(hub.clone()),
        _ => Err(Box::new(failure(
            StatusCode::FORBIDDEN,
            "hub_disabled",
            "The Hugging Face search is switched off ([huggingface] search)",
            false,
        ))),
    }
}

fn budgets(app: &App) -> Budgets {
    let config = app.config.get();
    let usable = |gb: f64, reserve: f64| {
        eliezer_core::memory::gb_bytes(gb)
            .and_then(|b| eliezer_core::memory::gb_bytes(reserve).map(|r| b.saturating_sub(r)))
            .unwrap_or(0)
    };
    Budgets {
        gpu: usable(config.gpu.vram_budget_gb, config.gpu.reserve_gb),
        ram: if config.cpu.ram_budget_gb > 0.0 {
            usable(config.cpu.ram_budget_gb, config.cpu.reserve_gb)
        } else {
            0
        },
    }
}

fn detail_json(detail: &Detail, assessment: &Assessment) -> Value {
    let file = |f: &eliezer_hub::RepoFile| json!({"file": f.name, "bytes": f.bytes});
    json!({
        "repo": detail.repo,
        "sha": detail.sha,
        "task": detail.task,
        "license": detail.license,
        "gated": detail.gated,
        "downloads": detail.downloads,
        "architecture": detail.architecture,
        "support": {"level": assessment.support.name(), "backend": assessment.backend},
        "variants": assessment.variants.iter().map(|v| json!({
            "file": v.name,
            "quantization": v.quantization,
            "bytes": v.bytes,
            "parts": v.parts,
            "files": v.files,
            "estimate_bytes": v.estimate_bytes,
            "fit": v.fit.name(),
        })).collect::<Vec<_>>(),
        "projector": assessment.projector.as_ref().map(file),
        "required_files": assessment.required.iter().map(file).collect::<Vec<_>>(),
    })
}

fn invalid(message: &str) -> Response {
    failure(StatusCode::BAD_REQUEST, "invalid_request", message, false)
}

/// `GET /api/v1/hub/search?q=&task=&format=&license=&limit=&details=`
pub async fn search(
    State(app): State<App>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let hub = match hub_of(&app) {
        Ok(hub) => hub,
        Err(response) => return *response,
    };
    let text = params
        .get("q")
        .map(String::as_str)
        .unwrap_or_default()
        .trim();
    if text.len() > 100 || text.chars().any(char::is_control) {
        return invalid("q is at most 100 characters without control characters");
    }
    let task = match params.get("task").map(String::as_str) {
        None => Task::Any,
        Some(t) => match Task::parse(t) {
            Some(task) => task,
            None => return invalid("task is one of: any, llm, vision, embedding, image"),
        },
    };
    let format = match params.get("format").map(String::as_str) {
        None => Format::Gguf,
        Some(f) => match Format::parse(f) {
            Some(format) => format,
            None => return invalid("format is gguf or safetensors"),
        },
    };
    let license = params.get("license").cloned();
    if license.as_ref().is_some_and(|l| {
        l.is_empty()
            || l.len() > 40
            || !l
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
    }) {
        return invalid("license is a license tag such as apache-2.0");
    }
    if text.is_empty() && task == Task::Any {
        return invalid("Give a search text (q) or a task");
    }
    let limit = match params.get("limit") {
        None => DEFAULT_LIMIT,
        Some(l) => match l.parse::<usize>() {
            Ok(n) if (1..=MAX_LIMIT).contains(&n) => n,
            _ => return invalid("limit is a number from 1 to 20"),
        },
    };
    let details = match params.get("details").map(String::as_str) {
        None | Some("false") => false,
        Some("true") => true,
        Some(_) => return invalid("details is true or false"),
    };
    let query = HubQuery {
        text: text.to_owned(),
        task,
        format,
        license,
        limit,
    };
    let hits = match hub.search(&query).await {
        Ok(hits) => hits,
        Err(error) => return hub_error(error),
    };
    let budgets = budgets(&app);
    let mut tasks = tokio::task::JoinSet::new();
    if details {
        for (index, hit) in hits.iter().take(MAX_DETAILED).enumerate() {
            let (hub, repo) = (hub.clone(), hit.repo.clone());
            tasks.spawn(async move { (index, hub.detail(&repo).await) });
        }
    }
    let mut detailed: HashMap<usize, Value> = HashMap::new();
    while let Some(Ok((index, result))) = tasks.join_next().await {
        // A repository that cannot be opened stays a plain hit; the search itself worked.
        if let Ok(detail) = result {
            detailed.insert(index, detail_json(&detail, &assess(&detail, budgets)));
        }
    }
    let data: Vec<Value> = hits
        .iter()
        .enumerate()
        .map(|(index, hit)| {
            json!({
                "repo": hit.repo,
                "downloads": hit.downloads,
                "likes": hit.likes,
                "task": hit.task,
                "license": hit.license,
                "gguf": hit.gguf,
                "detail": detailed.remove(&index),
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data, "budgets": {
        "gpu_bytes": budgets.gpu, "ram_bytes": budgets.ram}}))
    .into_response()
}

/// `GET /api/v1/hub/models/{org}/{repo}`: files, support and fit of one repository.
pub async fn detail(State(app): State<App>, Path((org, name)): Path<(String, String)>) -> Response {
    let hub = match hub_of(&app) {
        Ok(hub) => hub,
        Err(response) => return *response,
    };
    let repo = format!("{org}/{name}");
    if !valid_repo(&repo) {
        return invalid("The repository is written Organisation/Repository");
    }
    match hub.detail(&repo).await {
        Ok(detail) => {
            let assessment = assess(&detail, budgets(&app));
            Json(detail_json(&detail, &assessment)).into_response()
        }
        Err(error) => hub_error(error),
    }
}

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use eliezer_api::{App, managed_router};
use eliezer_backend::{supervisor::*, *};
use eliezer_core::{config::Config, scheduler::Clock};
use eliezer_hub::{Detail, FakeHub, HubError, RepoFile, Token, store};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tower::ServiceExt;

struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "eliezer-dlapi-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(std::fs::canonicalize(path).unwrap())
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const REPO: &str = "org/model-GGUF";
const REV: &str = "0123456789abcdef0123456789abcdef01234567";
const SECRET: &str = "hf_SECRETSECRETSECRET0123456789";

fn data(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 13 % 251) as u8).collect()
}
fn file(name: &str, data: &[u8]) -> RepoFile {
    RepoFile {
        name: name.into(),
        bytes: data.len() as u64,
        sha256: Some(store::sha256_hex(Sha256::digest(data))),
    }
}
fn fake(gated: bool) -> FakeHub {
    let (a, b) = (data(30_000), data(7_000));
    let mut hub = FakeHub {
        chunk_size: 200,
        details: vec![Detail {
            repo: REPO.into(),
            sha: REV.into(),
            task: Some("text-generation".into()),
            license: Some("apache-2.0".into()),
            gated,
            downloads: 1,
            architecture: Some("llama".into()),
            files: vec![file("m-Q4_K_M.gguf", &a), file("vae/v.safetensors", &b)],
        }],
        ..FakeHub::default()
    };
    hub.contents
        .insert((REPO.into(), "m-Q4_K_M.gguf".into()), a);
    hub.contents
        .insert((REPO.into(), "vae/v.safetensors".into()), b);
    hub
}

struct Env {
    app: Router,
    handle: App,
    dir: Temp,
    hub: Arc<FakeHub>,
}
fn env_with(extra: &str, hub: FakeHub) -> Env {
    let dir = Temp::new();
    let config = Arc::new(
        Config::parse(&format!(
            r#"
[server]
listen="127.0.0.1:1234"
[storage]
models_dir="{}"
[gpu]
vendor="none"
measure="none"
vram_budget_gb=12.0
reserve_gb=1.0
[huggingface]
cache_dir="{}/hf"
{extra}
[backend.mistralrs]
enabled=true
"#,
            dir.0.display(),
            dir.0.display()
        ))
        .unwrap(),
    );
    let supervisor = Supervisor::spawn(
        vec![],
        Arc::new(FakeBackend::new(Default::default())),
        Arc::new(TestClock),
        Settings {
            budget: 10_000_000_000,
            ram_budget: 0,
            idle_ms: 1000,
            wait_ms: 100,
            queue_max: 4,
            retry_ms: 30,
        },
    );
    let backend: Arc<dyn Backend> = Arc::new(FakeBackend::new(Default::default()));
    let hub = Arc::new(hub);
    let handle = App::new(
        config,
        supervisor,
        backend,
        Arc::new(eliezer_gpu::sources(
            eliezer_core::config::GpuVendor::None,
            eliezer_core::config::ProbeMethod::None,
            0,
            8_000_000_000,
            "/nonexistent".into(),
        )),
        vec![],
    )
    .with_hub(hub.clone());
    Env {
        app: managed_router(handle.clone()),
        handle,
        dir,
        hub,
    }
}
fn env(hub: FakeHub) -> Env {
    env_with("allow = [\"org/*\"]", hub)
}

async fn send(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}
fn download(files: &[&str]) -> Value {
    json!({"repo": REPO, "files": files})
}
/// Polls the job until it is done or failed.
async fn finish(app: &Router, id: &str) -> Value {
    for _ in 0..2000 {
        let (_, job) = send(app, "GET", &format!("/api/v1/models/download/{id}"), None).await;
        if job["state"] == "done" || job["state"] == "failed" {
            return job;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("the download did not finish");
}

#[tokio::test]
async fn a_download_lands_in_the_store_with_a_pin() {
    let e = env(fake(false));
    let (status, job) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf", "vae/v.safetensors"])),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{job}");
    assert_eq!(job["repo"], REPO);
    assert_eq!(job["revision"], REV);
    assert_eq!(job["total_bytes"], 37_000);
    let id = job["id"].as_str().unwrap().to_owned();
    let done = finish(&e.app, &id).await;
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["done_bytes"], 37_000);
    assert!(
        done["files"]
            .as_array()
            .unwrap()
            .iter()
            .all(|f| f["state"] == "done")
    );
    let root = &e.dir.0;
    assert_eq!(
        std::fs::read(root.join(REPO).join("m-Q4_K_M.gguf")).unwrap(),
        data(30_000)
    );
    assert_eq!(
        std::fs::read(root.join(REPO).join("vae/v.safetensors")).unwrap(),
        data(7_000)
    );
    assert_eq!(done["directory"], root.join(REPO).display().to_string());
    // the pin and the report
    let (_, pins) = send(&e.app, "GET", "/api/v1/models/pins", None).await;
    assert_eq!(pins["deviations"], 0);
    assert_eq!(pins["pins"][0]["repo"], REPO);
    assert_eq!(pins["pins"][0]["files"][0]["revision"], REV);
    assert_eq!(pins["pins"][0]["files"][0]["status"], "ok");
    // the status lists it, and the list endpoint too
    let (_, status) = send(&e.app, "GET", "/api/v1/status", None).await;
    assert_eq!(status["downloads"][0]["id"], id.as_str());
    let (_, list) = send(&e.app, "GET", "/api/v1/models/download", None).await;
    assert_eq!(list["data"][0]["state"], "done");
    // the same download again finds the files and fetches nothing
    let opened = e.hub.opened.lock().unwrap().len();
    let (_, again) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    let done = finish(&e.app, again["id"].as_str().unwrap()).await;
    assert_eq!(done["files"][0]["state"], "present");
    assert_eq!(e.hub.opened.lock().unwrap().len(), opened);
}

#[tokio::test]
async fn the_gates_come_first_and_every_input_is_checked() {
    let off = env_with("downloads = \"never\"\nallow = [\"org/*\"]", fake(false));
    let (status, body) = send(
        &off.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    assert_eq!(
        (status, body["error"]["type"].as_str()),
        (StatusCode::FORBIDDEN, Some("downloads_disabled"))
    );
    let none = env_with("", fake(false));
    let (status, body) = send(
        &none.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    assert_eq!(
        (status, body["error"]["type"].as_str()),
        (StatusCode::FORBIDDEN, Some("not_allowed"))
    );
    assert!(
        std::fs::read_dir(&none.dir.0).unwrap().next().is_none(),
        "nothing was created"
    );
    let other = env_with("allow = [\"someone/else\"]", fake(false));
    assert_eq!(
        send(
            &other.app,
            "POST",
            "/api/v1/models/download",
            Some(download(&["m-Q4_K_M.gguf"]))
        )
        .await
        .0,
        StatusCode::FORBIDDEN
    );

    let e = env(fake(false));
    let post = |body: Value| {
        let app = e.app.clone();
        async move { send(&app, "POST", "/api/v1/models/download", Some(body)).await }
    };
    for body in [
        json!({"files": ["m-Q4_K_M.gguf"]}),
        json!({"repo": "org", "files": ["a"]}),
        json!({"repo": "org/../x", "files": ["a"]}),
        json!({"repo": REPO}),
        json!({"repo": REPO, "files": []}),
        json!({"repo": REPO, "files": "m-Q4_K_M.gguf"}),
        json!({"repo": REPO, "files": ["../escape.gguf"]}),
        json!({"repo": REPO, "files": ["/etc/passwd"]}),
        json!({"repo": REPO, "files": [".hidden"]}),
        json!({"repo": REPO, "files": ["a/b/c/d.gguf"]}),
        json!({"repo": REPO, "files": [1]}),
        json!({"repo": REPO, "files": ["nope.gguf"]}),
        json!({"repo": REPO, "files": (0..17).map(|i| format!("f{i}.gguf")).collect::<Vec<_>>()}),
        json!({"repo": REPO, "files": ["m-Q4_K_M.gguf"], "revision": "main"}),
        json!({"repo": REPO, "files": ["m-Q4_K_M.gguf"], "revision": 5}),
    ] {
        let (status, answer) = post(body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} -> {answer}");
    }
    // a commit the hub does not know
    let (status, _) =
        post(json!({"repo": REPO, "files": ["m-Q4_K_M.gguf"], "revision": "f".repeat(40)})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    // a commit it knows works
    let (status, _) =
        post(json!({"repo": REPO, "files": ["m-Q4_K_M.gguf"], "revision": REV})).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    // nothing outside the repository directory was written
    let root_entries: Vec<_> = std::fs::read_dir(&e.dir.0)
        .unwrap()
        .flatten()
        .map(|d| d.file_name())
        .collect();
    assert!(
        root_entries.iter().all(|n| n == "org" || n == "hf"),
        "{root_entries:?}"
    );
}

#[tokio::test]
async fn the_cache_limit_answers_507_and_a_running_job_blocks_the_repository() {
    let small = env_with("allow = [\"org/*\"]\ncache_limit_gb = 0.00002", fake(false));
    let (status, body) = send(
        &small.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    assert_eq!(status, StatusCode::INSUFFICIENT_STORAGE, "{body}");
    assert_eq!(body["error"]["type"], "insufficient_storage");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cache_limit_gb")
    );

    let mut slow = fake(false);
    slow.chunk_size = 1;
    let e = env(slow);
    let (status, first) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["vae/v.safetensors"])),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let (status, body) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["vae/v.safetensors"])),
    )
    .await;
    assert_eq!(
        (status, body["error"]["type"].as_str()),
        (StatusCode::CONFLICT, Some("busy"))
    );
    assert_eq!(
        finish(&e.app, first["id"].as_str().unwrap()).await["state"],
        "done"
    );
}

#[tokio::test]
async fn a_gated_repository_needs_the_token_and_the_token_never_shows() {
    let e = env(fake(true));
    let (status, body) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    assert_eq!(
        (status, body["error"]["type"].as_str()),
        (StatusCode::FORBIDDEN, Some("license_required"))
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("license")
    );
    e.handle.set_hf_token(Token::new(SECRET));
    let (status, job) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let done = finish(&e.app, job["id"].as_str().unwrap()).await;
    assert_eq!(done["state"], "done");
    assert!(
        e.hub.opened.lock().unwrap().iter().all(|o| o.4),
        "the token went to the hub"
    );
    // no answer of the service contains it
    for uri in [
        "/api/v1/status",
        "/api/v1/models/download",
        "/api/v1/models/pins",
        "/api/v1/models",
    ] {
        let (_, body) = send(&e.app, "GET", uri, None).await;
        assert!(!body.to_string().contains("SECRET"), "{uri}");
    }
    assert!(!done.to_string().contains("SECRET") && !job.to_string().contains("SECRET"));
}

#[tokio::test]
async fn failures_are_reported_in_the_job_and_leave_nothing_wrong_behind() {
    // the connection breaks: the job fails, the partial file stays for a later try
    let mut broken = fake(false);
    broken.break_after = Some(5_000);
    let e = env(broken);
    let (_, job) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    let done = finish(&e.app, job["id"].as_str().unwrap()).await;
    assert_eq!(done["state"], "failed");
    assert!(
        done["error"]
            .as_str()
            .unwrap()
            .contains("cannot be reached"),
        "{done}"
    );
    assert!(!e.dir.0.join(REPO).join("m-Q4_K_M.gguf").exists());
    assert!(e.dir.0.join(REPO).join("m-Q4_K_M.gguf.part").exists());
    assert_eq!(
        send(&e.app, "GET", "/api/v1/models/pins", None).await.1["pins"]
            .as_array()
            .unwrap()
            .len(),
        0
    );

    // a wrong checksum: the file is discarded
    let mut wrong = fake(false);
    wrong.details[0].files[0].sha256 = Some("0".repeat(64));
    let e = env(wrong);
    let (_, job) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    let done = finish(&e.app, job["id"].as_str().unwrap()).await;
    assert_eq!(done["state"], "failed");
    assert!(
        done["error"].as_str().unwrap().contains("checksum"),
        "{done}"
    );
    assert!(!e.dir.0.join(REPO).join("m-Q4_K_M.gguf").exists());
    assert!(!e.dir.0.join(REPO).join("m-Q4_K_M.gguf.part").exists());

    // the hub is down for the listing
    let mut down = fake(false);
    down.fail = Some(HubError::Unreachable);
    let e = env(down);
    let (status, body) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf"])),
    )
    .await;
    assert_eq!(
        (status, body["error"]["type"].as_str()),
        (StatusCode::BAD_GATEWAY, Some("hub_unreachable"))
    );
}

#[tokio::test]
async fn the_pin_report_finds_deviations_and_verify_finds_changed_content() {
    let e = env(fake(false));
    let (_, job) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf", "vae/v.safetensors"])),
    )
    .await;
    finish(&e.app, job["id"].as_str().unwrap()).await;
    let dir = e.dir.0.join(REPO);
    std::fs::write(dir.join("m-Q4_K_M.gguf"), vec![1_u8; 30_000]).unwrap();
    let (_, quick) = send(&e.app, "GET", "/api/v1/models/pins", None).await;
    assert_eq!(quick["deviations"], 0, "same size: only the hash shows it");
    let (_, full) = send(&e.app, "GET", "/api/v1/models/pins?verify=true", None).await;
    assert_eq!(full["deviations"], 1);
    assert_eq!(full["verified"], true);
    let status_of = |report: &Value, name: &str| {
        report["pins"][0]["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["file"] == name)
            .unwrap()["status"]
            .clone()
    };
    assert_eq!(status_of(&full, "m-Q4_K_M.gguf"), "modified");
    std::fs::remove_file(dir.join("vae/v.safetensors")).unwrap();
    std::fs::write(dir.join("m-Q4_K_M.gguf"), b"short").unwrap();
    let (_, report) = send(&e.app, "GET", "/api/v1/models/pins", None).await;
    assert_eq!(status_of(&report, "vae/v.safetensors"), "missing");
    assert_eq!(status_of(&report, "m-Q4_K_M.gguf"), "size_mismatch");
    assert_eq!(report["deviations"], 2);
    assert_eq!(
        send(&e.app, "GET", "/api/v1/models/pins?verify=maybe", None)
            .await
            .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        send(&e.app, "GET", "/api/v1/models/download/dl-999", None)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn a_fetched_repository_can_be_deleted_and_nothing_else() {
    let e = env(fake(false));
    let (_, job) = send(
        &e.app,
        "POST",
        "/api/v1/models/download",
        Some(download(&["m-Q4_K_M.gguf", "vae/v.safetensors"])),
    )
    .await;
    finish(&e.app, job["id"].as_str().unwrap()).await;
    let dir = e.dir.0.join(REPO);
    assert!(dir.is_dir());
    // a directory without a pin is not ours to delete
    std::fs::create_dir_all(e.dir.0.join("other/handmade")).unwrap();
    let (status, body) = send(&e.app, "DELETE", "/api/v1/models/pins/other/handmade", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(e.dir.0.join("other/handmade").is_dir());
    // a bad name never reaches the file system
    let (status, _) = send(&e.app, "DELETE", "/api/v1/models/pins/org/..", None).await;
    assert!(status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND);
    assert!(e.dir.0.is_dir());
    // the real one
    let (status, body) = send(
        &e.app,
        "DELETE",
        &format!("/api/v1/models/pins/{REPO}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["freed_bytes"].as_u64().unwrap() >= 37_000);
    assert!(!dir.exists());
    let (_, pins) = send(&e.app, "GET", "/api/v1/models/pins", None).await;
    assert_eq!(pins["pins"].as_array().unwrap().len(), 0);
    let (status, _) = send(
        &e.app,
        "DELETE",
        &format!("/api/v1/models/pins/{REPO}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

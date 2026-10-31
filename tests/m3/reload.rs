//! End-to-end check of configuration reload: the real binary, real HTTP over loopback, real SIGHUP.
use serde_json::Value;
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
fn get(port: u16, path: &str, key: Option<&str>) -> Option<(u16, Value)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    let auth = key.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n"
    )
    .ok()?;
    let mut text = String::new();
    stream.read_to_string(&mut text).ok()?;
    let status = text.split(' ').nth(1)?.parse().ok()?;
    let body = text.split_once("\r\n\r\n")?.1;
    Some((status, serde_json::from_str(body).unwrap_or(Value::Null)))
}
fn model_keys(port: u16, key: &str) -> Option<Vec<String>> {
    let (status, body) = get(port, "/api/v1/models", Some(key))?;
    (status == 200).then(|| {
        body["models"]
            .as_array()
            .map(|models| {
                models
                    .iter()
                    .filter_map(|m| m["key"].as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    })
}
fn wait_for(what: &str, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(8);
    while !check() {
        assert!(Instant::now() < deadline, "timed out waiting for: {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}
fn config(dir: &Path, port: u16, keys: &Path, models: &[&str]) -> String {
    let mut text = format!(
        "[server]\nlisten=\"127.0.0.1:{port}\"\napi_keys_file=\"{}\"\n[storage]\nmodels_dir=\"{d}/models\"\n[huggingface]\ncache_dir=\"{d}/models/hf\"\n[gpu]\nvendor=\"none\"\nmeasure=\"none\"\nvram_budget_gb=8.0\nreserve_gb=1.0\n[backend.mistralrs]\nenabled=true\n",
        keys.display(),
        d = dir.display()
    );
    for key in models {
        text += &format!(
            "[[model]]\nkey=\"{key}\"\nbackend=\"mistralrs\"\nsource={{hf=\"t/{key}\"}}\nkind=\"llm\"\nvram_gb=1.0\n"
        );
    }
    text
}
fn write_keys(path: &Path, key: &str) {
    fs::write(path, format!("{key}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn hup(child: &Child) {
    let status = Command::new("kill")
        .args(["-HUP", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn sighup_applies_models_and_keys_and_keeps_state_on_invalid_files() {
    let dir = std::env::temp_dir().join(format!("eliezer-reload-{}", std::process::id()));
    fs::create_dir_all(dir.join("models")).unwrap();
    let (port, keys, path) = (free_port(), dir.join("keys"), dir.join("config.toml"));
    write_keys(&keys, "key-1");
    fs::write(&path, config(&dir, port, &keys, &["a"])).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .arg("--config")
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for("service ready", || get(port, "/healthz", None).is_some());
    assert_eq!(model_keys(port, "key-1"), Some(vec!["a".into()]));
    assert_eq!(get(port, "/api/v1/models", None).map(|r| r.0), Some(401));

    // a new model and a new key take effect; the old key stops working
    write_keys(&keys, "key-2");
    fs::write(&path, config(&dir, port, &keys, &["a", "b"])).unwrap();
    hup(&child);
    wait_for("model b appears", || {
        model_keys(port, "key-2") == Some(vec!["a".into(), "b".into()])
    });
    assert_eq!(
        get(port, "/api/v1/models", Some("key-1")).map(|r| r.0),
        Some(401)
    );

    // a removed model disappears and is refused
    fs::write(&path, config(&dir, port, &keys, &["b"])).unwrap();
    hup(&child);
    wait_for("model a disappears", || {
        model_keys(port, "key-2") == Some(vec!["b".into()])
    });
    let (_, status) = get(port, "/api/v1/status", Some("key-2")).unwrap();
    assert!(
        status["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["key"] != "a" || m["state"] == "unloaded")
    );

    // an invalid file is rejected and the service keeps running with the old state
    fs::write(&path, "[server]\nlisten=\"CONFIDENTIAL_VALUE\"\n").unwrap();
    hup(&child);
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        child.try_wait().unwrap().is_none(),
        "service must survive a bad reload"
    );
    assert_eq!(model_keys(port, "key-2"), Some(vec!["b".into()]));

    // a key file that is too open is rejected the same way
    fs::write(&path, config(&dir, port, &keys, &["b", "c"])).unwrap();
    fs::set_permissions(&keys, fs::Permissions::from_mode(0o644)).unwrap();
    hup(&child);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(model_keys(port, "key-2"), Some(vec!["b".into()]));

    child.kill().unwrap();
    child.wait().unwrap();
    fs::remove_dir_all(dir).unwrap();
}

use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "eliezer-m2-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn config(&self, body: &str) -> PathBuf {
        let path = self.0.join("config.toml");
        fs::write(&path, body).unwrap();
        path
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .args(args)
        .output()
        .unwrap()
}
fn minimal() -> String {
    "[server]\nlisten='127.0.0.1:1234'\n[gpu]\n".into()
}

#[test]
fn cli_checks_example_and_reports_safe_errors() {
    let config =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packaging/config.example.toml");
    let result = run(&["--check-config", config.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("in Ordnung"));
    let temp = Temp::new();
    let path = temp.config(&minimal().replace("127.0.0.1:1234", "VERTRAULICHER_WERT"));
    let result = run(&["--config", path.to_str().unwrap(), "--check-config"]);
    assert_eq!(result.status.code(), Some(2));
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(
        error.contains("Zeile 2") && error.contains("server.listen"),
        "{error}"
    );
    assert!(!error.contains("VERTRAULICHER_WERT"));
}

#[test]
fn explicit_config_overrides_environment_and_bad_arguments_fail() {
    let temp = Temp::new();
    let path = temp.config(&minimal());
    let result = Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env("ELIEZER_CONFIG", "/nicht/vorhanden")
        .args(["--config", path.to_str().unwrap(), "--check-config"])
        .output()
        .unwrap();
    assert!(result.status.success());
    for args in [
        &["--config"][..],
        &["--unbekannt"],
        &["--check-config", "--check-config"],
    ] {
        assert_eq!(run(args).status.code(), Some(2));
    }
}

#[test]
fn real_symlink_escape_and_dangling_symlinks_are_rejected() {
    let temp = Temp::new();
    let models = temp.0.join("models");
    let outside = temp.0.join("outside");
    fs::create_dir(&models).unwrap();
    fs::create_dir(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, models.join("escape")).unwrap();
    let input = format!(
        "{}\n[storage]\nmodels_dir={:?}\n[huggingface]\ncache_dir={:?}\n",
        minimal(),
        models,
        models.join("escape/cache")
    );
    let path = temp.config(&input);
    let output = run(&["--check-config", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("huggingface.cache_dir"));
    fs::remove_file(models.join("escape")).unwrap();
    std::os::unix::fs::symlink(outside.join("fehlt"), models.join("escape")).unwrap();
    assert_eq!(
        run(&["--check-config", path.to_str().unwrap()])
            .status
            .code(),
        Some(2)
    );
}

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
    assert!(String::from_utf8_lossy(&result.stdout).contains("Configuration OK"));
    let temp = Temp::new();
    let path = temp.config(&minimal().replace("127.0.0.1:1234", "CONFIDENTIAL_VALUE"));
    let result = run(&["--config", path.to_str().unwrap(), "--check-config"]);
    assert_eq!(result.status.code(), Some(2));
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(
        error.contains("line 2") && error.contains("server.listen"),
        "{error}"
    );
    assert!(!error.contains("CONFIDENTIAL_VALUE"));
}

#[test]
fn explicit_config_overrides_environment_and_bad_arguments_fail() {
    let temp = Temp::new();
    let path = temp.config(&minimal());
    let result = Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env("ELIEZER_CONFIG", "/does/not/exist")
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

#[test]
fn check_config_warns_about_moved_files_but_still_succeeds() {
    let temp = Temp::new();
    let models = temp.0.join("models");
    fs::create_dir(&models).unwrap();
    let missing = models.join("moved-away.gguf");
    let input = format!(
        "{}\n[storage]\nmodels_dir={:?}\n[huggingface]\ncache_dir={:?}\n[backend.llamacpp]\nenabled=true\n[[model]]\nkey=\"m\"\nbackend=\"llamacpp\"\nsource={{ file={:?} }}\nkind=\"llm\"\nvram_gb=1.0\n",
        minimal(),
        models,
        models.join("hf"),
        missing
    );
    let path = temp.config(&input);
    let output = run(&["--check-config", path.to_str().unwrap()]);
    let (out, err) = (
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(output.status.success(), "{err}");
    assert!(
        err.contains("Warning:") && err.contains("moved-away.gguf") && err.contains("not found"),
        "{err}"
    );
    assert!(
        out.contains("Configuration OK") && out.contains("1 file problem"),
        "{out}"
    );
}

#[test]
fn a_link_that_leaves_the_store_is_refused_with_advice() {
    let temp = Temp::new();
    let models = temp.0.join("models");
    let elsewhere = temp.0.join("elsewhere");
    fs::create_dir(&models).unwrap();
    fs::create_dir(&elsewhere).unwrap();
    fs::write(elsewhere.join("m.gguf"), b"x").unwrap();
    std::os::unix::fs::symlink(&elsewhere, models.join("out")).unwrap();
    let input = format!(
        "{}\n[storage]\nmodels_dir={:?}\n[huggingface]\ncache_dir={:?}\n[backend.llamacpp]\nenabled=true\n[[model]]\nkey=\"m\"\nbackend=\"llamacpp\"\nsource={{ file={:?} }}\nkind=\"llm\"\nvram_gb=1.0\n",
        minimal(),
        models,
        models.join("hf"),
        models.join("out/m.gguf")
    );
    let path = temp.config(&input);
    let output = run(&["--check-config", path.to_str().unwrap()]);
    let err = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{err}");
    assert!(
        err.contains("storage.models_dir")
            && err.contains("symbolic link")
            && err.contains("--import"),
        "{err}"
    );
}

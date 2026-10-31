use std::{path::PathBuf, process::Command};

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/check-secrets.sh")
}
fn temp(name: &str, text: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("eliezer-secrets-{}-{name}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}
fn run(args: &[&str], secret: Option<&str>) -> (i32, String) {
    let mut command = Command::new("sh");
    command.arg(script()).args(args).env_remove("CHECK_SECRET");
    if let Some(secret) = secret {
        command.env("CHECK_SECRET", secret);
    }
    let output = command.output().unwrap();
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr),
    )
}

#[test]
fn clean_logs_pass_and_secrets_are_found_without_being_printed() {
    let token = "hf_abcdefghijklmnopqrstuvwxyz0123";
    let key = format!("sk-{}", "a1".repeat(32));
    let clean = temp(
        "clean.log",
        "INFO Model loaded\nINFO download_end repo=org/model failed=false\n",
    );
    let (code, text) = run(&["--no-ps", "--log", clean.to_str().unwrap()], None);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("OK"));

    for (name, line) in [
        ("hf", format!("WARN something went wrong with {token}")),
        ("sk", format!("INFO key {key} created")),
        ("bearer", format!("DEBUG Authorization: Bearer {token}")),
    ] {
        let dirty = temp(&format!("{name}.log"), &format!("INFO fine\n{line}\n"));
        let (code, text) = run(&["--no-ps", "--log", dirty.to_str().unwrap()], None);
        assert_eq!(code, 1, "{name}: {text}");
        assert!(text.contains("FOUND"), "{name}: {text}");
        assert!(
            !text.contains(token) && !text.contains(&key),
            "{name}: the output shows the secret: {text}"
        );
    }

    // a known secret is searched literally, even when it has no recognizable shape
    let odd = temp("odd.log", "ERROR failed for s3cr3t-value-123\n");
    let (code, text) = run(
        &["--no-ps", "--log", odd.to_str().unwrap()],
        Some("s3cr3t-value-123"),
    );
    assert_eq!(code, 1, "{text}");
    assert!(!text.contains("s3cr3t-value-123"), "{text}");
    let (code, _) = run(&["--no-ps", "--log", odd.to_str().unwrap()], Some("other"));
    assert_eq!(code, 0);
}

#[test]
fn a_process_list_is_checked_and_usage_errors_exit_2() {
    // a saved process list: arguments without secrets pass, a token in an argument is found
    let clean = temp(
        "ps-clean.txt",
        "/usr/bin/eliezer --config /etc/eliezer/config.toml\nsleep 5\n",
    );
    let (code, text) = run(&["--ps-file", clean.to_str().unwrap()], None);
    assert_eq!(code, 0, "{text}");
    let token = "hf_ARGUMENTTOKEN0123456789abcdef";
    let dirty = temp(
        "ps-dirty.txt",
        &format!("/usr/bin/sd-server --api-key {token}\n"),
    );
    let (code, text) = run(&["--ps-file", dirty.to_str().unwrap()], None);
    assert_eq!(code, 1, "{text}");
    assert!(
        text.contains("process list") && !text.contains("ARGUMENTTOKEN"),
        "{text}"
    );
    assert_eq!(run(&["--no-ps"], None).0, 2, "nothing to check");
    assert_eq!(run(&["--bogus"], None).0, 2);
    assert_eq!(run(&["--log"], None).0, 2);
    assert_eq!(run(&["--ps-file"], None).0, 2);
}

#[test]
fn the_live_process_list_shows_a_token_in_an_argument() {
    // `$0` of the inner shell is the token: it is part of the command line, as `ps` shows it
    let mut child = Command::new("sh")
        .args(["-c", "sleep 5", "hf_LIVEARGUMENTTOKEN0123456789abcdef"])
        .spawn()
        .unwrap();
    let (code, text) = run(&[], None);
    child.kill().unwrap();
    let _ = child.wait();
    assert_eq!(code, 1, "{text}");
    assert!(!text.contains("LIVEARGUMENT"), "{text}");
}

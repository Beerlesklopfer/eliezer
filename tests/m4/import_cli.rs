//! `eliezer --import` through the real binary.
use std::{
    fs,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn id(flag: &str) -> Option<String> {
    let out = Command::new("id").arg(flag).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_eliezer"))
        .env_remove("ELIEZER_CONFIG")
        .args(args)
        .output()
        .unwrap()
}
fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}
fn setup(name: &str, accounts: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = fs::canonicalize(std::env::temp_dir())
        .unwrap()
        .join(format!("eliezer-import-cli-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src/community/Model-GGUF")).unwrap();
    fs::write(
        dir.join("src/community/Model-GGUF/Model-Q4.gguf"),
        b"GGUF-not-really",
    )
    .unwrap();
    fs::write(
        dir.join("src/community/Model-GGUF/mmproj-F16.gguf"),
        b"proj",
    )
    .unwrap();
    let config = dir.join("config.toml");
    fs::write(
        &config,
        format!(
            "[server]\nlisten = \"127.0.0.1:1234\"\n{accounts}\n[storage]\nmodels_dir = \"{d}/store\"\n[huggingface]\ncache_dir = \"{d}/store/hf\"\n[gpu]\n",
            d = dir.display()
        ),
    )
    .unwrap();
    (dir.clone(), dir.join("src"), config)
}
fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn dry_run_changes_nothing_and_a_move_hands_files_to_the_configured_account() {
    let (Some(user), Some(group)) = (id("-un"), id("-gn")) else {
        return; // no `id` command: nothing to test against
    };
    let (dir, src, config) = setup(
        "move",
        &format!("run_user = \"{user}\"\nrun_group = \"{group}\""),
    );
    let dry = run(&["--config", s(&config), "--import", s(&src), "--dry-run"]);
    assert!(dry.status.success(), "{}", err(&dry));
    assert!(out(&dry).contains("would move") && out(&dry).contains("Model-Q4.gguf"));
    assert!(src.join("community/Model-GGUF/Model-Q4.gguf").exists());
    assert!(!dir.join("store").exists(), "a dry run creates nothing");

    let real = run(&["--config", s(&config), "--import", s(&src)]);
    assert!(real.status.success(), "{}\n{}", out(&real), err(&real));
    let target = dir.join("store/community/Model-GGUF/Model-Q4.gguf");
    assert!(target.exists() && !src.join("community/Model-GGUF/Model-Q4.gguf").exists());
    assert!(
        dir.join("store/community/Model-GGUF/mmproj-F16.gguf")
            .exists()
    );
    assert_eq!(
        fs::metadata(&target).unwrap().uid(),
        fs::metadata(&dir).unwrap().uid()
    );
    let text = out(&real);
    assert!(
        text.contains("2 file(s)") && text.contains("0 failed"),
        "{text}"
    );
    assert!(
        text.contains("Added to") && text.contains("model-q4"),
        "{text}"
    );
    assert!(text.contains(".bak"), "the backup is announced: {text}");
    let written = fs::read_to_string(&config).unwrap();
    assert!(
        written.contains("[backend.llamacpp]\nenabled = true"),
        "{written}"
    );
    assert!(written.contains("key = \"model-q4\"") && written.contains("backend = \"llamacpp\""));
    assert!(
        !written.contains("mmproj-f16"),
        "projector files get no model entry"
    );
    assert!(dir.join("config.toml.bak").exists());
    let check = run(&["--check-config", s(&config)]);
    assert!(
        check.status.success(),
        "the edited configuration is valid: {}",
        err(&check)
    );

    // running it again finds nothing new
    let again = run(&["--config", s(&config), "--import", s(&src)]);
    assert!(
        again.status.success() && out(&again).contains("Nothing to import"),
        "{}",
        out(&again)
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn copy_keeps_the_originals() {
    let (dir, src, config) = setup("copy", "");
    let result = run(&["--config", s(&config), "--import", s(&src), "--copy"]);
    assert!(result.status.success(), "{}", err(&result));
    assert!(src.join("community/Model-GGUF/Model-Q4.gguf").exists());
    assert!(
        dir.join("store/community/Model-GGUF/Model-Q4.gguf")
            .exists()
    );
    assert!(
        err(&result).contains("ownership stays"),
        "a missing run_user is mentioned"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unknown_accounts_and_bad_combinations_fail_clearly() {
    let (dir, src, config) = setup("bad", "run_user = \"no-such-user-xyz\"");
    let result = run(&["--config", s(&config), "--import", s(&src)]);
    assert_eq!(result.status.code(), Some(2));
    assert!(
        err(&result).contains("no-such-user-xyz"),
        "{}",
        err(&result)
    );
    assert!(
        src.join("community/Model-GGUF/Model-Q4.gguf").exists(),
        "nothing moved"
    );
    for args in [
        vec!["--copy"],
        vec!["--dry-run"],
        vec!["--import"],
        vec!["--import", "a", "--import", "b"],
    ] {
        let mut full = vec!["--config", s(&config)];
        full.extend(args.iter());
        let result = run(&full);
        assert_eq!(result.status.code(), Some(2), "{args:?}");
    }
    let result = run(&[
        "--config",
        s(&config),
        "--import",
        s(&dir.join("store")),
        "--check-config",
    ]);
    assert_eq!(result.status.code(), Some(2));
    fs::remove_dir_all(dir).unwrap();
}

/// A GGUF header with only the given string and number entries, as diffusion models have.
fn tiny_gguf(entries: &[(&str, &str)]) -> Vec<u8> {
    let text = |out: &mut Vec<u8>, t: &str| {
        out.extend((t.len() as u64).to_le_bytes());
        out.extend(t.as_bytes());
    };
    let mut body = Vec::new();
    for (key, value) in entries {
        text(&mut body, key);
        body.extend(8_u32.to_le_bytes());
        text(&mut body, value);
    }
    let mut file = b"GGUF".to_vec();
    file.extend(3_u32.to_le_bytes());
    file.extend(0_u64.to_le_bytes());
    file.extend((entries.len() as u64).to_le_bytes());
    file.extend(body);
    file
}

#[test]
fn no_config_prints_the_blocks_and_leaves_the_file_alone() {
    let (dir, src, config) = setup("noconfig", "");
    let before = fs::read_to_string(&config).unwrap();
    let result = run(&["--config", s(&config), "--import", s(&src), "--no-config"]);
    assert!(result.status.success(), "{}", err(&result));
    assert!(out(&result).contains("[[model]]") && out(&result).contains("key = \"model-q4\""));
    assert_eq!(fs::read_to_string(&config).unwrap(), before);
    assert!(!dir.join("config.toml.bak").exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_broken_model_entry_does_not_block_the_import() {
    // The situation from real life: a pasted [[model]] block whose backend section is missing.
    let (dir, src, config) = setup("broken", "");
    let mut text = fs::read_to_string(&config).unwrap();
    text += "\n[[model]]\nkey = \"pasted\"\nbackend = \"llamacpp\"\nsource = { file = \"/nowhere/pasted.gguf\" }\nkind = \"llm\"\nsize_gb = 4.6\n";
    fs::write(&config, &text).unwrap();
    assert!(
        !run(&["--check-config", s(&config)]).status.success(),
        "the starting point is invalid"
    );
    let result = run(&["--config", s(&config), "--import", s(&src)]);
    assert!(
        result.status.success(),
        "{}\n{}",
        out(&result),
        err(&result)
    );
    assert!(
        dir.join("store/community/Model-GGUF/Model-Q4.gguf")
            .exists(),
        "the files are moved anyway"
    );
    assert!(
        err(&result).contains("model list"),
        "the problem is mentioned: {}",
        err(&result)
    );
    // The result is either a valid configuration with the new entry, or an unchanged file with
    // the reason; it is never a half-written file.
    let after = fs::read_to_string(&config).unwrap();
    if run(&["--check-config", s(&config)]).status.success() {
        assert!(after.contains("model-q4"));
    } else {
        assert_eq!(after, text, "{}", err(&result));
        assert!(err(&result).contains("not changed"), "{}", err(&result));
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn image_models_are_moved_but_stay_out_of_the_active_configuration() {
    let (dir, src, config) = setup("image", "");
    fs::write(
        src.join("community/Model-GGUF/qwen-image.gguf"),
        tiny_gguf(&[("general.architecture", "qwen_image21")]),
    )
    .unwrap();
    let result = run(&["--config", s(&config), "--import", s(&src)]);
    assert!(result.status.success(), "{}", err(&result));
    let text = out(&result);
    assert!(
        dir.join("store/community/Model-GGUF/qwen-image.gguf")
            .exists(),
        "{text}"
    );
    assert!(
        text.contains("image model") && text.contains("# backend = \"sdcpp\""),
        "{text}"
    );
    let written = fs::read_to_string(&config).unwrap();
    assert!(
        !written.contains("qwen-image"),
        "an image model is not an active entry: {written}"
    );
    assert!(
        written.contains("model-q4"),
        "the language model next to it is entered"
    );
    assert!(run(&["--check-config", s(&config)]).status.success());
    fs::remove_dir_all(dir).unwrap();
}

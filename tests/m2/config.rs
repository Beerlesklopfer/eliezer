use eliezer_core::config::{Config, PathResolutionError, PathResolver};
use std::path::{Path, PathBuf};

const EXAMPLE: &str = include_str!("../../packaging/config.example.toml");
struct Identity;
impl PathResolver for Identity {
    fn resolve(&self, path: &Path) -> Result<PathBuf, PathResolutionError> {
        Ok(path.into())
    }
}

#[test]
fn example_and_defaults_are_valid() {
    let config = Config::parse(EXAMPLE).unwrap();
    config.validate_paths(EXAMPLE, &Identity).unwrap();
    assert_eq!(config.model.len(), 2);
    assert_eq!(config.scheduler.failed_retry_secs, 30);
}

#[test]
fn schema_rejects_unknown_fields_without_echoing_values() {
    for (old, new) in [
        (
            "priority = 10",
            "priority = 10\nmisspelled = 'CONFIDENTIAL_VALUE'",
        ),
        (
            "listen = \"127.0.0.1:1234\"",
            "listen = 'CONFIDENTIAL_VALUE'",
        ),
        ("layers = 48", "layers = 'CONFIDENTIAL_VALUE'"),
        ("text_encoder =", "unbekannt ="),
    ] {
        let error = Config::parse(&EXAMPLE.replace(old, new)).unwrap_err();
        assert!(error.line > 1, "{error}");
        assert!(!error.to_string().contains("CONFIDENTIAL_VALUE"));
        assert!(!error.field.is_empty());
    }
}

#[test]
fn semantic_errors_report_correct_field_and_line() {
    for (old, new, field) in [
        ("reserve_gb = 2.0", "reserve_gb = 99.0", "gpu.reserve_gb"),
        (
            "vram_budget_gb = 24.0",
            "vram_budget_gb = nan",
            "gpu.vram_budget_gb",
        ),
        (
            "port_range = [24600, 24609]",
            "port_range = [24609, 24600]",
            "backend.sdcpp.port_range",
        ),
        ("queue_max = 64", "queue_max = 0", "scheduler.queue_max"),
        (
            "default_context = 32768",
            "default_context = 999999",
            "model[0].default_context",
        ),
        (
            "key = \"qwen-image\"",
            "key = \"qwen/qwen3-coder-30b\"",
            "model[1].key",
        ),
        ("vram_gb = 14.0", "vram_gb = -1.0", "model[1].vram_gb"),
    ] {
        assert!(EXAMPLE.contains(old));
        let input = EXAMPLE.replace(old, new);
        let error = Config::parse(&input).unwrap_err();
        assert_eq!(error.field, field, "{error}");
        assert_eq!(
            error.line,
            input
                .lines()
                .collect::<Vec<_>>()
                .iter()
                .rposition(|l| l.trim() == new)
                .unwrap()
                + 1,
            "{error}"
        );
    }
}

#[test]
fn rejects_sources_and_reasoning_outside_contract() {
    for (old, new) in [
        (
            "hf = \"Qwen/Qwen3-Coder-30B-A3B-Instruct\"",
            "hf = '../falsch'",
        ),
        ("default = \"medium\"", "default = 'unbekannt'"),
        ("parallel = 1", "parallel = 0"),
        ("layers = 48", "layers = 0"),
    ] {
        assert!(Config::parse(&EXAMPLE.replace(old, new)).is_err());
    }
}

#[test]
fn rejects_paths_outside_storage_and_resolved_symlinks() {
    let input = EXAMPLE.replace(
        "/var/lib/eliezer/models/qwen-image-2.1-UC-Q4_K_M.gguf",
        "/var/lib/eliezer/models-nachbar/model.gguf",
    );
    let config = Config::parse(&input).unwrap();
    assert_eq!(
        config.validate_paths(&input, &Identity).unwrap_err().field,
        "model[1].source.file"
    );
    struct Escape;
    impl PathResolver for Escape {
        fn resolve(&self, path: &Path) -> Result<PathBuf, PathResolutionError> {
            if path.extension().is_some_and(|e| e == "gguf") {
                Ok("/ausserhalb/model.gguf".into())
            } else {
                Ok(path.into())
            }
        }
    }
    let config = Config::parse(EXAMPLE).unwrap();
    assert!(config.validate_paths(EXAMPLE, &Escape).is_err());
}

#[test]
fn semantic_locations_accept_comments_and_quoted_table_names() {
    let input = EXAMPLE
        .replace("[[model]]", "[[\"model\"]] # A model")
        .replace("default_context = 32768", "default_context = 999999");
    let error = Config::parse(&input).unwrap_err();
    assert_eq!(error.field, "model[0].default_context");
    assert_eq!(
        error.line,
        input
            .lines()
            .position(|line| line == "default_context = 999999")
            .unwrap()
            + 1
    );
}

#[test]
fn duplicate_model_keys_and_duplicate_toml_keys_name_the_place() {
    let model = "[[model]]\nkey=\"a\"\nbackend=\"mistralrs\"\nsource={hf=\"t/a\"}\nkind=\"llm\"\nvram_gb=1.0\n";
    let head = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\nmeasure=\"none\"\nvram_budget_gb=8.0\nreserve_gb=1.0\n[backend.mistralrs]\nenabled=true\n";
    let two = format!("{head}{model}{model}");
    let message = Config::parse(&two).unwrap_err().to_string();
    assert!(message.contains("more than once"), "{message}");
    assert!(message.contains("model[1].key"), "{message}");
    // duplicate TOML key in the same section
    let dup = format!("{head}vram_budget_gb=9.0\n{model}");
    let message = Config::parse(&dup).unwrap_err().to_string();
    assert!(message.contains("line"), "{message}");
}

#[test]
fn a_known_field_in_the_wrong_section_says_where_it_belongs() {
    let head = "[server]\nlisten = \"127.0.0.1:1234\"\n";
    // the real-life slip: cache_dir written into [storage]
    let text = format!("{head}[storage]\nmodels_dir = \"/m\"\ncache_dir = \"/m/hf\"\n[gpu]\n");
    let error = Config::parse(&text).unwrap_err().to_string();
    assert!(error.contains("storage.cache_dir"), "{error}");
    assert!(
        error.contains("[huggingface]") && error.contains("not of [storage]"),
        "{error}"
    );
    assert!(error.contains("line 5"), "{error}");
    // a field that exists nowhere is still not echoed
    let text = format!("{head}[storage]\nVERTRAULICH = 1\n[gpu]\n");
    let error = Config::parse(&text).unwrap_err().to_string();
    assert!(
        !error.contains("VERTRAULICH") && !error.contains("belongs"),
        "{error}"
    );
    // a field in its own section is not touched by the hint
    let text = format!("{head}[huggingface]\ncache_dir = 5\n[gpu]\n");
    let error = Config::parse(&text).unwrap_err().to_string();
    assert!(
        error.contains("huggingface.cache_dir") && !error.contains("not of"),
        "{error}"
    );
}

#[test]
fn moved_files_and_directories_are_reported_as_problems_not_errors() {
    let dir = std::env::temp_dir().join(format!("eliezer-problems-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    std::fs::write(dir.join("models/here.gguf"), b"x").unwrap();
    std::fs::create_dir(dir.join("models/a-directory.gguf")).unwrap();
    let text = |models_dir: &str, files: &[(&str, &str)]| {
        let mut t = format!(
            "[server]\nlisten = \"127.0.0.1:1234\"\n[storage]\nmodels_dir = \"{models_dir}\"\n[huggingface]\ncache_dir = \"{models_dir}/hf\"\n[gpu]\n[backend.llamacpp]\nenabled = true\n"
        );
        for (key, file) in files {
            t += &format!(
                "[[model]]\nkey = \"{key}\"\nbackend = \"llamacpp\"\nsource = {{ file = \"{file}\" }}\nkind = \"llm\"\nvram_gb = 1.0\n"
            );
        }
        t
    };
    let store = dir.join("models");
    let (here, gone, folder) = (
        store.join("here.gguf"),
        store.join("gone.gguf"),
        store.join("a-directory.gguf"),
    );
    let config = Config::parse(&text(
        store.to_str().unwrap(),
        &[
            ("ok", here.to_str().unwrap()),
            ("moved", gone.to_str().unwrap()),
            ("folder", folder.to_str().unwrap()),
        ],
    ))
    .unwrap();
    let problems = config.file_problems();
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems
            .iter()
            .any(|p| p.contains("'moved'") && p.contains("not found") && p.contains("--import")),
        "{problems:?}"
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("'folder'") && p.contains("not a regular file")),
        "{problems:?}"
    );
    // a model store that is gone is reported with its path
    let missing = dir.join("nowhere");
    let config = Config::parse(&text(missing.to_str().unwrap(), &[])).unwrap();
    let problems = config.file_problems();
    assert!(
        problems.len() == 1
            && problems[0].contains("nowhere")
            && problems[0].contains("storage.models_dir"),
        "{problems:?}"
    );
    // everything in place: no problem
    let config = Config::parse(&text(
        store.to_str().unwrap(),
        &[("ok", here.to_str().unwrap())],
    ))
    .unwrap();
    assert!(config.file_problems().is_empty());
    std::fs::remove_dir_all(dir).unwrap();
}

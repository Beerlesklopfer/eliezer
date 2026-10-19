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
            "priority = 10\nmisspelled = 'VERTRAULICHER_WERT'",
        ),
        (
            "listen = \"127.0.0.1:1234\"",
            "listen = 'VERTRAULICHER_WERT'",
        ),
        ("layers = 48", "layers = 'VERTRAULICHER_WERT'"),
        ("text_encoder =", "unbekannt ="),
    ] {
        let error = Config::parse(&EXAMPLE.replace(old, new)).unwrap_err();
        assert!(error.line > 1, "{error}");
        assert!(!error.to_string().contains("VERTRAULICHER_WERT"));
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
        .replace("[[model]]", "[[\"model\"]] # Ein Modell")
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

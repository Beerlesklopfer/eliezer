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
fn cpu_placement_is_off_by_default_and_image_models_need_the_gpu() {
    let config = Config::parse(EXAMPLE).unwrap();
    assert_eq!(config.cpu.ram_budget_gb, 0.0);
    let with = EXAMPLE.replace("[scheduler]", "[cpu]\nram_budget_gb = 32.0\n\n[scheduler]");
    assert_eq!(Config::parse(&with).unwrap().cpu.ram_budget_gb, 32.0);
    let negative = EXAMPLE.replace("[scheduler]", "[cpu]\nram_budget_gb = -1.0\n\n[scheduler]");
    assert!(Config::parse(&negative).is_err());
    let image = format!(
        "{EXAMPLE}\n[[model]]\nkey=\"pic\"\nbackend=\"sdcpp\"\nkind=\"image\"\ndevice=\"cpu\"\nvram_gb=4.0\nsource={{file=\"/var/lib/eliezer/models/p.gguf\"}}\n"
    );
    let error = Config::parse(&image).unwrap_err().to_string();
    assert!(error.contains("generation on the CPU"), "{error}");
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

#[test]
fn a_missing_section_is_named() {
    let error = Config::parse("[server]\nlisten=\"127.0.0.1:1234\"\n")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("gpu") && error.contains("required section or field is missing"),
        "{error}"
    );
    // a missing field inside a model entry names the entry and the field
    let error = Config::parse("[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[[model]]\nbackend=\"llamacpp\"\n")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("model[0].key") && error.contains("missing"),
        "{error}"
    );
}

#[test]
fn the_library_path_of_the_image_program_is_checked() {
    let base = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[backend.sdcpp]\nprogram=\"/usr/local/bin/sd-server\"\nport_range=[24600,24609]\n";
    let ok = format!(
        "{base}library_path=[\"/usr/lib/x86_64-linux-gnu/eliezer\",\"/opt/rocm/lib/llvm/lib\"]\n"
    );
    assert_eq!(
        Config::parse(&ok)
            .unwrap()
            .backend
            .sdcpp
            .unwrap()
            .library_path
            .len(),
        2
    );
    assert!(
        Config::parse(base)
            .unwrap()
            .backend
            .sdcpp
            .unwrap()
            .library_path
            .is_empty()
    );
    for bad in ["[\"relative\"]", "[\"/a/../b\"]", "[\"/a:/b\"]"] {
        let error = Config::parse(&format!("{base}library_path={bad}\n"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("library_path"), "{bad}: {error}");
    }
}

#[test]
fn the_ocr_flag_needs_vision() {
    let base = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[backend.llamacpp]\nenabled=true\n[[model]]\nkey=\"r\"\nbackend=\"llamacpp\"\nkind=\"llm\"\nsource={file=\"/var/lib/eliezer/models/r.gguf\",projector=\"/var/lib/eliezer/models/p.gguf\"}\nsize_gb=1.0\narch={layers=2,kv_heads=2,head_dim=64}\n";
    assert!(
        Config::parse(&format!("{base}vision=true\nocr=true\n"))
            .unwrap()
            .model[0]
            .ocr
    );
    assert!(
        !Config::parse(&format!("{base}vision=true\n"))
            .unwrap()
            .model[0]
            .ocr
    );
    let plain = base.replace(",projector=\"/var/lib/eliezer/models/p.gguf\"", "");
    let error = Config::parse(&format!("{plain}ocr=true\n"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("model[0].ocr"), "{error}");
}

#[test]
fn a_prompt_helper_names_a_language_model_and_belongs_to_an_image_model() {
    let base = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[backend.llamacpp]\nenabled=true\n[backend.sdcpp]\nprogram=\"/usr/bin/sd-server\"\nport_range=[24600,24609]\n[[model]]\nkey=\"w\"\nbackend=\"llamacpp\"\nkind=\"llm\"\nsource={file=\"/var/lib/eliezer/models/w.gguf\"}\nsize_gb=1.0\narch={layers=2,kv_heads=2,head_dim=64}\n[[model]]\nkey=\"p\"\nbackend=\"sdcpp\"\nkind=\"image\"\nsource={file=\"/var/lib/eliezer/models/p.gguf\"}\nvram_gb=9.0\n";
    let ok = format!("{base}extra={{prompt_helper=\"w\"}}\n");
    assert!(Config::parse(&ok).is_ok());
    for bad in [
        "extra={prompt_helper=\"missing\"}",
        "extra={prompt_helper=\"p\"}",
    ] {
        let error = Config::parse(&format!("{base}{bad}\n"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("extra.prompt_helper"), "{bad}: {error}");
    }
    // a language model cannot have a helper
    let text = base.replace(
        "size_gb=1.0\n",
        "size_gb=1.0\nextra={prompt_helper=\"w\"}\n",
    );
    let error = Config::parse(&text).unwrap_err().to_string();
    assert!(error.contains("extra.prompt_helper"), "{error}");
}

#[test]
fn the_download_allowlist_takes_wildcards_in_the_parts_but_never_everything() {
    let parse = |entry: &str| {
        Config::parse(&format!(
            "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[huggingface]\nallow=[\"{entry}\"]\n"
        ))
    };
    let allows = |entry: &str, repo: &str| parse(entry).unwrap().huggingface.allows(repo);
    // exact, whole organisation, and a pattern for the repository name in any organisation
    assert!(allows(
        "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "Qwen/Qwen3-VL-8B-Instruct-GGUF"
    ));
    assert!(!allows(
        "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "Qwen/Qwen3-VL-8B"
    ));
    assert!(allows("Qwen/*", "Qwen/anything-at-all"));
    assert!(!allows("Qwen/*", "mradermacher/Qwen3-VL-8B-GGUF"));
    assert!(allows(
        "*/Qwen3-VL*",
        "mradermacher/Qwen3-VL-8B-Instruct-abliterated-GGUF"
    ));
    assert!(allows("*/Qwen3-VL*", "Qwen/Qwen3-VL-8B-Instruct-GGUF"));
    assert!(!allows("*/Qwen3-VL*", "mradermacher/Llama-3-8B-GGUF"));
    assert!(allows(
        "mradermacher/*VL-8B*GGUF",
        "mradermacher/Qwen3-VL-8B-Instruct-GGUF"
    ));
    // `**` is the same as `*`, so the pattern of a hurried entry works
    assert!(allows("**/Qwen*", "mradermacher/Qwen3-VL-8B-GGUF"));
    // the organisation part does not leak into the name and the other way round
    assert!(!allows("Qwen/*", "Qwen"));
    assert!(!allows("*/Qwen*", "Qwen3/other"));
    // everything at once is refused, and so are malformed entries
    for bad in [
        "*/*",
        "**/**",
        "*",
        "a/b/c",
        "../x",
        "a b/c",
        "/x",
        "x/",
        ".hidden/x",
        "a/b;c",
    ] {
        let error = parse(bad).unwrap_err().to_string();
        assert!(error.contains("huggingface.allow"), "{bad}: {error}");
    }
    assert!(
        Config::parse("[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n")
            .unwrap()
            .huggingface
            .allow
            .is_empty()
    );
}

#[test]
fn only_an_image_model_can_edit_pictures() {
    let base = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[backend.llamacpp]\nenabled=true\n[backend.sdcpp]\nprogram=\"/usr/bin/sd-server\"\nport_range=[24600,24609]\n[[model]]\nkey=\"w\"\nbackend=\"llamacpp\"\nkind=\"llm\"\nsource={file=\"/var/lib/eliezer/models/w.gguf\"}\nsize_gb=1.0\narch={layers=2,kv_heads=2,head_dim=64}\n[[model]]\nkey=\"p\"\nbackend=\"sdcpp\"\nkind=\"image\"\nsource={file=\"/var/lib/eliezer/models/p.gguf\"}\nvram_gb=9.0\n";
    assert!(Config::parse(&format!("{base}extra={{image_edit=true}}\n")).is_ok());
    assert!(Config::parse(&format!("{base}extra={{image_edit=false}}\n")).is_ok());
    let text = base.replace("size_gb=1.0\n", "size_gb=1.0\nextra={image_edit=true}\n");
    let error = Config::parse(&text).unwrap_err().to_string();
    assert!(error.contains("extra.image_edit"), "{error}");
}

#[test]
fn a_music_model_needs_the_music_backend_a_memory_figure_and_plain_component_names() {
    let base = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[backend.sdcpp]\nprogram=\"/usr/bin/sd-server\"\nport_range=[24600,24609]\n[backend.audiocpp]\nprogram=\"/usr/bin/audiocpp_server\"\nport_range=[24700,24709]\n";
    let model = |extra: &str| {
        format!(
            "{base}[[model]]\nkey=\"m\"\nbackend=\"audiocpp\"\nkind=\"music\"\nsource={{file=\"/var/lib/eliezer/models/music\"}}\nvram_gb=10.0\n{extra}"
        )
    };
    assert!(Config::parse(&model("")).is_ok());
    assert!(
        Config::parse(&model(
            "extra={language_model=\"language_model_q8_0.gguf\",rvq_depth_decoder=\"a.gguf\",flow_transformer=\"b.gguf\",duration_sec=120,steps=20,cfg_scale=1.7}\n"
        ))
        .is_ok()
    );
    for (extra, field) in [
        ("extra={language_model=\"../x.gguf\"}\n", "extra"),
        ("extra={language_model=\"a/b.gguf\"}\n", "extra"),
        ("extra={flow_transformer=\"x.bin\"}\n", "extra"),
        ("extra={rvq_depth_decoder=\".hidden.gguf\"}\n", "extra"),
        ("extra={duration_sec=0}\n", "extra.duration_sec"),
        ("extra={duration_sec=301}\n", "extra.duration_sec"),
    ] {
        let error = Config::parse(&model(extra)).unwrap_err().to_string();
        assert!(error.contains(field), "{extra}: {error}");
    }
    // no memory figure, the wrong backend, the wrong kind, the CPU
    let no_figure = model("").replace("vram_gb=10.0\n", "");
    assert!(
        Config::parse(&no_figure)
            .unwrap_err()
            .to_string()
            .contains("vram_gb")
    );
    let wrong_backend = model("").replace("backend=\"audiocpp\"", "backend=\"sdcpp\"");
    assert!(
        Config::parse(&wrong_backend)
            .unwrap_err()
            .to_string()
            .contains("kind")
    );
    let wrong_kind = model("").replace("kind=\"music\"", "kind=\"image\"");
    assert!(
        Config::parse(&wrong_kind)
            .unwrap_err()
            .to_string()
            .contains("kind")
    );
    let cpu = model("device=\"cpu\"\n");
    assert!(
        Config::parse(&cpu)
            .unwrap_err()
            .to_string()
            .contains("device")
    );
    // the music settings belong to music models
    let image = format!(
        "{base}[[model]]\nkey=\"i\"\nbackend=\"sdcpp\"\nkind=\"image\"\nsource={{file=\"/var/lib/eliezer/models/p.gguf\"}}\nvram_gb=9.0\nextra={{duration_sec=30}}\n"
    );
    assert!(
        Config::parse(&image)
            .unwrap_err()
            .to_string()
            .contains("extra")
    );
    // the backend: its compute setting and a port range of its own
    let compute = base.replace(
        "port_range=[24700,24709]",
        "port_range=[24700,24709]\ncompute=\"metal\"",
    );
    assert!(
        Config::parse(&compute)
            .unwrap_err()
            .to_string()
            .contains("compute")
    );
    for ok in ["hip", "cuda", "vulkan", "cpu"] {
        let text = base.replace(
            "port_range=[24700,24709]",
            &format!("port_range=[24700,24709]\ncompute=\"{ok}\""),
        );
        assert!(Config::parse(&text).is_ok(), "{ok}");
    }
    let overlap = base.replace("[24700,24709]", "[24605,24615]");
    assert!(
        Config::parse(&overlap)
            .unwrap_err()
            .to_string()
            .contains("audiocpp")
    );
    let on_service = base.replace("[24700,24709]", "[1200,1300]");
    assert!(
        Config::parse(&on_service)
            .unwrap_err()
            .to_string()
            .contains("audiocpp")
    );
    let relative = base.replace("/usr/bin/audiocpp_server", "audiocpp_server");
    assert!(
        Config::parse(&relative)
            .unwrap_err()
            .to_string()
            .contains("audiocpp")
    );
    // a music model without the backend
    let without = model("").replace(
        "[backend.audiocpp]\nprogram=\"/usr/bin/audiocpp_server\"\nport_range=[24700,24709]\n",
        "",
    );
    assert!(
        Config::parse(&without)
            .unwrap_err()
            .to_string()
            .contains("backend")
    );
}

const SPEECH_BASE: &str = "[server]\nlisten=\"127.0.0.1:1234\"\n[gpu]\nvendor=\"none\"\n[backend.sdcpp]\nprogram=\"/usr/bin/sd-server\"\nport_range=[24600,24609]\n[backend.audiocpp]\nprogram=\"/usr/bin/audiocpp_server\"\nport_range=[24700,24709]\n[backend.whispercpp]\nprogram=\"/usr/bin/whisper-server\"\nport_range=[24800,24809]\n";

fn speech_model(extra: &str) -> String {
    format!(
        "{SPEECH_BASE}[[model]]\nkey=\"w\"\nbackend=\"whispercpp\"\nkind=\"stt\"\nsource={{file=\"/var/lib/eliezer/models/ggml-model-q5_0.bin\"}}\nvram_gb=2.0\n{extra}"
    )
}

#[test]
fn a_speech_model_needs_the_whisper_backend_a_memory_figure_and_sane_settings() {
    let config = Config::parse(&speech_model("")).unwrap();
    let speech = config.backend.whispercpp.as_ref().unwrap();
    // the defaults of the section
    assert_eq!(
        (
            speech.chunk_secs,
            speech.overlap_secs,
            speech.max_file_mb,
            speech.max_audio_minutes
        ),
        (600, 5, 100, 120)
    );
    assert!(speech.split_on_word);
    assert_eq!(speech.max_segment_chars, 0);
    assert_eq!(
        (
            speech.load_timeout_secs,
            speech.request_timeout_secs,
            speech.stop_grace_secs
        ),
        (300, 3600, 15)
    );
    assert!(speech.flash_attn.is_none() && speech.gpu_device.is_none());
    assert!(
        Config::parse(&speech_model(
            "device=\"cpu\"\nextra={language=\"de\",threads=8,max_file_mb=300,max_audio_minutes=90}\n"
        ))
        .is_ok(),
        "speech recognition may run on the CPU"
    );
    assert!(Config::parse(&speech_model("extra={language=\"auto\"}\n")).is_ok());
    for (extra, field) in [
        ("extra={language=\"German\"}\n", "extra.language"),
        ("extra={language=\"de;x\"}\n", "extra.language"),
        ("extra={threads=0}\n", "extra"),
        ("extra={threads=1000}\n", "extra"),
        ("extra={max_file_mb=0}\n", "extra"),
        ("extra={max_audio_minutes=0}\n", "extra"),
        ("extra={max_segment_chars=19}\n", "extra.max_segment_chars"),
        ("extra={max_segment_chars=401}\n", "extra.max_segment_chars"),
    ] {
        let error = Config::parse(&speech_model(extra)).unwrap_err().to_string();
        assert!(error.contains(field), "{extra}: {error}");
    }
    // the per-model segment settings
    let custom = Config::parse(&speech_model(
        "extra={split_on_word=false,max_segment_chars=120}\n",
    ))
    .unwrap();
    let extra = custom.model[0].extra.as_ref().unwrap();
    assert_eq!(
        (extra.split_on_word, extra.max_segment_chars),
        (Some(false), Some(120))
    );
    assert!(Config::parse(&speech_model("extra={max_segment_chars=0}\n")).is_ok());
    // the speech settings belong to speech models
    let music = format!(
        "{SPEECH_BASE}[[model]]\nkey=\"m\"\nbackend=\"audiocpp\"\nkind=\"music\"\nsource={{file=\"/var/lib/eliezer/models/music\"}}\nvram_gb=10.0\nextra={{language=\"de\"}}\n"
    );
    assert!(
        Config::parse(&music)
            .unwrap_err()
            .to_string()
            .contains("extra")
    );
    // no memory figure, the wrong backend, the wrong kind, a Hugging Face source
    let no_figure = speech_model("").replace("vram_gb=2.0\n", "");
    assert!(
        Config::parse(&no_figure)
            .unwrap_err()
            .to_string()
            .contains("vram_gb")
    );
    let wrong_backend = speech_model("").replace("backend=\"whispercpp\"", "backend=\"audiocpp\"");
    assert!(
        Config::parse(&wrong_backend)
            .unwrap_err()
            .to_string()
            .contains("kind")
    );
    let wrong_kind = speech_model("").replace("kind=\"stt\"", "kind=\"music\"");
    assert!(
        Config::parse(&wrong_kind)
            .unwrap_err()
            .to_string()
            .contains("kind")
    );
    let hf = speech_model("").replace(
        "source={file=\"/var/lib/eliezer/models/ggml-model-q5_0.bin\"}",
        "source={hf=\"cstr/whisper-large-v3-turbo-german-ggml\"}",
    );
    assert!(
        Config::parse(&hf)
            .unwrap_err()
            .to_string()
            .contains("source")
    );
    // a music model on the CPU is still refused
    let cpu_music = music.replace("extra={language=\"de\"}\n", "device=\"cpu\"\n");
    assert!(
        Config::parse(&cpu_music)
            .unwrap_err()
            .to_string()
            .contains("device")
    );
    // without the backend section
    let without = speech_model("").replace(
        "[backend.whispercpp]\nprogram=\"/usr/bin/whisper-server\"\nport_range=[24800,24809]\n",
        "",
    );
    assert!(
        Config::parse(&without)
            .unwrap_err()
            .to_string()
            .contains("backend")
    );
}

#[test]
fn the_whisper_backend_section_is_checked() {
    assert!(Config::parse(SPEECH_BASE).is_ok());
    let with = |line: &str| {
        SPEECH_BASE.replace(
            "port_range=[24800,24809]",
            &format!("port_range=[24800,24809]\n{line}"),
        )
    };
    assert!(Config::parse(&with("flash_attn=true\ngpu_device=1\nchunk_secs=300\noverlap_secs=10\nsplit_on_word=false\nmax_segment_chars=100\nload_timeout_secs=600\nrequest_timeout_secs=7200\nstop_grace_secs=30\nmax_file_mb=500\nmax_audio_minutes=240\nlibrary_path=[\"/opt/rocm/lib\"]")).is_ok());
    for (line, field) in [
        ("chunk_secs=10", "chunk_secs"),
        ("chunk_secs=4000", "chunk_secs"),
        ("chunk_secs=60\noverlap_secs=30", "overlap_secs"),
        ("overlap_secs=61", "overlap_secs"),
        ("max_file_mb=0", "max_file_mb"),
        ("max_audio_minutes=0", "max_file_mb"),
        ("max_segment_chars=10", "max_segment_chars"),
        ("max_segment_chars=500", "max_segment_chars"),
        ("load_timeout_secs=0", "load_timeout_secs"),
        ("load_timeout_secs=4000", "load_timeout_secs"),
        ("request_timeout_secs=0", "request_timeout_secs"),
        ("request_timeout_secs=90000", "request_timeout_secs"),
        ("stop_grace_secs=0", "stop_grace_secs"),
        ("stop_grace_secs=301", "stop_grace_secs"),
        ("unknown=1", "unknown"),
    ] {
        let error = Config::parse(&with(line)).unwrap_err().to_string();
        assert!(
            error.contains(field) || error.contains("Unknown"),
            "{line}: {error}"
        );
    }
    for (range, other) in [("[24605,24615]", "sdcpp"), ("[24705,24715]", "audiocpp")] {
        let text = SPEECH_BASE.replace("[24800,24809]", range);
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(
            error.contains("whispercpp") && error.contains(other),
            "{error}"
        );
    }
    let on_service = SPEECH_BASE.replace("[24800,24809]", "[1200,1300]");
    assert!(
        Config::parse(&on_service)
            .unwrap_err()
            .to_string()
            .contains("whispercpp")
    );
    let relative = SPEECH_BASE.replace("/usr/bin/whisper-server", "whisper-server");
    assert!(
        Config::parse(&relative)
            .unwrap_err()
            .to_string()
            .contains("whispercpp")
    );
}

use eliezer_backend::{Backend, BackendError, ChatRequest, LoadSpec, TranscriptionRequest};
use eliezer_core::config::DevicePolicy;
use eliezer_whispercpp::{
    GENERATION_BASE, NO_WRAP, ProgramSettings, SpeechModel, WhispercppBackend, WordTiming,
    arguments, dtw_preset_of, inference_fields, multipart_body, parse_transcription,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU16, AtomicU64, Ordering},
    time::Duration,
};

const FAKE: &str = env!("CARGO_BIN_EXE_fake-whisper-server");

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "eliezer-whisper-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A block of ports per test so that tests running side by side do not meet (apart from the music tests' blocks).
fn ports(width: u16) -> (u16, u16) {
    static NEXT: AtomicU16 = AtomicU16::new(0);
    let base = 38000
        + (std::process::id() % 20) as u16 * 200
        + NEXT.fetch_add(width + 1, Ordering::Relaxed);
    (base, base + width - 1)
}

/// A "ggml file" whose first line tells the fake server how to behave.
fn model(dir: &Path, mode: &str) -> SpeechModel {
    let file = dir.join("ggml-test.bin");
    std::fs::write(&file, format!("{mode}\n")).unwrap();
    SpeechModel {
        file,
        language: "de".into(),
        threads: Some(3),
        split_on_word: true,
        max_segment_chars: 0,
        word_timing: WordTiming::Off,
    }
}

fn backend_with(model: SpeechModel, ports: (u16, u16), ready: Duration) -> WhispercppBackend {
    let backend = WhispercppBackend::with_timeouts(
        FAKE.into(),
        ports,
        ProgramSettings {
            flash_attn: Some(true),
            gpu_device: Some(1),
        },
        ready,
        Duration::from_secs(10),
    );
    backend.set_models(HashMap::from([("stt".to_owned(), model)]));
    backend
}
fn backend(model: SpeechModel, ports: (u16, u16)) -> WhispercppBackend {
    backend_with(model, ports, Duration::from_secs(10))
}

fn spec(device: DevicePolicy) -> LoadSpec {
    LoadSpec {
        model: "stt".into(),
        context: 1,
        parallel: 1,
        bytes: 1,
        device,
    }
}
fn wav() -> Vec<u8> {
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&[0; 40]);
    wav
}
fn request() -> TranscriptionRequest {
    TranscriptionRequest {
        model: "stt".into(),
        wav: wav(),
        language: Some("de".into()),
        prompt: None,
        temperature: None,
    }
}
fn recorded(dir: &Path, what: &str) -> String {
    std::fs::read_to_string(dir.join(format!("fake.{what}"))).unwrap_or_default()
}
fn private_directory(port: u16) -> PathBuf {
    std::env::temp_dir().join(format!("eliezer-whispercpp-{}-{port}", std::process::id()))
}

#[test]
fn the_arguments_are_a_fixed_list_without_conversion() {
    let model = SpeechModel {
        file: "/m/a b/$(x).bin".into(),
        language: "de".into(),
        threads: Some(4),
        split_on_word: true,
        max_segment_chars: 0,
        word_timing: WordTiming::Off,
    };
    let strings = |cpu, settings: &ProgramSettings| -> Vec<String> {
        arguments(&model, 24800, Path::new("/run/x y"), settings, cpu, None)
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    };
    let gpu = strings(
        false,
        &ProgramSettings {
            flash_attn: Some(false),
            gpu_device: Some(1),
        },
    );
    assert_eq!(
        gpu,
        [
            "--model",
            "/m/a b/$(x).bin",
            "--host",
            "127.0.0.1",
            "--port",
            "24800",
            "--language",
            "de",
            "--public",
            "/run/x y",
            "--threads",
            "4",
            "--device",
            "1",
            "--no-flash-attn"
        ]
    );
    let cpu = strings(
        true,
        &ProgramSettings {
            flash_attn: Some(true),
            gpu_device: Some(1),
        },
    );
    assert!(cpu.contains(&"--no-gpu".to_owned()));
    assert!(!cpu.contains(&"--device".to_owned()) && !cpu.contains(&"--flash-attn".to_owned()));
    assert!(!gpu.iter().chain(&cpu).any(|a| a.contains("convert")));
    // word timing: the preset, and flash attention off whatever the setting says (the program drops DTW with it)
    for cpu in [false, true] {
        let timed: Vec<String> = arguments(
            &model,
            24800,
            Path::new("/run/x y"),
            &ProgramSettings {
                flash_attn: Some(true),
                gpu_device: Some(1),
            },
            cpu,
            Some("large.v3.turbo"),
        )
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
        assert!(timed.ends_with(&[
            "--dtw".to_owned(),
            "large.v3.turbo".to_owned(),
            "--no-flash-attn".to_owned()
        ]));
        assert!(!timed.contains(&"--flash-attn".to_owned()), "{timed:?}");
    }
}

/// The first 48 bytes of a ggml whisper file: the magic and the shape.
fn ggml_header(vocab: i32, audio_state: i32, audio_layer: i32, text_layer: i32) -> Vec<u8> {
    let numbers = [
        0x6767_6d6c_u32 as i32,
        vocab,
        1500,
        audio_state,
        20,
        audio_layer,
        448,
        audio_state,
        20,
        text_layer,
        128,
        8,
    ];
    numbers.iter().flat_map(|n| n.to_le_bytes()).collect()
}

#[test]
fn the_word_timing_preset_is_read_from_the_shape_of_the_model() {
    let t = Temp::new();
    let file = t.0.join("ggml.bin");
    // the shape of cstr/whisper-large-v3-turbo-german-ggml (read from ggml-model-q5_0.bin, 2026-11-10)
    std::fs::write(&file, ggml_header(51866, 1280, 32, 4)).unwrap();
    assert_eq!(dtw_preset_of(&file), Some("large.v3.turbo"));
    std::fs::write(&file, ggml_header(51866, 1280, 32, 32)).unwrap();
    assert_eq!(dtw_preset_of(&file), Some("large.v3"));
    std::fs::write(&file, ggml_header(51864, 1024, 24, 24)).unwrap();
    assert_eq!(dtw_preset_of(&file), Some("medium.en"));
    // large-v1 and large-v2 cannot be told apart, an unknown shape, a file that is no ggml file, no file
    std::fs::write(&file, ggml_header(51865, 1280, 32, 32)).unwrap();
    assert_eq!(dtw_preset_of(&file), None);
    std::fs::write(&file, ggml_header(51866, 1280, 32, 2)).unwrap();
    assert_eq!(dtw_preset_of(&file), None);
    std::fs::write(&file, "ok\n").unwrap();
    assert_eq!(dtw_preset_of(&file), None);
    assert_eq!(dtw_preset_of(&t.0.join("missing.bin")), None);
    assert_eq!(WordTiming::from_setting("auto"), WordTiming::Auto);
    assert_eq!(WordTiming::from_setting("off"), WordTiming::Off);
    assert_eq!(
        WordTiming::from_setting("large.v3"),
        WordTiming::Preset("large.v3".into())
    );
}

#[test]
fn the_form_carries_the_text_only_as_fields() {
    let hostile = "name \"quoted\" $(touch x)\r\n--boundary";
    let mut request = request();
    request.prompt = Some(hostile.into());
    request.temperature = Some(0.2);
    let fields = inference_fields(&request, true, 0, false);
    assert!(fields.contains(&("response_format", "verbose_json".to_owned())));
    assert!(
        fields.contains(&("split_on_word", "true".to_owned())),
        "a segment must not end inside a word"
    );
    assert!(
        !fields.iter().any(|(name, _)| *name == "max_len"),
        "0 leaves the program's own wrapping"
    );
    assert!(fields.contains(&("language", "de".to_owned())));
    assert!(fields.contains(&("temperature", "0.2".to_owned())));
    // the settings change the fields
    // with word timing the program does not wrap at all: the service cuts at word times
    let timed = inference_fields(&request, true, 120, true);
    assert!(timed.contains(&("max_len", NO_WRAP.to_string())));
    assert_eq!(timed.iter().filter(|(n, _)| *n == "max_len").count(), 1);
    let fields = inference_fields(&request, false, 120, false);
    assert!(fields.contains(&("split_on_word", "false".to_owned())));
    assert!(fields.contains(&("max_len", "120".to_owned())));
    let body = multipart_body("XYZ", &fields, b"RIFFdata");
    let text = String::from_utf8_lossy(&body);
    assert!(text.starts_with("--XYZ\r\n"));
    assert!(text.contains("name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFdata\r\n--XYZ--\r\n"));
    assert!(text.ends_with("--XYZ--\r\n"));
}

#[test]
fn the_answer_of_the_program_is_read_from_verbose_json() {
    let value = serde_json::json!({
        "language": "german",
        "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": " Hallo Welt "}],
    });
    let parsed = parse_transcription(&value).unwrap();
    assert_eq!(parsed.language.as_deref(), Some("german"));
    assert_eq!(parsed.segments[0].end, 2.5);
    assert_eq!(parsed.segments[0].text, "Hallo Welt");
    assert!(parse_transcription(&serde_json::json!({"text": "x"})).is_none());
    assert!(parse_transcription(&serde_json::json!({"segments": [{"start": 0.0}]})).is_none());
    assert!(parsed.segments[0].words.is_empty(), "no tokens, no words");
}

/// A token of `verbose_json` (`t_dtw` in centiseconds).
fn token(text: &str, t_dtw: i64) -> serde_json::Value {
    serde_json::json!({"word": text, "start": 0.0, "end": 0.0, "t_dtw": t_dtw, "probability": 0.9})
}

#[test]
fn the_words_are_put_together_from_the_tokens_with_their_dtw_times() {
    let value = serde_json::json!({
        "segments": [
            {"start": 0.0, "end": 3.0, "text": " Und Gott sprach: Licht.", "words": [
                token(" Und", 52), token(" Gott", 80), token(" spr", 120), token("ach", 140), token(":", 150),
                token(" Licht", 260), token(".", 280),
            ]},
            {"start": 3.0, "end": 6.0, "text": " Es ward.", "words": [
                token(" Es", 460), token(" ward", 480), token(".", 500),
            ]},
        ],
    });
    let parsed = parse_transcription(&value).unwrap();
    let words: Vec<(&str, f64, f64)> = parsed
        .segments
        .iter()
        .flat_map(|s| s.words.iter().map(|w| (w.text.as_str(), w.start, w.end)))
        .collect();
    let texts: Vec<&str> = words.iter().map(|w| w.0).collect();
    assert_eq!(texts, ["Und", "Gott", "sprach:", "Licht.", "Es", "ward."]);
    // a word starts at the DTW time of its first token
    // a word starts at the DTW time of its first token, 0.1 s earlier (DTW hears a token late)
    let starts: Vec<f64> = words.iter().map(|w| w.1).collect();
    let expected = [0.42, 0.7, 1.1, 2.5, 4.5, 4.7];
    for (got, want) in starts.iter().zip(expected) {
        assert!((got - want).abs() < 1e-9, "{starts:?}");
    }
    // a word lasts its estimated length ("Und": 0.15 + 3 * 0.03 s), but never reaches into the next word ("Es" would
    // last until 4.71, "ward" starts at 4.7)
    assert!((words[0].2 - 0.66).abs() < 1e-9 && (words[1].2 - 0.97).abs() < 1e-9);
    assert!((words[4].2 - 4.7).abs() < 1e-9, "{:?}", words[4]);
    // "sprach:" would last 0.15 + 7 * 0.03 s, until 1.46, but its last token (":") is heard at 1.4: until 1.5
    assert!((words[2].2 - 1.5).abs() < 1e-9, "{}", words[2].2);
    // "Licht." (6 characters) 0.33 s, also at the end of a segment
    assert!((words[3].2 - 2.83).abs() < 1e-9, "{}", words[3].2);
    // the last word: its estimated length, at least until 0.1 s after its last token ("." at 5.0 - 0.1)
    let last = words[5];
    assert!((last.2 - 5.0).abs() < 1e-9, "{last:?}");
    for pair in words.windows(2) {
        assert!(pair[0].2 <= pair[1].1 && pair[0].1 <= pair[0].2);
    }
}

#[test]
fn without_dtw_times_no_segment_has_words() {
    // the program without --dtw (or with flash attention) sends t_dtw −1: the segments keep the program's times
    let value = serde_json::json!({
        "segments": [
            {"start": 0.0, "end": 3.0, "text": " Und Gott", "words": [token(" Und", 52), token(" Gott", 80)]},
            {"start": 3.0, "end": 6.0, "text": " sprach", "words": [token(" sprach", -1)]},
        ],
    });
    let parsed = parse_transcription(&value).unwrap();
    assert!(parsed.segments.iter().all(|s| s.words.is_empty()));
    assert_eq!(parsed.segments[1].start, 3.0);
    assert_eq!(parsed.segments[0].text, "Und Gott");
}

#[tokio::test]
async fn a_model_is_started_transcribes_and_is_stopped() {
    let t = Temp::new();
    let range = ports(2);
    let b = backend(model(&t.0, "ok"), range);
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    assert!(running.generation >= GENERATION_BASE);
    assert!(b.health(&running).await.unwrap());
    let mut asked = request();
    asked.prompt = Some("Jesaja".into());
    asked.temperature = Some(0.0);
    let done = b.transcribe(&running, asked).await.unwrap();
    assert_eq!(done.language.as_deref(), Some("german"));
    assert_eq!(done.segments.len(), 2);
    assert_eq!(
        done.segments[0].text,
        "lang=de prompt=Jesaja temperature=0 format=verbose_json"
    );
    assert_eq!(done.segments[1].start, 1.5);
    let sent = recorded(&t.0, "request");
    assert!(sent.contains("name=\"file\"; filename=\"audio.wav\"") && sent.contains("RIFF"));
    // what the program was started with
    let args = recorded(&t.0, "args");
    let args: Vec<&str> = args.lines().collect();
    assert_eq!(args[0], "--model");
    assert_eq!(args[2], "--host");
    assert_eq!(args[3], "127.0.0.1");
    let port: u16 = args[5].parse().unwrap();
    assert!((range.0..=range.1).contains(&port));
    assert!(args.contains(&"--flash-attn") && args.contains(&"--device"));
    assert!(!args.contains(&"--no-gpu") && !args.iter().any(|a| a.contains("convert")));
    // the run directory is private, is the working directory, and is gone after the stop
    let directory = private_directory(port);
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        std::fs::canonicalize(recorded(&t.0, "cwd")).unwrap(),
        std::fs::canonicalize(&directory).unwrap()
    );
    b.stop(&running).await.unwrap();
    assert!(!directory.exists());
    assert!(!b.health(&running).await.unwrap());
    assert_eq!(
        b.transcribe(&running, request()).await,
        Err(BackendError::Crashed)
    );
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        std::net::TcpListener::bind(("127.0.0.1", port)).is_ok(),
        "the port is free again"
    );
}

#[tokio::test]
async fn a_model_placed_on_the_cpu_is_started_without_the_gpu() {
    let t = Temp::new();
    let b = backend(model(&t.0, "ok"), ports(1));
    let running = b.start(&spec(DevicePolicy::Cpu)).await.unwrap();
    assert_eq!(running.spec.device, DevicePolicy::Cpu);
    let args = recorded(&t.0, "args");
    assert!(args.lines().any(|a| a == "--no-gpu"));
    assert!(!args.lines().any(|a| a == "--device" || a == "--flash-attn"));
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn the_child_gets_a_cleared_environment() {
    let t = Temp::new();
    // SAFETY: the variable is set before any child starts and no other test reads it.
    unsafe { std::env::set_var("ELIEZER_SECRET_FOR_TEST", "hf_secret") };
    let b = backend(model(&t.0, "ok"), ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    let names = recorded(&t.0, "env");
    assert!(!names.contains("ELIEZER_SECRET_FOR_TEST"), "{names}");
    assert!(!names.contains("HF_TOKEN"));
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn bad_requests_and_wrong_use_are_refused() {
    let t = Temp::new();
    let b = backend(model(&t.0, "ok"), ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    let mut short = request();
    short.wav = vec![1, 2, 3];
    assert_eq!(
        b.transcribe(&running, short).await,
        Err(BackendError::InvalidRequest)
    );
    let mut bad_language = request();
    bad_language.language = Some("de\r\nx".into());
    assert_eq!(
        b.transcribe(&running, bad_language).await,
        Err(BackendError::InvalidRequest)
    );
    let chat = ChatRequest {
        model: "stt".into(),
        input: "hi".into(),
        reasoning: None,
    };
    assert_eq!(
        b.chat(&running, chat).await.err(),
        Some(BackendError::InvalidRequest)
    );
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn answers_that_are_no_transcript_are_mapped_to_errors() {
    for (mode, expected) in [
        ("error-500", BackendError::Crashed),
        ("bad-request", BackendError::InvalidRequest),
        ("no-segments", BackendError::Crashed),
    ] {
        let t = Temp::new();
        let b = backend(model(&t.0, mode), ports(1));
        let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
        assert_eq!(
            b.transcribe(&running, request()).await,
            Err(expected),
            "{mode}"
        );
        b.stop(&running).await.unwrap();
    }
}

#[tokio::test]
async fn a_program_that_dies_is_noticed() {
    let t = Temp::new();
    let b = backend(model(&t.0, "die-after-first"), ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    assert!(b.transcribe(&running, request()).await.is_ok());
    for _ in 0..50 {
        if !b.health(&running).await.unwrap() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!b.health(&running).await.unwrap());
    assert_eq!(
        b.transcribe(&running, request()).await,
        Err(BackendError::Crashed)
    );
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn a_slow_load_is_waited_for_and_starting_fails_cleanly() {
    // the program listens only after loading
    let t = Temp::new();
    let b = backend(model(&t.0, "slow-load 800"), ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    b.stop(&running).await.unwrap();

    // a program that dies while loading
    let t = Temp::new();
    let range = ports(1);
    let b = backend(model(&t.0, "crash"), range);
    assert_eq!(
        b.start(&spec(DevicePolicy::Gpu)).await.err(),
        Some(BackendError::Crashed)
    );
    assert!(!private_directory(range.0).exists());

    // never ready: gives up at the deadline
    let t = Temp::new();
    let range = ports(1);
    let b = backend_with(
        model(&t.0, "never-ready"),
        range,
        Duration::from_millis(900),
    );
    assert_eq!(
        b.start(&spec(DevicePolicy::Gpu)).await.err(),
        Some(BackendError::Unavailable)
    );
    assert!(!private_directory(range.0).exists());

    // a missing ggml file, an unknown model, a program that does not exist
    let t = Temp::new();
    let mut missing = model(&t.0, "ok");
    missing.file = t.0.join("gone.bin");
    let b = backend(missing, ports(1));
    assert_eq!(
        b.start(&spec(DevicePolicy::Gpu)).await.err(),
        Some(BackendError::ModelFile)
    );
    let mut other = spec(DevicePolicy::Gpu);
    other.model = "unknown".into();
    assert_eq!(b.start(&other).await.err(), Some(BackendError::Unavailable));
    let t = Temp::new();
    let b = WhispercppBackend::new(
        "/nonexistent/whisper-server".into(),
        ports(1),
        ProgramSettings::default(),
    );
    b.set_models(HashMap::from([("stt".to_owned(), model(&t.0, "ok"))]));
    assert_eq!(
        b.start(&spec(DevicePolicy::Gpu)).await.err(),
        Some(BackendError::Unavailable)
    );

    // a language that is not a language code never reaches the command line
    let t = Temp::new();
    let mut hostile = model(&t.0, "ok");
    hostile.language = "de; rm -rf".into();
    let b = backend(hostile, ports(1));
    assert_eq!(
        b.start(&spec(DevicePolicy::Gpu)).await.err(),
        Some(BackendError::InvalidRequest)
    );
}

#[tokio::test]
async fn instances_use_their_own_ports_and_the_range_is_respected() {
    let t1 = Temp::new();
    let t2 = Temp::new();
    let range = ports(2);
    let b = backend(model(&t1.0, "ok"), range);
    b.set_models(HashMap::from([
        ("stt".to_owned(), model(&t1.0, "ok")),
        ("stt2".to_owned(), model(&t2.0, "ok")),
    ]));
    let first = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    let mut second_spec = spec(DevicePolicy::Gpu);
    second_spec.model = "stt2".into();
    let second = b.start(&second_spec).await.unwrap();
    assert_ne!(first.generation, second.generation);
    let mut third_spec = spec(DevicePolicy::Gpu);
    third_spec.model = "stt".into();
    assert_eq!(
        b.start(&third_spec).await.err(),
        Some(BackendError::Unavailable),
        "the range holds two"
    );
    b.stop(&first).await.unwrap();
    b.stop(&second).await.unwrap();
}

#[tokio::test]
async fn the_segment_settings_of_the_model_reach_the_program() {
    let t = Temp::new();
    let mut custom = model(&t.0, "ok");
    custom.split_on_word = false;
    custom.max_segment_chars = 90;
    let b = backend(custom, ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    b.transcribe(&running, request()).await.unwrap();
    let sent = recorded(&t.0, "request");
    assert!(sent.contains("name=\"split_on_word\"\r\n\r\nfalse\r\n"));
    assert!(sent.contains("name=\"max_len\"\r\n\r\n90\r\n"));
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn with_word_timing_the_program_gets_the_preset_and_the_words_come_back() {
    let t = Temp::new();
    let mut timed = model(&t.0, "ok");
    timed.word_timing = WordTiming::Preset("large.v3.turbo".into());
    timed.max_segment_chars = 90;
    let b = backend(timed, ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    let args = recorded(&t.0, "args");
    let args: Vec<&str> = args.lines().collect();
    assert!(args.windows(2).any(|w| w == ["--dtw", "large.v3.turbo"]));
    assert!(args.contains(&"--no-flash-attn") && !args.contains(&"--flash-attn"));
    let done = b.transcribe(&running, request()).await.unwrap();
    let sent = recorded(&t.0, "request");
    assert!(sent.contains(&format!("name=\"max_len\"\r\n\r\n{NO_WRAP}\r\n")));
    assert!(done.segments.iter().all(|s| !s.words.is_empty()));
    let second = &done.segments[1];
    let texts: Vec<&str> = second.words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, ["second"], "the two tokens make one word");
    assert!((second.words[0].start - 1.6).abs() < 1e-9);
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn a_preset_that_does_not_fit_falls_back_to_the_times_of_the_program() {
    let t = Temp::new();
    let mut timed = model(&t.0, "crash-with-dtw");
    timed.word_timing = WordTiming::Preset("large.v3".into());
    let b = backend(timed, ports(2));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    let args = recorded(&t.0, "args");
    assert!(!args.contains("--dtw"), "started once more without: {args}");
    assert!(args.contains("--flash-attn"), "the setting applies again");
    let done = b.transcribe(&running, request()).await.unwrap();
    assert!(done.segments.iter().all(|s| s.words.is_empty()));
    let sent = recorded(&t.0, "request");
    assert!(
        !sent.contains("name=\"max_len\""),
        "the program's own wrapping"
    );
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn auto_without_a_fitting_shape_and_missing_dtw_times_keep_the_old_behaviour() {
    // `auto` with a file whose header is no ggml shape: no --dtw
    let t = Temp::new();
    let mut auto = model(&t.0, "ok");
    auto.word_timing = WordTiming::Auto;
    let b = backend(auto, ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    assert!(!recorded(&t.0, "args").contains("--dtw"));
    let done = b.transcribe(&running, request()).await.unwrap();
    assert!(done.segments.iter().all(|s| s.words.is_empty()));
    b.stop(&running).await.unwrap();
    // started with --dtw, but the answers carry no DTW times: no words, no failure
    let t = Temp::new();
    let mut timed = model(&t.0, "no-dtw-times");
    timed.word_timing = WordTiming::Preset("large.v3.turbo".into());
    let b = backend(timed, ports(1));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    assert!(recorded(&t.0, "args").contains("--dtw"));
    let done = b.transcribe(&running, request()).await.unwrap();
    assert_eq!(done.segments.len(), 2);
    assert!(done.segments.iter().all(|s| s.words.is_empty()));
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn the_stop_grace_time_is_a_setting() {
    let t = Temp::new();
    let b = backend(model(&t.0, "ok"), ports(1)).with_stop_grace(Duration::from_secs(2));
    let running = b.start(&spec(DevicePolicy::Gpu)).await.unwrap();
    b.stop(&running).await.unwrap();
}

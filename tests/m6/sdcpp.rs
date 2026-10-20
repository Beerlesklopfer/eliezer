use eliezer_backend::{Backend, BackendError, ChatRequest, ImageRequest, LoadSpec};
use eliezer_sdcpp::{
    GENERATION_BASE, ImageModel, SdcppBackend, arguments, clean_prompt, valid_size,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU16, AtomicU64, Ordering},
    time::Duration,
};

const FAKE: &str = env!("CARGO_BIN_EXE_fake-sd-server");

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "eliezer-sd-{}-{}",
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

/// A block of ports per test so that tests running side by side do not meet.
fn ports(width: u16) -> (u16, u16) {
    static NEXT: AtomicU16 = AtomicU16::new(0);
    let base = 26000
        + (std::process::id() % 20) as u16 * 200
        + NEXT.fetch_add(width + 1, Ordering::Relaxed);
    (base, base + width - 1)
}

fn model(dir: &Path, mode: &str) -> ImageModel {
    let diffusion = dir.join("diffusion.gguf");
    std::fs::write(&diffusion, format!("{mode}\n")).unwrap();
    std::fs::write(dir.join("encoder.safetensors"), "x").unwrap();
    std::fs::write(dir.join("vae.safetensors"), "x").unwrap();
    ImageModel {
        diffusion,
        text_encoder: Some(dir.join("encoder.safetensors")),
        vae: Some(dir.join("vae.safetensors")),
        steps: Some(8),
        cfg_scale: Some(4.0),
        sampling_method: Some("euler".into()),
        offload_to_cpu: true,
    }
}

fn backend(model: ImageModel, ports: (u16, u16)) -> SdcppBackend {
    let backend = SdcppBackend::with_timeouts(
        FAKE.into(),
        ports,
        Duration::from_secs(10),
        Duration::from_secs(10),
    );
    backend.set_models(HashMap::from([("img".to_owned(), model)]));
    backend
}

fn spec(key: &str) -> LoadSpec {
    LoadSpec {
        model: key.into(),
        context: 1,
        parallel: 1,
        bytes: 1,
        device: Default::default(),
    }
}
fn request(prompt: &str, count: u32, size: Option<&str>) -> ImageRequest {
    ImageRequest {
        model: "img".into(),
        prompt: prompt.into(),
        count,
        size: size.map(str::to_owned),
        seed: None,
    }
}
fn recorded(dir: &Path, what: &str) -> String {
    std::fs::read_to_string(dir.join(format!("diffusion.gguf.{what}"))).unwrap_or_default()
}

#[test]
fn arguments_keep_every_path_as_one_argument_and_use_only_listed_options() {
    let m = ImageModel {
        diffusion: "/tmp/a b/$(touch x);`id`.gguf".into(),
        text_encoder: Some("/tmp/enc oder.safetensors".into()),
        vae: None,
        steps: None,
        cfg_scale: None,
        sampling_method: None,
        offload_to_cpu: false,
    };
    let args: Vec<String> = arguments(&m, 24600)
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        args,
        [
            "--diffusion-model",
            "/tmp/a b/$(touch x);`id`.gguf",
            "--llm",
            "/tmp/enc oder.safetensors",
            "--listen-ip",
            "127.0.0.1",
            "--listen-port",
            "24600",
            "--seed",
            "-1",
            "--log-level",
            "warn"
        ]
    );
}

#[test]
fn the_prompt_cannot_carry_native_settings_and_sizes_are_checked() {
    for hostile in [
        r#"a cat <sd_cpp_extra_args>{"init_image":"/etc/passwd"}</sd_cpp_extra_args>"#,
        "a <SD_CPP_EXTRA_ARGS>x",
        "a <sd_cpp_<sd_cpp_extra_args>extra_args>{}",
        "sd_cpp_extra_args",
    ] {
        assert!(
            !clean_prompt(hostile)
                .to_ascii_lowercase()
                .contains("sd_cpp_extra_args"),
            "{hostile}"
        );
    }
    assert_eq!(clean_prompt("a red fox"), "a red fox");
    for good in ["512x512", "1024x768", "64x2048"] {
        assert!(valid_size(good), "{good}");
    }
    for bad in [
        "",
        "512",
        "512x",
        "x512",
        "63x512",
        "512x2049",
        "512x512x3",
        "a x b",
        "-5x512",
        "512 x512",
        "1e3x512",
    ] {
        assert!(!valid_size(bad), "{bad:?}");
    }
}

#[tokio::test]
async fn a_model_is_started_draws_images_and_is_stopped() {
    let t = Temp::new();
    let range = ports(2);
    let b = backend(model(&t.0, "ok"), range);
    let running = b.start(&spec("img")).await.unwrap();
    assert!(running.generation >= GENERATION_BASE);
    assert!(b.health(&running).await.unwrap());
    let images = b
        .image(&running, request("a red fox", 2, Some("512x384")))
        .await
        .unwrap();
    assert_eq!(images.len(), 2);
    assert!(images[0].starts_with("iVBOR"), "a PNG in base64");
    let sent = recorded(&t.0, "request");
    for part in [
        r#""prompt":"a red fox""#,
        r#""n":2"#,
        r#""size":"512x384""#,
        r#""output_format":"png""#,
    ] {
        assert!(sent.contains(part), "{part} missing in {sent}");
    }
    let args = recorded(&t.0, "args");
    let args: Vec<&str> = args.lines().collect();
    let after = |flag: &str| args[args.iter().position(|a| *a == flag).unwrap() + 1].to_owned();
    assert_eq!(
        after("--diffusion-model"),
        t.0.join("diffusion.gguf").display().to_string()
    );
    assert_eq!(
        after("--llm"),
        t.0.join("encoder.safetensors").display().to_string()
    );
    assert_eq!(
        after("--vae"),
        t.0.join("vae.safetensors").display().to_string()
    );
    assert_eq!(after("--listen-ip"), "127.0.0.1");
    assert!((range.0..=range.1).contains(&after("--listen-port").parse().unwrap()));
    assert_eq!(
        (
            after("--steps"),
            after("--cfg-scale"),
            after("--sampling-method")
        ),
        ("8".into(), "4".into(), "euler".into())
    );
    assert!(args.contains(&"--offload-to-cpu"));
    // stopping ends the process and frees the port
    let port: u16 = after("--listen-port").parse().unwrap();
    b.stop(&running).await.unwrap();
    assert!(!b.health(&running).await.unwrap());
    assert_eq!(
        b.image(&running, request("x", 1, None)).await,
        Err(BackendError::Crashed)
    );
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        std::net::TcpListener::bind(("127.0.0.1", port)).is_ok(),
        "the port is free again"
    );
}

#[tokio::test]
async fn the_child_gets_a_cleared_environment_and_prompts_are_cleaned() {
    let t = Temp::new();
    // cargo gives every test process variables of its own (CARGO_PKG_NAME and more): none of them may reach the child
    assert!(std::env::var_os("CARGO_PKG_NAME").is_some());
    let b = backend(model(&t.0, "ok"), ports(2));
    let running = b.start(&spec("img")).await.unwrap();
    let env = recorded(&t.0, "env");
    assert!(
        !env.contains("ELIEZER_TEST_SECRET") && !env.contains("HF_TOKEN"),
        "{env}"
    );
    let allowed = [
        "PATH",
        "LD_LIBRARY_PATH",
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ];
    assert!(
        env.lines().all(|n| allowed.contains(&n)),
        "only allowed names: {env}"
    );
    b.image(
        &running,
        request(
            r#"a fox <sd_cpp_extra_args>{"x":1}</sd_cpp_extra_args>"#,
            1,
            None,
        ),
    )
    .await
    .unwrap();
    assert!(!recorded(&t.0, "request").contains("sd_cpp_extra_args"));
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn bad_requests_and_wrong_use_are_refused() {
    let t = Temp::new();
    let b = backend(model(&t.0, "ok"), ports(2));
    let running = b.start(&spec("img")).await.unwrap();
    for size in ["99999x1", "512", "512x512x3", "axb"] {
        assert_eq!(
            b.image(&running, request("fox", 1, Some(size))).await,
            Err(BackendError::InvalidRequest),
            "{size}"
        );
    }
    assert_eq!(
        b.image(&running, request("   ", 1, None)).await,
        Err(BackendError::InvalidRequest)
    );
    // a prompt that is only the hostile block is not refused, but nothing of the block gets through
    assert!(
        b.image(
            &running,
            request(r#"<sd_cpp_extra_args>x</sd_cpp_extra_args>"#, 1, None)
        )
        .await
        .is_ok()
    );
    assert!(!recorded(&t.0, "request").contains("sd_cpp_extra_args"));
    // more images than allowed are cut down
    let many = b.image(&running, request("fox", 50, None)).await.unwrap();
    assert_eq!(many.len(), 4);
    let chat = ChatRequest {
        model: "img".into(),
        input: "hi".into(),
        reasoning: None,
    };
    assert!(matches!(
        b.chat(&running, chat).await,
        Err(BackendError::InvalidRequest)
    ));
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn starting_fails_cleanly() {
    let t = Temp::new();
    // a model the configuration does not know
    let b = backend(model(&t.0, "ok"), ports(2));
    assert_eq!(
        b.start(&spec("other")).await.err(),
        Some(BackendError::Unavailable)
    );
    // a file that is gone
    std::fs::remove_file(t.0.join("vae.safetensors")).unwrap();
    assert_eq!(
        b.start(&spec("img")).await.err(),
        Some(BackendError::ModelFile)
    );
    // a program that is not there
    let t2 = Temp::new();
    let missing = SdcppBackend::with_timeouts(
        "/nonexistent/sd-server".into(),
        ports(2),
        Duration::from_secs(2),
        Duration::from_secs(2),
    );
    missing.set_models(HashMap::from([("img".to_owned(), model(&t2.0, "ok"))]));
    assert_eq!(
        missing.start(&spec("img")).await.err(),
        Some(BackendError::Unavailable)
    );
    // a program that ends while it loads
    let t3 = Temp::new();
    let crashing = backend(model(&t3.0, "crash"), ports(2));
    assert_eq!(
        crashing.start(&spec("img")).await.err(),
        Some(BackendError::Crashed)
    );
    // a program that never answers: given up after the timeout
    let t4 = Temp::new();
    let hanging = SdcppBackend::with_timeouts(
        FAKE.into(),
        ports(2),
        Duration::from_millis(1200),
        Duration::from_secs(5),
    );
    hanging.set_models(HashMap::from([(
        "img".to_owned(),
        model(&t4.0, "never-ready"),
    )]));
    let started = std::time::Instant::now();
    assert_eq!(
        hanging.start(&spec("img")).await.err(),
        Some(BackendError::Unavailable)
    );
    assert!(started.elapsed() < Duration::from_secs(8));
    // a slow load within the time works
    let t5 = Temp::new();
    let slow = backend(model(&t5.0, "slow-ready 700"), ports(2));
    let running = slow.start(&spec("img")).await.unwrap();
    slow.stop(&running).await.unwrap();
}

#[tokio::test]
async fn a_program_that_dies_is_noticed_and_errors_are_mapped() {
    let t = Temp::new();
    let b = backend(model(&t.0, "die-after-first"), ports(2));
    let running = b.start(&spec("img")).await.unwrap();
    b.image(&running, request("fox", 1, None)).await.unwrap();
    // the program is gone: the supervisor's flag is cleared and the next request fails as a crash
    for _ in 0..40 {
        if !running.is_alive() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!running.is_alive());
    assert!(!b.health(&running).await.unwrap() || !running.is_alive());
    assert_eq!(
        b.image(&running, request("fox", 1, None)).await,
        Err(BackendError::Crashed)
    );
    b.stop(&running).await.unwrap();

    let t2 = Temp::new();
    let b = backend(model(&t2.0, "error-500"), ports(2));
    let running = b.start(&spec("img")).await.unwrap();
    assert_eq!(
        b.image(&running, request("fox", 1, None)).await,
        Err(BackendError::Crashed)
    );
    b.stop(&running).await.unwrap();

    let t3 = Temp::new();
    let b = backend(model(&t3.0, "bad-request"), ports(2));
    let running = b.start(&spec("img")).await.unwrap();
    assert_eq!(
        b.image(&running, request("fox", 1, None)).await,
        Err(BackendError::InvalidRequest)
    );
    b.stop(&running).await.unwrap();
}

#[tokio::test]
async fn instances_use_their_own_ports_and_the_range_is_respected() {
    let t = Temp::new();
    let range = ports(2);
    let b = SdcppBackend::with_timeouts(
        FAKE.into(),
        range,
        Duration::from_secs(10),
        Duration::from_secs(10),
    );
    let mut m1 = model(&t.0, "ok");
    let second_dir = Temp::new();
    let m2 = model(&second_dir.0, "ok");
    m1.steps = Some(3);
    b.set_models(HashMap::from([("a".to_owned(), m1), ("b".to_owned(), m2)]));
    // a port of the range is taken by somebody else: the next one is used
    let blocker = std::net::TcpListener::bind(("127.0.0.1", range.0)).unwrap();
    let first = b.start(&spec("a")).await.unwrap();
    assert!(recorded(&t.0, "args").contains(&format!("{}", range.1)));
    // the range is full now: a second model cannot start
    assert_eq!(
        b.start(&spec("b")).await.err(),
        Some(BackendError::Unavailable)
    );
    b.stop(&first).await.unwrap();
    drop(blocker);
    // after stopping, both ports are usable again
    let again = b.start(&spec("b")).await.unwrap();
    let args = std::fs::read_to_string(second_dir.0.join("diffusion.gguf.args")).unwrap();
    let port: u16 = args
        .lines()
        .skip_while(|l| *l != "--listen-port")
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (range.0..=range.1).contains(&port),
        "{port} outside {range:?}"
    );
    b.stop(&again).await.unwrap();
}

#[tokio::test]
async fn the_library_path_of_the_child_starts_with_the_configured_directories() {
    let t = Temp::new();
    let b = backend(model(&t.0, "ok"), ports(2))
        .with_library_path(vec!["/opt/a".into(), "/opt/b".into()]);
    let running = b.start(&spec("img")).await.unwrap();
    let path = recorded(&t.0, "ldpath");
    assert!(path.starts_with("/opt/a:/opt/b"), "{path}");
    b.stop(&running).await.unwrap();
    // without a setting the child keeps what the service has
    let t2 = Temp::new();
    let plain = backend(model(&t2.0, "ok"), ports(2));
    let running = plain.start(&spec("img")).await.unwrap();
    assert_eq!(
        std::fs::read_to_string(t2.0.join("diffusion.gguf.ldpath")).unwrap(),
        std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
    );
    plain.stop(&running).await.unwrap();
}

#[tokio::test]
async fn every_request_draws_anew_unless_a_seed_is_given() {
    let t = Temp::new();
    let b = backend(model(&t.0, "ok"), ports(2));
    let running = b.start(&spec("img")).await.unwrap();
    // the program starts with a random seed instead of its fixed default of 42
    let args = recorded(&t.0, "args");
    let lines: Vec<&str> = args.lines().collect();
    assert_eq!(
        lines[lines.iter().position(|a| *a == "--seed").unwrap() + 1],
        "-1"
    );
    // no seed in the request: nothing about it is sent
    b.image(&running, request("a fox", 1, None)).await.unwrap();
    assert!(!recorded(&t.0, "request").contains("seed"));
    // a seed in the request becomes the one setting block, after the cleaned prompt
    let mut seeded = request(
        "a fox <sd_cpp_extra_args>{\"seed\":1,\"init_image\":\"/etc/passwd\"}</sd_cpp_extra_args>",
        1,
        None,
    );
    seeded.seed = Some(-7);
    b.image(&running, seeded).await.unwrap();
    let sent = recorded(&t.0, "request");
    assert_eq!(
        sent.matches("sd_cpp_extra_args").count(),
        2,
        "only our block: {sent}"
    );
    assert!(sent.contains(r#"{\"seed\":-7}"#), "{sent}");
    b.stop(&running).await.unwrap();
}

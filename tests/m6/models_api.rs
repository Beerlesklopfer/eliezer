use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use eliezer_api::{App, managed_router};
use eliezer_backend::{supervisor::*, *};
use eliezer_core::{config::Config, scheduler::Clock};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

struct TestClock;
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn an_image_model_has_a_size_from_its_files_and_no_context_window() {
    let dir = std::env::temp_dir().join(format!("eliezer-models-api-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("models/te")).unwrap();
    let write = |name: &str, bytes: usize| {
        std::fs::write(dir.join("models").join(name), vec![0_u8; bytes]).unwrap()
    };
    write("diffusion.gguf", 4_000);
    write("te/encoder.safetensors", 9_000);
    write("vae.safetensors", 700);
    write("chat.gguf", 1_000);
    write("mmproj-chat.gguf", 300);
    let models = dir.join("models").display().to_string();
    let config = Arc::new(
        Config::parse(&format!(
            r#"
[server]
listen="127.0.0.1:1234"
[storage]
models_dir="{models}"
[gpu]
vendor="none"
measure="none"
[huggingface]
cache_dir="{models}/hf"
[backend.sdcpp]
program="/usr/bin/sd-server"
port_range=[24600,24609]
[backend.llamacpp]
enabled=true
[[model]]
key="pic"
backend="sdcpp"
kind="image"
source={{file="{models}/diffusion.gguf"}}
vram_gb=9.0
extra={{text_encoder="{models}/te/encoder.safetensors",vae="{models}/vae.safetensors"}}
[[model]]
key="chat"
backend="llamacpp"
kind="llm"
vision=true
source={{file="{models}/chat.gguf",projector="{models}/mmproj-chat.gguf"}}
vram_gb=2.0
arch={{layers=2,kv_heads=2,head_dim=64}}
[[model]]
key="sized"
backend="llamacpp"
kind="llm"
source={{file="{models}/chat.gguf"}}
size_gb=4.4
arch={{layers=2,kv_heads=2,head_dim=64}}
"#
        ))
        .unwrap(),
    );
    let supervisor = Supervisor::spawn(
        vec![],
        Arc::new(FakeBackend::new(Default::default())),
        Arc::new(TestClock),
        Settings {
            budget: 1,
            ram_budget: 0,
            idle_ms: 1,
            wait_ms: 1,
            queue_max: 1,
            retry_ms: 1,
        },
    );
    let app = managed_router(App::new(
        config,
        supervisor,
        Arc::new(FakeBackend::new(Default::default())),
        Arc::new(eliezer_gpu::sources(
            eliezer_core::config::GpuVendor::None,
            eliezer_core::config::ProbeMethod::None,
            0,
            1,
            "/nonexistent".into(),
        )),
        vec![],
    ));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
    let by = |key: &str| {
        body["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["key"] == key)
            .unwrap()
            .clone()
    };
    let pic = by("pic");
    assert_eq!(pic["type"], "image");
    // the diffusion model, the text encoder and the VAE
    assert_eq!(pic["size_bytes"], 4_000 + 9_000 + 700);
    assert!(
        pic.get("max_context_length").is_none(),
        "an image model has no context window"
    );
    // a language model without size_gb: weights and vision projector
    assert_eq!(by("chat")["size_bytes"], 1_300);
    assert!(by("chat")["max_context_length"].is_number());
    // size_gb wins when it is given
    assert_eq!(by("sized")["size_bytes"], 4_400_000_000_u64);
    let _ = std::fs::remove_dir_all(&dir);
}

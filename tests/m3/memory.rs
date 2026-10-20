use eliezer_core::{
    config::{Architecture, Config},
    memory::*,
};
fn arch() -> Architecture {
    Architecture {
        layers: 48,
        kv_heads: 4,
        head_dim: 128,
        swa_layers: None,
        window: None,
        full_kv_heads: None,
        full_head_dim: None,
        linear_layers: None,
        state_bytes: None,
    }
}
#[test]
fn specification_example_and_parallel() {
    assert_eq!(cache_bytes(&arch(), 32768, 1).unwrap(), 3_221_225_472);
    assert_eq!(cache_bytes(&arch(), 32768, 4).unwrap(), 3_221_225_472);
    let config = Config::parse(include_str!("../../packaging/config.example.toml")).unwrap();
    assert_eq!(
        estimate(&config.model[0], 32768, 1, None, None).unwrap(),
        21_021_225_472
    );
    assert_eq!(
        estimate(&config.model[0], 32768, 1, None, Some(42)).unwrap(),
        42
    );
    assert_eq!(
        estimate(&config.model[1], 32768, 1, None, None).unwrap(),
        14_000_000_000
    );
}
#[test]
fn hybrid_architectures_and_overflow() {
    let mut a = arch();
    a.layers = 4;
    a.swa_layers = Some(2);
    a.window = Some(1024);
    a.linear_layers = Some(1);
    a.state_bytes = Some(1000);
    a.full_kv_heads = Some(8);
    a.full_head_dim = Some(256);
    assert_eq!(
        cache_bytes(&a, 32768, 2).unwrap(),
        268_435_456 + 16_777_216 + 2000
    );
    a.window = Some(u32::MAX);
    assert_eq!(
        cache_bytes(&a, 32768, 2).unwrap(),
        268_435_456 + 134_217_728 + 2000
    );
    a.linear_layers = Some(4);
    assert_eq!(
        cache_bytes(&a, 32768, 2),
        Err(MemoryError::InvalidArchitecture)
    );
    a = arch();
    a.layers = u32::MAX;
    a.kv_heads = u32::MAX;
    assert_eq!(cache_bytes(&a, u32::MAX, 1), Err(MemoryError::Overflow));
    assert_eq!(gb_bytes(f64::INFINITY), Err(MemoryError::InvalidSize));
    assert_eq!(cache_bytes(&arch(), 1, 2), Err(MemoryError::InvalidSize));
}
#[test]
fn external_usage_and_reserve() {
    assert_eq!(available(24, 2, 24, 12, 8, 8), 14);
    assert_eq!(available(24, 2, 24, 20, 8, 8), 10);
    assert_eq!(available(24, 2, 8, 0, 0, 0), 6);
    assert_eq!(available(24, 2, 8, 9, 0, 0), 0);
}
#[test]
fn parts_add_up_and_the_largest_fitting_context_is_exact() {
    let config = Config::parse(include_str!("../../packaging/config.example.toml")).unwrap();
    let model = &config.model[0];
    let parts = estimate_parts(model, 32768, 1, None, None).unwrap();
    assert_eq!(parts.basis, Basis::Calculated);
    assert_eq!(parts.cache, 3_221_225_472);
    assert_eq!(
        parts.total,
        parts.weights + parts.cache + parts.overhead + parts.vision
    );
    assert_eq!(
        estimate_parts(model, 32768, 1, None, Some(42))
            .unwrap()
            .basis,
        Basis::Measured
    );
    // The result fits and the next larger context does not.
    let budget = 20_000_000_000;
    let best = max_context_within(model, 1, None, None, budget).unwrap();
    assert!(estimate(model, best, 1, None, None).unwrap() <= budget);
    assert!(
        best == model.max_context || estimate(model, best + 1, 1, None, None).unwrap() > budget
    );
    // Weights alone do not fit: no context fits.
    assert_eq!(
        max_context_within(model, 1, None, None, 1_000_000_000),
        None
    );
}

#[test]
fn a_fixed_figure_is_shown_as_the_weights_so_that_the_parts_add_up() {
    let config = Config::parse(
        r#"
[server]
listen="127.0.0.1:1234"
[gpu]
vendor="none"
[backend.sdcpp]
program="/usr/bin/sd-server"
port_range=[24600,24609]
[[model]]
key="pic"
backend="sdcpp"
kind="image"
source={file="/var/lib/eliezer/models/pic.gguf"}
vram_gb=9.0
"#,
    )
    .unwrap();
    let parts = estimate_parts(&config.model[0], 1024, 1, None, None).unwrap();
    assert_eq!(parts.basis, Basis::Configured);
    assert_eq!(parts.total, 9_000_000_000);
    assert_eq!(parts.weights, 9_000_000_000);
    assert_eq!(
        parts.weights + parts.cache + parts.overhead + parts.vision,
        parts.total
    );
    // a measured value too
    let measured = estimate_parts(&config.model[0], 1024, 1, None, Some(7_000_000_000)).unwrap();
    assert_eq!(
        (measured.basis, measured.weights, measured.total),
        (Basis::Measured, 7_000_000_000, 7_000_000_000)
    );
}

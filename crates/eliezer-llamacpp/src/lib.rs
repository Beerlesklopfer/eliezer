//! Embedded llama.cpp runtime: chat templates, output parsing and, behind the `engine` feature
//! (or one of `rocm`, `vulkan`, `cuda`, `cpu`), the engine and the `Backend` implementation.
pub mod output;
pub mod template;
pub mod vision;

#[cfg(feature = "engine")]
mod backend;
#[cfg(feature = "engine")]
mod engine;

#[cfg(feature = "engine")]
pub use backend::{LlamaCppBackend, ModelFile};

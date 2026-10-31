//! A ROCm build links against AMD's runtime, which AMD installs outside the default library path.
//! The directory is recorded as an rpath so the binary starts without `LD_LIBRARY_PATH`.
//! `ROCM_PATH` overrides the default `/opt/rocm`.

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_ROCM").is_some() {
        let root = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_owned());
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{root}/lib");
    }
}

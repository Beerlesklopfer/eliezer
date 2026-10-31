use eliezer_gpu::*;
use std::{os::unix::fs::PermissionsExt, sync::atomic::AtomicU64};
/// Tests that write an executable script and run it must not overlap: a process forked by
/// another test thread can still hold the freshly written file open, and `exec` then fails
/// with "text file busy", which would show up as a bogus `Unavailable`.
static SCRIPTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn script_lock() -> std::sync::MutexGuard<'static, ()> {
    SCRIPTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
#[test]
fn same_byte_semantics_and_invalid_results() {
    let expected = GpuMemory::new(8 * 1_048_576, 2 * 1_048_576).unwrap();
    let accounting = Accounting {
        budget: expected.total_bytes,
        leases: AtomicU64::new(expected.used_bytes),
    };
    assert_eq!(accounting.memory().unwrap(), expected);
    assert_eq!(parse_nvidia("8, 2\n").unwrap(), expected);
    assert_eq!(parse_rocm(r#"{"card0":{"VRAM Total Memory (B)":"8388608","VRAM Total Used Memory (B)":"2097152"}}"#,0).unwrap(),expected);
    assert_eq!(parse_nvidia("2, 8"), Err(ProbeError::Inconsistent));
    assert_eq!(parse_nvidia("N/A, 2"), Err(ProbeError::InvalidOutput));
}
#[test]
fn fake_sysfs_command_and_fallback() {
    let _lock = script_lock();
    let root = std::env::temp_dir().join(format!(
        "eliezer-probe-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let device = root.join("card0/device");
    std::fs::create_dir_all(&device).unwrap();
    for (name, value) in [
        ("vendor", "0x1002\n"),
        ("mem_info_vram_total", "8388608"),
        ("mem_info_vram_used", "2097152"),
    ] {
        std::fs::write(device.join(name), value).unwrap();
    }
    assert_eq!(
        detect_vendor(&root, 0),
        eliezer_core::config::GpuVendor::Amd
    );
    let sysfs = Sysfs {
        root: root.clone(),
        index: 0,
    };
    let program = root.join("smi");
    std::fs::write(&program, "#!/bin/sh\nprintf '8, 2\\n'\n").unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
    let smi = Smi {
        program: program.clone(),
        index: 0,
        kind: SmiKind::Nvidia,
    };
    assert_eq!(sysfs.memory(), smi.memory());
    std::fs::remove_file(device.join("mem_info_vram_total")).unwrap();
    let fallback = Fallback::new(vec![Box::new(sysfs), Box::new(smi)]);
    assert!(fallback.memory().is_ok());
    assert_eq!(fallback.active_name(), "nvidia-smi");
    std::fs::remove_dir_all(root).unwrap();
}

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "eliezer-probe-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
fn script(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
    let program = dir.join("smi");
    std::fs::write(&program, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
    program
}
#[test]
fn rocm_smi_script_gives_same_bytes_as_nvidia_smi() {
    let _lock = script_lock();
    let dir = temp("rocm");
    let rocm = Smi {
        program: script(
            &dir,
            r#"printf '%s' '{"card0":{"VRAM Total Memory (B)":"8388608","VRAM Total Used Memory (B)":"2097152"}}'"#,
        ),
        index: 0,
        kind: SmiKind::Rocm,
    };
    assert_eq!(rocm.name(), "rocm-smi");
    assert_eq!(rocm.memory(), GpuMemory::new(8_388_608, 2_097_152));
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn hanging_failing_and_garbled_programs_are_reported() {
    let _lock = script_lock();
    let dir = temp("error");
    let smi = |body: &str| Smi {
        program: script(&dir, body),
        index: 0,
        kind: SmiKind::Nvidia,
    };
    let start = std::time::Instant::now();
    assert_eq!(smi("sleep 30").memory(), Err(ProbeError::Timeout));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(4),
        "Timeout must take effect"
    );
    assert_eq!(smi("exit 3").memory(), Err(ProbeError::Unavailable));
    assert_eq!(
        smi("printf 'garbage'").memory(),
        Err(ProbeError::InvalidOutput)
    );
    assert_eq!(
        smi("printf '9, 1'").memory(),
        GpuMemory::new(9 << 20, 1 << 20)
    );
    let missing = Smi {
        program: dir.join("does-not-exist"),
        index: 0,
        kind: SmiKind::Nvidia,
    };
    assert_eq!(missing.memory(), Err(ProbeError::Unavailable));
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn vendor_detection_and_none_fallback() {
    use eliezer_core::config::{GpuVendor, ProbeMethod};
    let root = temp("vendor");
    for (index, vendor) in [(0, "0x10de"), (1, "0x8086")] {
        let device = root.join(format!("card{index}/device"));
        std::fs::create_dir_all(&device).unwrap();
        std::fs::write(device.join("vendor"), vendor).unwrap();
    }
    assert_eq!(detect_vendor(&root, 0), GpuVendor::Nvidia);
    assert_eq!(detect_vendor(&root, 1), GpuVendor::None, "foreign vendor");
    assert_eq!(detect_vendor(&root, 7), GpuVendor::None, "card missing");
    // Method none: bookkeeping only, same byte meaning
    let none = sources(
        GpuVendor::None,
        ProbeMethod::None,
        0,
        5_000_000_000,
        root.clone(),
    );
    assert_eq!(none.active_name(), "none");
    assert_eq!(none.memory(), GpuMemory::new(5_000_000_000, 0));
    // Sysfs selected but files missing: fall back to bookkeeping
    let broken = sources(
        GpuVendor::Amd,
        ProbeMethod::Sysfs,
        0,
        5_000_000_000,
        root.clone(),
    );
    assert_eq!(broken.memory(), GpuMemory::new(5_000_000_000, 0));
    assert_eq!(broken.active_name(), "none");
    // Sysfs with files: measurement wins over bookkeeping
    let device = root.join("card0/device");
    std::fs::write(device.join("mem_info_vram_total"), "100").unwrap();
    std::fs::write(device.join("mem_info_vram_used"), "40").unwrap();
    let ok = sources(
        GpuVendor::Amd,
        ProbeMethod::Sysfs,
        0,
        5_000_000_000,
        root.clone(),
    );
    assert_eq!(ok.memory(), GpuMemory::new(100, 40));
    assert_eq!(ok.active_name(), "sysfs");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn nvml_without_driver_is_unavailable_and_falls_through() {
    let nvml = Nvml::new(0);
    assert_eq!(nvml.name(), "nvml");
    match nvml.memory() {
        Ok(memory) => assert!(memory.used_bytes <= memory.total_bytes),
        Err(error) => assert_eq!(error, ProbeError::Unavailable),
    }
    let expected = GpuMemory::new(8 * 1_048_576, 0).unwrap();
    let fallback = Fallback::new(vec![
        Box::new(Nvml::new(u32::MAX)),
        Box::new(Accounting {
            budget: expected.total_bytes,
            leases: AtomicU64::new(0),
        }),
    ]);
    assert_eq!(fallback.memory().unwrap(), expected);
    assert_eq!(fallback.active_name(), "none");
}

#[test]
#[ignore = "needs an NVIDIA GPU with the NVML driver library"]
fn real_nvml_values() {
    let nvml = Nvml::new(0);
    let memory = nvml.memory().unwrap();
    assert!(memory.total_bytes > 0);
    assert!(memory.used_bytes <= memory.total_bytes);
    assert_eq!(nvml.name(), "nvml");
}

#[test]
#[ignore = "needs an AMD GPU with amdgpu sysfs files"]
fn real_sysfs_amd() {
    let root = std::path::Path::new("/sys/class/drm");
    assert_eq!(detect_vendor(root, 0), eliezer_core::config::GpuVendor::Amd);
    let memory = Sysfs {
        root: root.into(),
        index: 0,
    }
    .memory()
    .unwrap();
    println!(
        "sysfs card0: total={} used={}",
        memory.total_bytes, memory.used_bytes
    );
    assert!(memory.total_bytes > 0);
    assert!(memory.used_bytes <= memory.total_bytes);
}

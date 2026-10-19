use eliezer_gpu::*;
use std::{os::unix::fs::PermissionsExt, sync::atomic::AtomicU64};
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

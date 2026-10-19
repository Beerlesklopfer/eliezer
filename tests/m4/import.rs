//! `eliezer --import`: planning, moving, copying, ownership and the GGUF header reader.
use eliezer_core::{config::Config, import::*};
use std::{
    fs,
    os::unix::fs::{MetadataExt, PermissionsExt, symlink},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "eliezer-import-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(fs::canonicalize(path).unwrap())
    }
    fn file(&self, relative: &str, content: &[u8]) -> PathBuf {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();
        path
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const PASSWD: &str = "root:x:0:0:root:/root:/bin/bash\n# comment\nbroken line\neliezer:x:122:140:Eliezer:/var/lib/eliezer:/usr/sbin/nologin\nbad:x:abc:1:x:/:/bin/sh\n";
const GROUP: &str = "root:x:0:\nrender:x:105:eliezer\nmodels:x:777:\nodd:x:notanumber:\n";

#[test]
fn account_lookup_and_owner_resolution() {
    assert_eq!(lookup_user(PASSWD, "eliezer"), Some((122, 140)));
    assert_eq!(lookup_user(PASSWD, "bad"), None, "malformed uid");
    assert_eq!(lookup_user(PASSWD, "nobody"), None);
    assert_eq!(lookup_group(GROUP, "render"), Some(105));
    assert_eq!(lookup_group(GROUP, "odd"), None);
    let owner = |user, group| resolve_owner(PASSWD, GROUP, user, group).unwrap();
    assert_eq!(owner(None, None), None);
    assert_eq!(
        owner(Some("eliezer"), None),
        Some(Owner {
            uid: Some(122),
            gid: Some(140)
        }),
        "a user alone gets its primary group"
    );
    assert_eq!(
        owner(Some("eliezer"), Some("models")),
        Some(Owner {
            uid: Some(122),
            gid: Some(777)
        }),
        "an explicit group wins"
    );
    assert_eq!(
        owner(None, Some("render")),
        Some(Owner {
            uid: None,
            gid: Some(105)
        }),
        "a group alone leaves the owner alone"
    );
    assert!(
        resolve_owner(PASSWD, GROUP, Some("ghost"), None)
            .unwrap_err()
            .contains("ghost")
    );
    assert!(
        resolve_owner(PASSWD, GROUP, None, Some("ghosts"))
            .unwrap_err()
            .contains("ghosts")
    );
}

#[test]
fn owner_is_resolved_on_the_running_system() {
    // root exists everywhere, with uid 0 and gid 0
    assert_eq!(
        resolve_owner_system(Some("root"), Some("root")).unwrap(),
        Some(Owner {
            uid: Some(0),
            gid: Some(0)
        })
    );
    assert_eq!(
        resolve_owner_system(Some("root"), None).unwrap(),
        Some(Owner {
            uid: Some(0),
            gid: Some(0)
        }),
        "primary group from the passwd entry"
    );
    assert_eq!(resolve_owner_system(None, None).unwrap(), None);
    assert!(resolve_owner_system(Some("no-such-user-xyz"), None).is_err());
    assert!(resolve_owner_system(None, Some("no-such-group-xyz")).is_err());
    // a name that looks like an option is just an unknown account
    assert!(resolve_owner_system(Some("-s"), None).is_err());
}

#[test]
fn plan_finds_only_regular_gguf_files_and_keeps_the_structure() {
    let t = Temp::new();
    t.file("src/community/Qwen/model-Q4.gguf", b"aaaa");
    t.file("src/community/Qwen/mmproj-F16.gguf", b"bb");
    t.file("src/UPPER/Model.GGUF", b"cc");
    t.file("src/community/Qwen/readme.md", b"text");
    t.file("src/community/Qwen/weights.bin", b"bin");
    symlink(t.0.join("src/UPPER/Model.GGUF"), t.0.join("src/link.gguf")).unwrap();
    t.file("store/community/Qwen/mmproj-F16.gguf", b"already here");
    let planned = plan(&t.0.join("src"), &t.0.join("store")).unwrap();
    let targets: Vec<_> = planned
        .items
        .iter()
        .map(|i| i.target.strip_prefix(t.0.join("store")).unwrap().to_owned())
        .collect();
    assert_eq!(
        targets,
        vec![
            PathBuf::from("UPPER/Model.GGUF"),
            PathBuf::from("community/Qwen/model-Q4.gguf")
        ]
    );
    assert_eq!(planned.items.iter().map(|i| i.size).sum::<u64>(), 6);
    assert!(
        planned
            .skipped
            .iter()
            .any(|(p, s)| p.ends_with("mmproj-F16.gguf") && *s == Skip::Exists)
    );
    assert!(
        planned
            .skipped
            .iter()
            .any(|(p, s)| p.ends_with("link.gguf") && *s == Skip::NotRegular)
    );
}

#[test]
fn plan_refuses_unusable_sources() {
    let t = Temp::new();
    t.file("src/a.gguf", b"x");
    assert!(plan(&t.0.join("missing"), &t.0.join("store")).is_err());
    assert!(
        plan(&t.0.join("src/a.gguf"), &t.0.join("store")).is_err(),
        "a file is not a directory"
    );
    assert!(
        plan(&t.0.join("src"), &t.0.join("src/store")).is_err(),
        "store inside the source"
    );
    assert!(
        plan(&t.0.join("src"), &t.0).is_err(),
        "source inside the store"
    );
    // a store that does not exist yet is fine
    assert_eq!(
        plan(&t.0.join("src"), &t.0.join("new/deeper/store"))
            .unwrap()
            .items
            .len(),
        1
    );
}

#[test]
fn move_hands_the_file_over_and_never_overwrites() {
    let t = Temp::new();
    let source = t.file("src/a/b/model.gguf", b"model bytes");
    fs::set_permissions(&source, fs::Permissions::from_mode(0o600)).unwrap();
    fs::create_dir_all(t.0.join("store")).unwrap();
    let owner = owner_of(&t.0);
    let planned = plan(&t.0.join("src"), &t.0.join("store")).unwrap();
    let how = execute_item(&planned.items[0], Mode::Move, owner).unwrap();
    assert_eq!(how, How::Renamed);
    let target = t.0.join("store/a/b/model.gguf");
    assert_eq!(fs::read(&target).unwrap(), b"model bytes");
    assert!(!source.exists());
    let meta = fs::metadata(&target).unwrap();
    assert_eq!(meta.permissions().mode() & 0o777, 0o644);
    assert_eq!(Some(meta.uid()), owner.unwrap().uid);
    for dir in ["store/a", "store/a/b"] {
        assert_eq!(
            fs::metadata(t.0.join(dir)).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }
    // the target appears between planning and moving: it is reported, not overwritten
    let source = t.file("src2/model.gguf", b"new");
    let planned = plan(&t.0.join("src2"), &t.0.join("store2")).unwrap();
    t.file("store2/model.gguf", b"precious");
    assert!(execute_item(&planned.items[0], Mode::Move, owner).is_err());
    assert_eq!(
        fs::read(t.0.join("store2/model.gguf")).unwrap(),
        b"precious"
    );
    assert!(source.exists(), "the source stays when the move is refused");
}

#[test]
fn copy_keeps_the_source_and_leaves_no_partial_file() {
    let t = Temp::new();
    let source = t.file("src/m.gguf", &vec![7_u8; 100_000]);
    let planned = plan(&t.0.join("src"), &t.0.join("store")).unwrap();
    let how = execute_item(&planned.items[0], Mode::Copy, None).unwrap();
    assert_eq!(how, How::Copied);
    assert_eq!(
        fs::read(&source).unwrap(),
        fs::read(t.0.join("store/m.gguf")).unwrap()
    );
    assert!(!t.0.join("store/m.gguf.partial").exists());
}

// --- GGUF header ---------------------------------------------------------------------------

fn text(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}
fn kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    text(out, key);
    out.extend(4_u32.to_le_bytes());
    out.extend(value.to_le_bytes());
}
fn kv_text(out: &mut Vec<u8>, key: &str, value: &str) {
    text(out, key);
    out.extend(8_u32.to_le_bytes());
    text(out, value);
}
/// A small but realistic header: strings, numbers, a vocabulary array to skip, per-layer head counts.
fn gguf(arch: &str, extra: impl Fn(&mut Vec<u8>) -> u64) -> Vec<u8> {
    let mut body = Vec::new();
    kv_text(&mut body, "general.architecture", arch);
    kv_text(&mut body, "general.name", "Test Model 7B");
    kv_u32(&mut body, &format!("{arch}.context_length"), 32768);
    kv_u32(&mut body, &format!("{arch}.block_count"), 32);
    kv_u32(&mut body, &format!("{arch}.attention.head_count"), 32);
    kv_u32(&mut body, &format!("{arch}.embedding_length"), 4096);
    // tokenizer vocabulary: an array of strings that must be skipped
    text(&mut body, "tokenizer.ggml.tokens");
    body.extend(9_u32.to_le_bytes());
    body.extend(8_u32.to_le_bytes());
    body.extend(3_u64.to_le_bytes());
    for token in ["<s>", "hello", "wörld"] {
        text(&mut body, token);
    }
    let extra_count = extra(&mut body);
    let mut file = b"GGUF".to_vec();
    file.extend(3_u32.to_le_bytes());
    file.extend(0_u64.to_le_bytes());
    file.extend((7 + extra_count).to_le_bytes());
    file.extend(body);
    file
}

#[test]
fn gguf_header_is_read_without_the_tensor_data() {
    let t = Temp::new();
    let path = t.file(
        "m.gguf",
        &gguf("llama", |out| {
            kv_u32(out, "llama.attention.head_count_kv", 8);
            1
        }),
    );
    let info = read_gguf(&path).unwrap();
    assert_eq!(info.architecture, "llama");
    assert_eq!(info.name.as_deref(), Some("Test Model 7B"));
    assert_eq!(info.context_length, Some(32768));
    assert_eq!(info.layers, Some(32));
    assert_eq!(info.kv_heads, Some(8));
    assert_eq!(
        info.head_dim,
        Some(128),
        "embedding length / heads when no key length is given"
    );
    assert!(!info.hybrid);
}

#[test]
fn gguf_per_layer_arrays_and_hybrid_markers() {
    let t = Temp::new();
    let path = t.file(
        "m.gguf",
        &gguf("gemma4", |out| {
            // per-layer KV head counts: 4 values, the largest counts
            text(out, "gemma4.attention.head_count_kv");
            out.extend(9_u32.to_le_bytes());
            out.extend(4_u32.to_le_bytes());
            out.extend(4_u64.to_le_bytes());
            for n in [2_u32, 8, 2, 8] {
                out.extend(n.to_le_bytes());
            }
            kv_u32(out, "gemma4.attention.sliding_window", 1024);
            kv_u32(out, "gemma4.attention.key_length", 256);
            3
        }),
    );
    let info = read_gguf(&path).unwrap();
    assert_eq!(info.kv_heads, Some(8));
    assert_eq!(info.head_dim, Some(256));
    assert!(info.hybrid);
    let suggestion = suggest(
        Path::new("/var/lib/eliezer/models/x/Gemma-4_Q4.gguf"),
        16_800_000_000,
        Some(&info),
    );
    assert!(suggestion.contains("swa_layers/window"), "{suggestion}");
}

#[test]
fn broken_headers_are_errors_not_panics() {
    let t = Temp::new();
    assert!(matches!(
        read_gguf(&t.file("a.gguf", b"NOPE")),
        Err(GgufError::NotGguf)
    ));
    assert!(read_gguf(&t.file("b.gguf", b"")).is_err());
    let full = gguf("llama", |_| 0);
    for cut in [5, 9, 20, 60, full.len() - 3] {
        assert!(
            read_gguf(&t.file("c.gguf", &full[..cut])).is_err(),
            "cut at {cut}"
        );
    }
    let mut huge = b"GGUF".to_vec();
    huge.extend(3_u32.to_le_bytes());
    huge.extend(0_u64.to_le_bytes());
    huge.extend(u64::MAX.to_le_bytes());
    assert!(
        read_gguf(&t.file("d.gguf", &huge)).is_err(),
        "absurd entry count"
    );
    let mut long = b"GGUF".to_vec();
    long.extend(3_u32.to_le_bytes());
    long.extend(0_u64.to_le_bytes());
    long.extend(1_u64.to_le_bytes());
    long.extend(u64::MAX.to_le_bytes());
    assert!(
        read_gguf(&t.file("e.gguf", &long)).is_err(),
        "absurd string length"
    );
}

#[test]
fn keys_and_suggestions_are_valid_configuration() {
    assert_eq!(
        key_from(Path::new("/x/Qwen3-Coder-30B-A3B_Instruct Q4.gguf")),
        "qwen3-coder-30b-a3b-instruct-q4"
    );
    assert_eq!(key_from(Path::new("/x/---.gguf")), "");
    let t = Temp::new();
    let path = t.file(
        "m.gguf",
        &gguf("llama", |out| {
            kv_u32(out, "llama.attention.head_count_kv", 8);
            1
        }),
    );
    let info = read_gguf(&path).unwrap();
    let head = "[server]\nlisten = \"127.0.0.1:1234\"\nrun_user = \"eliezer\"\nrun_group = \"render\"\n[gpu]\n[backend.llamacpp]\nenabled = true\n";
    let block = suggest(
        Path::new("/var/lib/eliezer/models/c/Llama-7B_Q4.gguf"),
        4_400_000_000,
        Some(&info),
    );
    let config =
        Config::parse(&format!("{head}{block}")).unwrap_or_else(|e| panic!("{e}\n{block}"));
    assert_eq!(config.model[0].key, "llama-7b-q4");
    assert_eq!(config.server.run_group.as_deref(), Some("render"));
    // without a readable header the block still parses once it carries a memory figure
    let blind = suggest(
        Path::new("/var/lib/eliezer/models/unknown.gguf"),
        1_000_000_000,
        None,
    );
    assert!(blind.contains("could not be read"));
    // embedding models are recognized by architecture
    let embedding = GgufInfo {
        architecture: "nomic-bert".into(),
        ..Default::default()
    };
    assert!(
        suggest(Path::new("/m/nomic.gguf"), 1, Some(&embedding)).contains("kind = \"embedding\"")
    );
}

#[test]
fn run_user_and_run_group_are_validated() {
    let base = |extra: &str| format!("[server]\nlisten = \"127.0.0.1:1234\"\n{extra}\n[gpu]\n");
    assert!(Config::parse(&base("run_user = \"eliezer\"\nrun_group = \"video\"")).is_ok());
    // names of domain-joined machines have dots and capitals
    assert!(
        Config::parse(&base(
            "run_user = \"joerg.bernau\"\nrun_group = \"Domain_Users\""
        ))
        .is_ok()
    );
    let too_long = format!("run_user = \"{}\"", "a".repeat(65));
    for bad in [
        "run_user = \"a b\"",
        "run_user = \"\"",
        "run_group = \"../x\"",
        "run_user = \"root;rm\"",
        "run_group = \"1abc\"",
        "run_user = \".hidden\"",
        "run_user = \"a/b\"",
        "run_user = \"a$(id)\"",
        too_long.as_str(),
    ] {
        let error = Config::parse(&base(bad)).unwrap_err().to_string();
        assert!(error.contains("server.run_"), "{bad}: {error}");
    }
}

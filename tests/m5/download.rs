use eliezer_hub::{
    Detail, FakeHub, HubError, RepoFile, Token, child_environment,
    download::{FetchError, Outcome, fetch},
    store::{self, PinFile, Problem},
    valid_file, valid_revision,
};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "eliezer-dl-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(std::fs::canonicalize(path).unwrap())
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const REPO: &str = "org/model-GGUF";
const REV: &str = "0123456789abcdef0123456789abcdef01234567";

fn content(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 % 251) as u8).collect()
}
fn hash(data: &[u8]) -> String {
    store::sha256_hex(Sha256::digest(data))
}
fn file_of(name: &str, data: &[u8]) -> RepoFile {
    RepoFile {
        name: name.into(),
        bytes: data.len() as u64,
        sha256: Some(hash(data)),
    }
}
fn hub(name: &str, data: &[u8]) -> FakeHub {
    let mut hub = FakeHub {
        chunk_size: 1000,
        ..FakeHub::default()
    };
    hub.contents
        .insert((REPO.into(), name.into()), data.to_vec());
    hub
}
fn noop(_: u64) {}

#[test]
fn names_revisions_and_the_token_are_checked_and_hidden() {
    for good in [
        "m.gguf",
        "a/b.safetensors",
        "text_encoders/q-8b_int8.safetensors",
        "x+y.gguf",
    ] {
        assert!(valid_file(good), "{good}");
    }
    for bad in [
        "",
        "/abs",
        "../x",
        "a/../b",
        "a//b",
        ".hidden",
        "a/.b",
        "a/b/c/d.gguf",
        "a b",
        "a\\b",
        "a;b",
        "a\0b",
        "~/x",
        "x?y",
    ] {
        assert!(!valid_file(bad), "{bad:?}");
    }
    assert!(valid_revision(REV));
    for bad in ["main", "", "0123", &REV.to_uppercase(), &format!("{REV}0")] {
        assert!(!valid_revision(bad), "{bad}");
    }
    let token = Token::new("  hf_abcdefghijklmnopqrstuvwxyz  ").unwrap();
    assert_eq!(token.expose(), "hf_abcdefghijklmnopqrstuvwxyz");
    assert!(!format!("{token:?}").contains("hf_abc"));
    assert!(
        Token::new("").is_none() && Token::new("a b").is_none() && Token::new("a\nb").is_none()
    );
    // the token reaches a child process only as an environment variable
    assert_eq!(
        child_environment(Some(&token)),
        vec![("HF_TOKEN", "hf_abcdefghijklmnopqrstuvwxyz".to_owned())]
    );
    assert!(child_environment(None).is_empty());
}

#[test]
fn files_go_below_organisation_and_repository_and_never_out_of_the_store() {
    let t = Temp::new();
    let path = store::target_path(&t.0, REPO, "vae/v.safetensors").unwrap();
    assert_eq!(path, t.0.join(REPO).join("vae/v.safetensors"));
    assert!(path.parent().unwrap().is_dir(), "the directories are made");
    for bad in ["../x", "a/../../x", "/etc/passwd", ".hidden"] {
        assert_eq!(
            store::target_path(&t.0, REPO, bad),
            Err(store::StoreError::Invalid),
            "{bad}"
        );
    }
    for repo in ["../x/y", "org", "org/a/b", "o rg/x"] {
        assert_eq!(
            store::target_path(&t.0, repo, "m.gguf"),
            Err(store::StoreError::Invalid),
            "{repo}"
        );
    }
    // a symbolic link inside the store that leads out of it is not followed
    let outside = Temp::new();
    std::os::unix::fs::symlink(&outside.0, t.0.join("evil")).unwrap();
    assert_eq!(
        store::target_path(&t.0, "evil/repo", "m.gguf"),
        Err(store::StoreError::Invalid)
    );
    // and neither is a link as the target itself
    let dir = t.0.join("org2/model");
    std::fs::create_dir_all(&dir).unwrap();
    std::os::unix::fs::symlink(outside.0.join("target"), dir.join("m.gguf")).unwrap();
    assert_eq!(
        store::target_path(&t.0, "org2/model", "m.gguf"),
        Err(store::StoreError::Invalid)
    );
}

#[tokio::test]
async fn a_file_is_fetched_verified_and_renamed() {
    let t = Temp::new();
    let data = content(25_000);
    let hub = hub("m.gguf", &data);
    let file = file_of("m.gguf", &data);
    let dest = store::target_path(&t.0, REPO, "m.gguf").unwrap();
    let seen = Mutex::new(Vec::new());
    let outcome = fetch(&hub, REPO, REV, &file, &dest, None, &|n| {
        seen.lock().unwrap().push(n)
    })
    .await;
    assert_eq!(outcome, Ok(Outcome::Downloaded));
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(!dest.with_extension("gguf.part").exists());
    let seen = seen.into_inner().unwrap();
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "progress only grows");
    assert_eq!(*seen.last().unwrap(), 25_000);
    let opened = hub.opened.lock().unwrap().clone();
    assert_eq!(
        opened,
        vec![(REPO.into(), REV.into(), "m.gguf".into(), 0, false)]
    );
    // a second fetch finds the file and asks the hub for nothing
    let again = fetch(&hub, REPO, REV, &file, &dest, None, &noop).await;
    assert_eq!(again, Ok(Outcome::AlreadyPresent));
    assert_eq!(hub.opened.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_wrong_checksum_discards_the_file() {
    let t = Temp::new();
    let data = content(5_000);
    let hub = hub("m.gguf", &data);
    let mut file = file_of("m.gguf", &data);
    file.sha256 = Some(hash(b"something else"));
    let dest = store::target_path(&t.0, REPO, "m.gguf").unwrap();
    assert_eq!(
        fetch(&hub, REPO, REV, &file, &dest, None, &noop).await,
        Err(FetchError::Checksum)
    );
    assert!(!dest.exists(), "nothing under the final name");
    assert!(
        !dest.with_extension("gguf.part").exists(),
        "and no partial file is kept"
    );
}

#[tokio::test]
async fn sizes_that_disagree_with_the_listing_are_errors() {
    let t = Temp::new();
    let data = content(5_000);
    let hub = hub("m.gguf", &data);
    let dest = store::target_path(&t.0, REPO, "m.gguf").unwrap();
    let mut bigger = file_of("m.gguf", &data);
    bigger.bytes = 9_000;
    bigger.sha256 = None;
    assert_eq!(
        fetch(&hub, REPO, REV, &bigger, &dest, None, &noop).await,
        Err(FetchError::Size)
    );
    assert!(!dest.exists());
    let mut smaller = file_of("m.gguf", &data);
    smaller.bytes = 1_000;
    smaller.sha256 = None;
    let dest2 = store::target_path(&t.0, REPO, "n.gguf").unwrap();
    assert_eq!(
        fetch(&hub, REPO, REV, &smaller, &dest2, None, &noop).await,
        Err(FetchError::Size)
    );
    assert!(!dest2.exists());
}

#[tokio::test]
async fn an_interrupted_download_continues_where_it_stopped() {
    let t = Temp::new();
    let data = content(30_000);
    let file = file_of("m.gguf", &data);
    let dest = store::target_path(&t.0, REPO, "m.gguf").unwrap();
    let mut broken = hub("m.gguf", &data);
    broken.break_after = Some(12_000);
    assert_eq!(
        fetch(&broken, REPO, REV, &file, &dest, None, &noop).await,
        Err(FetchError::Hub(HubError::Unreachable))
    );
    let part = PathBuf::from(format!("{}.part", dest.display()));
    // The bytes of the last chunk before the break may or may not have been written (11000 or 12000 were seen);
    // what counts is that a part is left and that the next try continues exactly behind it.
    let kept = std::fs::metadata(&part).unwrap().len();
    assert!((1..=12_000).contains(&kept), "{kept} bytes kept");
    assert!(!dest.exists());
    let healthy = hub("m.gguf", &data);
    assert_eq!(
        fetch(&healthy, REPO, REV, &file, &dest, None, &noop).await,
        Ok(Outcome::Downloaded)
    );
    assert_eq!(
        healthy.opened.lock().unwrap()[0].3,
        kept,
        "asked from the first byte behind the part"
    );
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        data,
        "the checksum over both parts is right"
    );
    assert!(!part.exists());
}

#[tokio::test]
async fn a_hub_without_range_support_starts_over() {
    let t = Temp::new();
    let data = content(8_000);
    let file = file_of("m.gguf", &data);
    let dest = store::target_path(&t.0, REPO, "m.gguf").unwrap();
    std::fs::write(format!("{}.part", dest.display()), &data[..3_000]).unwrap();
    let mut hub = hub("m.gguf", &data);
    hub.ignore_range = true;
    assert_eq!(
        fetch(&hub, REPO, REV, &file, &dest, None, &noop).await,
        Ok(Outcome::Downloaded)
    );
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    // a partial file that is longer than the file is dropped
    let dest2 = store::target_path(&t.0, REPO, "n.gguf").unwrap();
    std::fs::write(format!("{}.part", dest2.display()), content(99_999)).unwrap();
    let mut hub2 = FakeHub::default();
    hub2.contents
        .insert((REPO.into(), "n.gguf".into()), data.clone());
    let file2 = file_of("n.gguf", &data);
    assert_eq!(
        fetch(&hub2, REPO, REV, &file2, &dest2, None, &noop).await,
        Ok(Outcome::Downloaded)
    );
    assert_eq!(hub2.opened.lock().unwrap()[0].3, 0);
}

#[tokio::test]
async fn a_gated_repository_needs_a_token_and_the_token_is_passed_on() {
    let t = Temp::new();
    let data = content(2_000);
    let mut hub = hub("m.gguf", &data);
    hub.details.push(Detail {
        repo: REPO.into(),
        sha: REV.into(),
        task: None,
        license: None,
        gated: true,
        downloads: 0,
        architecture: None,
        files: vec![],
    });
    let file = file_of("m.gguf", &data);
    let dest = store::target_path(&t.0, REPO, "m.gguf").unwrap();
    assert_eq!(
        fetch(&hub, REPO, REV, &file, &dest, None, &noop).await,
        Err(FetchError::Hub(HubError::Forbidden))
    );
    let token = Token::new("hf_secret0123456789").unwrap();
    assert_eq!(
        fetch(&hub, REPO, REV, &file, &dest, Some(&token), &noop).await,
        Ok(Outcome::Downloaded)
    );
    assert!(
        hub.opened.lock().unwrap().last().unwrap().4,
        "the token was given"
    );
    // the error text of a refusal does not contain anything secret
    assert!(
        !FetchError::Hub(HubError::Forbidden)
            .to_string()
            .contains("hf_")
    );
}

fn pin(name: &str, data: &[u8]) -> PinFile {
    PinFile {
        name: name.into(),
        bytes: data.len() as u64,
        sha256: Some(hash(data)),
        revision: REV.into(),
        downloaded_at: 1,
    }
}

#[test]
fn pins_are_recorded_merged_and_checked() {
    let t = Temp::new();
    let (a, b) = (content(100), content(200));
    let dir = store::repo_dir(&t.0, REPO).unwrap();
    std::fs::create_dir_all(dir.join("vae")).unwrap();
    std::fs::write(dir.join("a.gguf"), &a).unwrap();
    std::fs::write(dir.join("vae/b.safetensors"), &b).unwrap();
    store::record(&t.0, REPO, vec![pin("a.gguf", &a)]).unwrap();
    store::record(&t.0, REPO, vec![pin("vae/b.safetensors", &b)]).unwrap();
    // a second record of the same name replaces it
    store::record(&t.0, REPO, vec![pin("a.gguf", &a)]).unwrap();
    let pins = store::scan(&t.0);
    assert_eq!(pins.len(), 1);
    assert_eq!(pins[0].files.len(), 2);
    assert_eq!(store::pinned_bytes(&t.0), 300);
    let ok = |verify| {
        store::check(&t.0, &pins[0], verify)
            .into_iter()
            .map(|(f, p)| (f.name, p))
            .collect::<Vec<_>>()
    };
    assert!(ok(true).iter().all(|(_, p)| p.is_none()));
    // same size, other content: only the hash check sees it
    std::fs::write(dir.join("a.gguf"), vec![9_u8; 100]).unwrap();
    assert!(ok(false).iter().all(|(_, p)| p.is_none()));
    assert!(ok(true).contains(&("a.gguf".to_owned(), Some(Problem::Modified))));
    // other size
    std::fs::write(dir.join("a.gguf"), vec![9_u8; 50]).unwrap();
    assert!(ok(false).contains(&(
        "a.gguf".to_owned(),
        Some(Problem::SizeMismatch { found: 50 })
    )));
    // gone
    std::fs::remove_file(dir.join("vae/b.safetensors")).unwrap();
    assert!(ok(false).contains(&("vae/b.safetensors".to_owned(), Some(Problem::Missing))));
}

#[test]
fn a_broken_pin_file_and_foreign_directories_are_ignored() {
    let t = Temp::new();
    let dir = store::repo_dir(&t.0, REPO).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(store::PIN_FILE), "{not json").unwrap();
    std::fs::create_dir_all(t.0.join("lmstudio-community/other")).unwrap();
    std::fs::create_dir_all(t.0.join("hf")).unwrap();
    assert!(store::scan(&t.0).is_empty());
    assert_eq!(store::pinned_bytes(&t.0), 0);
    assert!(store::scan(Path::new("/nonexistent/store")).is_empty());
}

//! API keys and the Hugging Face token through the real binary: create, show, use, revoke.
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

const HF: &str = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789";

struct Env {
    dir: PathBuf,
    config: PathBuf,
    home: PathBuf,
    port: u16,
}
impl Env {
    fn new(name: &str) -> Self {
        let dir = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("eliezer-keys-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("etc")).unwrap();
        fs::create_dir_all(dir.join("models")).unwrap();
        fs::create_dir_all(dir.join("home")).unwrap();
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let config = dir.join("etc/config.toml");
        fs::write(
            &config,
            format!(
                "# my configuration\n[server]\nlisten = \"127.0.0.1:{port}\"\n\n[storage]\nmodels_dir = \"{d}/models\"\n\n[huggingface]\ncache_dir = \"{d}/models/hf\"\n\n[gpu]\n",
                d = dir.display()
            ),
        )
        .unwrap();
        Self {
            home: dir.join("home"),
            dir,
            config,
            port,
        }
    }
    fn eliezer(&self, args: &[&str], stdin: Option<&str>) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_eliezer"));
        command
            .env_remove("ELIEZER_CONFIG")
            .env_remove("ELIEZER_API_KEY")
            .env_remove("ELIEZER_API_KEY_FILE")
            .env_remove("SUDO_USER")
            .env_remove("XDG_CONFIG_HOME")
            .env_remove("XDG_CACHE_HOME")
            .env_remove("HF_HOME")
            .env_remove("HF_TOKEN_PATH")
            .env_remove("HF_TOKEN")
            .env("HOME", &self.home)
            .args(["--config", self.config.to_str().unwrap()])
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();
        if let Some(text) = stdin {
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(text.as_bytes())
                .unwrap();
        }
        drop(child.stdin.take());
        child.wait_with_output().unwrap()
    }
    fn client(&self, args: &[&str], key: Option<&str>) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_eliezer"));
        command
            .env_remove("ELIEZER_CONFIG")
            .env_remove("ELIEZER_API_KEY_FILE")
            .env_remove("SUDO_USER")
            .env_remove("XDG_CONFIG_HOME")
            .env("HOME", &self.home)
            .args(args)
            .args(["--host", &format!("http://127.0.0.1:{}", self.port)]);
        match key {
            Some(k) => command.env("ELIEZER_API_KEY", k),
            None => command.env_remove("ELIEZER_API_KEY"),
        };
        command.output().unwrap()
    }
    fn get(&self, path: &str, headers: &[(&str, &str)]) -> (u16, String, String) {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let extra: String = headers
            .iter()
            .map(|(k, v)| format!("{k}: {v}\r\n"))
            .collect();
        write!(
            stream,
            "GET {path} HTTP/1.1\r\nHost: x\r\n{extra}Connection: close\r\n\r\n"
        )
        .unwrap();
        let mut text = String::new();
        stream.read_to_string(&mut text).unwrap();
        let status = text.split(' ').nth(1).unwrap().parse().unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        (status, head.to_owned(), body.to_owned())
    }
    fn start(&self) -> std::process::Child {
        let child = Command::new(env!("CARGO_BIN_EXE_eliezer"))
            .env_remove("ELIEZER_CONFIG")
            .args(["--config", self.config.to_str().unwrap()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(8);
        while TcpStream::connect(("127.0.0.1", self.port)).is_err() {
            assert!(Instant::now() < deadline, "service did not start");
            std::thread::sleep(Duration::from_millis(50));
        }
        child
    }
    fn reload(&self, child: &std::process::Child) {
        Command::new("kill")
            .args(["-HUP", &child.id().to_string()])
            .status()
            .unwrap();
        std::thread::sleep(Duration::from_millis(400));
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}
fn key_in(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|l| l.starts_with("sk-") && l.len() == 67)
        .unwrap_or_else(|| panic!("no key in: {text}"))
        .to_owned()
}
fn mode(path: &Path) -> u32 {
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[test]
fn a_new_key_is_created_entered_in_the_configuration_and_works() {
    let env = Env::new("flow");
    let created = env.eliezer(&["--new-key", "--name", "multigpt"], None);
    assert!(created.status.success(), "{}", err(&created));
    let key = key_in(&out(&created));
    assert!(key[3..].bytes().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(
        out(&created).matches(&key).count(),
        1,
        "the key is shown exactly once"
    );
    assert!(!err(&created).contains(&key), "never on standard error");
    // the file: next to the configuration, mode 0600, with the name
    let file = env.dir.join("etc/keys");
    assert_eq!(mode(&file), 0o600);
    let content = fs::read_to_string(&file).unwrap();
    assert!(
        content.contains(&key) && content.contains("# name: multigpt"),
        "{content}"
    );
    // the configuration names it, keeps its comment, and has a backup
    let config = fs::read_to_string(&env.config).unwrap();
    assert!(
        config.contains("# my configuration")
            && config.contains(&format!("api_keys_file = \"{}\"", file.display())),
        "{config}"
    );
    assert!(env.dir.join("etc/config.toml.bak").exists());
    assert!(env.eliezer(&["--check-config"], None).status.success());
    assert!(
        !config.contains(&key),
        "the key is never written into the configuration"
    );

    // listing shows the name and a masked key, never the key
    let listed = env.eliezer(&["--list-keys"], None);
    assert!(listed.status.success());
    let text = out(&listed);
    assert!(
        text.contains("multigpt") && text.contains("sk-…") && !text.contains(&key),
        "{text}"
    );
    assert!(
        text.contains("Hugging Face token") && text.contains("not set"),
        "{text}"
    );

    // the running service: no key, a wrong key and the right key
    let mut service = env.start();
    let (status, head, body) = env.get("/v1/models", &[]);
    assert_eq!(status, 401, "{body}");
    assert!(
        body.contains("invalid_api_key") && body.contains("didn't provide an API key"),
        "{body}"
    );
    assert!(
        head.to_lowercase().contains("www-authenticate: bearer"),
        "{head}"
    );
    let (status, _, body) = env.get("/v1/models", &[("Authorization", "Bearer sk-wrong")]);
    assert_eq!(status, 401);
    assert!(
        body.contains("Incorrect API key provided") && !body.contains("sk-wrong"),
        "the sent key is not echoed: {body}"
    );
    let bearer = format!("Bearer {key}");
    assert_eq!(env.get("/v1/models", &[("Authorization", &bearer)]).0, 200);
    assert_eq!(
        env.get("/healthz", &[]).0,
        200,
        "the health check stays open"
    );
    // Anthropic clients use x-api-key; the OpenAI dialect does not accept it
    assert_eq!(
        env.get(
            "/v1/models",
            &[("x-api-key", &key), ("anthropic-version", "2023-06-01")]
        )
        .0,
        200
    );
    assert_eq!(env.get("/v1/models", &[("x-api-key", &key)]).0, 401);

    // the command line client: no key -> a clear message; env key; the user's own file
    let denied = env.client(&["ls"], None);
    assert_eq!(denied.status.code(), Some(1));
    assert!(err(&denied).contains("API key"), "{}", err(&denied));
    assert!(env.client(&["ls"], Some(&key)).status.success());
    let own = env.home.join(".config/eliezer");
    fs::create_dir_all(&own).unwrap();
    fs::write(own.join("api_key"), format!("{key}\n")).unwrap();
    fs::set_permissions(own.join("api_key"), fs::Permissions::from_mode(0o600)).unwrap();
    let with_file = env.client(&["ls"], None);
    assert!(
        with_file.status.success(),
        "~/.config/eliezer/api_key is picked up: {}",
        err(&with_file)
    );
    assert!(!out(&with_file).contains(&key) && !err(&with_file).contains(&key));

    // a second key, then revoking the first: the running service follows after a reload
    let second = env.eliezer(&["--new-key"], None);
    let key2 = key_in(&out(&second));
    assert!(
        out(&second).contains("key-1"),
        "an unnamed key gets a free name: {}",
        out(&second)
    );
    assert_eq!(
        env.get(
            "/v1/models",
            &[("Authorization", &format!("Bearer {key2}"))]
        )
        .0,
        401,
        "not loaded yet"
    );
    env.reload(&service);
    assert_eq!(
        env.get(
            "/v1/models",
            &[("Authorization", &format!("Bearer {key2}"))]
        )
        .0,
        200
    );
    let revoked = env.eliezer(&["--revoke-key", "multigpt"], None);
    assert!(revoked.status.success(), "{}", err(&revoked));
    env.reload(&service);
    assert_eq!(
        env.get("/v1/models", &[("Authorization", &bearer)]).0,
        401,
        "the revoked key stops working"
    );
    assert_eq!(
        env.get(
            "/v1/models",
            &[("Authorization", &format!("Bearer {key2}"))]
        )
        .0,
        200
    );
    let missing = env.eliezer(&["--revoke-key", "multigpt"], None);
    assert_eq!(missing.status.code(), Some(1));
    assert!(
        err(&missing).contains("No key has this name"),
        "{}",
        err(&missing)
    );
    service.kill().unwrap();
    service.wait().unwrap();
}

#[test]
fn save_places_a_copy_for_the_client_and_keeps_an_existing_one() {
    let env = Env::new("save");
    let made = env.eliezer(&["--new-key", "--save"], None);
    assert!(made.status.success(), "{}", err(&made));
    let key = key_in(&out(&made));
    let file = env.home.join(".config/eliezer/api_key");
    assert_eq!(fs::read_to_string(&file).unwrap().trim(), key);
    assert_eq!(mode(&file), 0o600);
    assert_eq!(mode(&env.home.join(".config/eliezer")), 0o700);
    // a second key does not replace the user's file
    let again = env.eliezer(&["--new-key", "--save"], None);
    assert!(again.status.success());
    assert_eq!(fs::read_to_string(&file).unwrap().trim(), key);
    assert!(
        err(&again).contains("exists and is not changed"),
        "{}",
        err(&again)
    );
}

#[test]
fn the_hugging_face_token_comes_from_standard_input_and_is_never_shown() {
    let env = Env::new("hf");
    // a token that does not look like one is refused and nothing is written
    let bad = env.eliezer(&["--set-hf-token"], Some("not-a-token\n"));
    assert_eq!(bad.status.code(), Some(1));
    assert!(
        err(&bad).contains("does not look like a Hugging Face token"),
        "{}",
        err(&bad)
    );
    assert!(!env.dir.join("etc/hf-token").exists());
    assert!(
        !fs::read_to_string(&env.config)
            .unwrap()
            .contains("token_file")
    );

    let stored = env.eliezer(&["--set-hf-token"], Some(&format!("{HF}\n")));
    assert!(stored.status.success(), "{}", err(&stored));
    for text in [out(&stored), err(&stored)] {
        assert!(
            !text.contains(HF),
            "the token must not appear in the output: {text}"
        );
    }
    assert!(out(&stored).contains("hf_…6789"), "{}", out(&stored));
    let file = env.dir.join("etc/hf-token");
    assert_eq!(mode(&file), 0o600);
    assert_eq!(fs::read_to_string(&file).unwrap().trim(), HF);
    let config = fs::read_to_string(&env.config).unwrap();
    assert!(
        config.contains(&format!("token_file = \"{}\"", file.display())),
        "{config}"
    );
    assert!(!config.contains(HF));
    assert!(env.eliezer(&["--check-config"], None).status.success());
    let listed = out(&env.eliezer(&["--list-keys"], None));
    assert!(
        listed.contains("hf_…6789") && !listed.contains(HF),
        "{listed}"
    );
    // a new token replaces the file but not the configuration entry again
    let other = "hf_ZyXwVuTsRqPoNmLkJiHgFeDcBa9876543210";
    assert!(
        env.eliezer(&["--set-hf-token"], Some(other))
            .status
            .success()
    );
    assert_eq!(fs::read_to_string(&file).unwrap().trim(), other);
    assert_eq!(
        fs::read_to_string(&env.config)
            .unwrap()
            .matches("token_file")
            .count(),
        1
    );
}

#[test]
fn the_token_is_never_taken_from_the_command_line_and_bad_combinations_fail() {
    let env = Env::new("args");
    let with_arg = env.eliezer(&["--set-hf-token", HF], None);
    assert_eq!(with_arg.status.code(), Some(2), "{}", err(&with_arg));
    assert!(
        !err(&with_arg).contains(HF) && !out(&with_arg).contains(HF),
        "the argument is not echoed"
    );
    for args in [
        vec!["--name", "x"],
        vec!["--save"],
        vec!["--new-key", "--list-keys"],
        vec!["--new-key", "--import", "/tmp"],
        vec!["--revoke-key"],
        vec!["--new-key", "--name", "bad name"],
    ] {
        let o = env.eliezer(&args, None);
        assert!(!o.status.success(), "{args:?}");
    }
    assert!(
        !env.dir.join("etc/keys").exists(),
        "failed commands leave no key file"
    );
}

#[test]
fn a_key_file_that_others_can_read_is_reported() {
    let env = Env::new("mode");
    env.eliezer(&["--new-key"], None);
    let file = env.dir.join("etc/keys");
    fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).unwrap();
    let text = out(&env.eliezer(&["--list-keys"], None));
    assert!(
        text.contains("Warning") && text.contains("chmod 600"),
        "{text}"
    );
}

#[test]
fn the_token_is_found_in_the_default_files_in_a_fixed_order() {
    let env = Env::new("sources");
    let hf_default = env.home.join(".cache/huggingface/token");
    let own = env.home.join(".config/eliezer/hf_key");
    let other = "hf_ZyXwVuTsRqPoNmLkJiHgFeDcBa9876543210";
    let third = "hf_QqQqQqQqQqQqQqQqQqQqQqQqQqQqQqQq0001";
    let stored = || {
        fs::read_to_string(env.dir.join("etc/hf-token"))
            .unwrap()
            .trim()
            .to_owned()
    };

    // nothing anywhere: a clear message naming where it looked, nothing written
    let nothing = env.eliezer(&["--set-hf-token"], None);
    assert_eq!(nothing.status.code(), Some(1));
    let message = err(&nothing);
    assert!(
        message.contains("No Hugging Face token found")
            && message.contains(".cache/huggingface/token")
            && message.contains("hf_key"),
        "{message}"
    );
    assert!(!env.dir.join("etc/hf-token").exists());

    // the default file of the Hugging Face tools (`hf auth login`), here world-readable
    fs::create_dir_all(hf_default.parent().unwrap()).unwrap();
    fs::write(&hf_default, format!("{HF}\n")).unwrap();
    fs::set_permissions(&hf_default, fs::Permissions::from_mode(0o644)).unwrap();
    let used = env.eliezer(&["--set-hf-token"], None);
    assert!(used.status.success(), "{}", err(&used));
    assert_eq!(stored(), HF);
    assert!(
        out(&used).contains(".cache/huggingface/token"),
        "the source is named: {}",
        out(&used)
    );
    assert!(
        err(&used).contains("can be read by other users") && err(&used).contains("chmod 600"),
        "{}",
        err(&used)
    );
    for text in [out(&used), err(&used)] {
        assert!(!text.contains(HF));
    }

    // the user's own file wins over the Hugging Face default
    fs::create_dir_all(own.parent().unwrap()).unwrap();
    fs::write(&own, format!("{other}\n")).unwrap();
    assert!(env.eliezer(&["--set-hf-token"], None).status.success());
    assert_eq!(stored(), other);

    // something piped in wins over both
    assert!(
        env.eliezer(&["--set-hf-token"], Some(third))
            .status
            .success()
    );
    assert_eq!(stored(), third);

    // HF_HOME is honored for the person themselves
    fs::remove_file(&own).unwrap();
    let hf_home = env.dir.join("hfhome");
    fs::create_dir_all(&hf_home).unwrap();
    fs::write(hf_home.join("token"), format!("{other}\n")).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_eliezer"));
    let result = command
        .env_remove("ELIEZER_CONFIG")
        .env_remove("SUDO_USER")
        .env("HOME", &env.home)
        .env("HF_HOME", &hf_home)
        .args(["--config", env.config.to_str().unwrap(), "--set-hf-token"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", err(&result));
    assert_eq!(
        stored(),
        other,
        "HF_HOME/token is used instead of ~/.cache/huggingface/token"
    );
}

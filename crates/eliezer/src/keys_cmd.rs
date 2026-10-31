//! Secrets: `eliezer --new-key`, `--list-keys`, `--revoke-key NAME` and `--set-hf-token`.
//!
//! Two different secrets are handled here. The *API key* (a token in the style of OpenAI, `sk-...`)
//! is what clients such as MultiGPT send to Eliezer; it lives in `server.api_keys_file`. The
//! *Hugging Face token* is what Eliezer sends to Hugging Face when it downloads models; it lives in
//! `huggingface.token_file`. Both files are mode 0600 and owned by the service account, neither
//! secret is ever a command-line argument or a log line, and a key is shown once when it is made.
//!
//! The user's own copies are `~/.config/eliezer/api_key` (read by the command line client) and
//! `~/.config/eliezer/hf_key` (the source of `--set-hf-token`).

use crate::{FileSystem, import_cmd};
use eliezer_core::{
    config::Config,
    edit,
    import::{self, Owner},
    keyfile,
};
use std::{
    io::{IsTerminal, Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt, chown},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

/// Secret files that the configuration does not name yet go next to the configuration file
/// (`/etc/eliezer/keys` and `/etc/eliezer/hf-token` for the default `/etc/eliezer/config.toml`).
fn beside(config_path: &Path, name: &str) -> PathBuf {
    config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(name)
}

pub enum Action {
    New { name: Option<String>, save: bool },
    List,
    Revoke(String),
    SetHfToken,
}

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("{message}");
    ExitCode::FAILURE
}

/// The directory of the user's own Eliezer files: `$XDG_CONFIG_HOME/eliezer` or `~/.config/eliezer`.
pub fn user_dir(home: &Path) -> PathBuf {
    match std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        Some(xdg) if std::env::var_os("SUDO_USER").is_none() => PathBuf::from(xdg),
        _ => home.join(".config"),
    }
    .join("eliezer")
}

/// The person who runs the command: under `sudo` that is `SUDO_USER`, whose home and ids are
/// looked up; otherwise the current user's `$HOME`.
struct Invoker {
    home: PathBuf,
    uid: u32,
    gid: u32,
    /// The command runs as another account than the person at the keyboard (sudo).
    elevated: bool,
}
fn invoker() -> Option<Invoker> {
    if let Some(name) = std::env::var("SUDO_USER").ok().filter(|n| !n.is_empty()) {
        let account = import::account(&name)?;
        return Some(Invoker {
            home: account.home,
            uid: account.uid,
            gid: account.gid,
            elevated: true,
        });
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    let meta = std::fs::metadata(&home).ok()?;
    use std::os::unix::fs::MetadataExt;
    Some(Invoker {
        home,
        uid: meta.uid(),
        gid: meta.gid(),
        elevated: false,
    })
}

/// Writes a secret file atomically: a temporary file with mode 0600, handed to `owner`, then renamed.
fn write_secret(
    path: &Path,
    content: &str,
    owner: Option<(Option<u32>, Option<u32>)>,
) -> std::io::Result<()> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".new");
    let temporary = PathBuf::from(temporary);
    let _ = std::fs::remove_file(&temporary);
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        if let Some((uid, gid)) = owner {
            chown(&temporary, uid, gid)?;
        }
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn read_or_empty(path: &Path) -> Result<String, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(format!("{} cannot be read: {e}", path.display())),
    }
}

fn permission_hint(error: &std::io::Error) -> &'static str {
    if error.kind() == std::io::ErrorKind::PermissionDenied {
        "\nThese commands change files below /etc/eliezer and usually need root (sudo)."
    } else {
        ""
    }
}

/// A configuration text with `key = "path"` set in `section`, checked like a service start.
fn with_setting(
    config_text: &str,
    section: &str,
    key: &str,
    path: &Path,
) -> Result<String, String> {
    let text = edit::set_string(config_text, section, key, &path.display().to_string())
        .map_err(|e| e.to_string())?;
    let config = Config::parse(&text)
        .map_err(|e| format!("the edited configuration would be invalid ({e})"))?;
    config
        .validate_paths(&text, &FileSystem)
        .map_err(|e| format!("the edited configuration would be invalid ({e})"))?;
    Ok(text)
}

/// Where a secret goes: the configuration key that names its file and the default location.
struct Secret<'a> {
    section: &'a str,
    key: &'a str,
    default: PathBuf,
    /// The file the configuration names already, if any.
    named: Option<&'a Path>,
}

/// Stores a secret file and, when the configuration does not name it yet, enters its path.
fn store(
    config_path: &Path,
    config_text: &str,
    secret: &Secret<'_>,
    content: &str,
    owner: Option<Owner>,
) -> Result<PathBuf, String> {
    let path = secret
        .named
        .map_or_else(|| secret.default.clone(), Path::to_path_buf);
    // Check the configuration first, so nothing is written when it would end up invalid.
    let new_config = if secret.named.is_none() {
        Some(with_setting(
            config_text,
            secret.section,
            secret.key,
            &path,
        )?)
    } else {
        None
    };
    write_secret(&path, content, owner.map(|o| (o.uid, o.gid)))
        .map_err(|e| format!("{}: {e}{}", path.display(), permission_hint(&e)))?;
    if let Some(text) = new_config {
        import::write_config(config_path, &text)
            .map_err(|e| format!("The configuration could not be written ({e})"))?;
        println!(
            "Entered {}.{} = \"{}\" in {} (the previous file is kept as {}.bak).",
            secret.section,
            secret.key,
            path.display(),
            config_path.display(),
            config_path.display()
        );
    }
    Ok(path)
}

pub fn run(config_path: &Path, action: Action) -> ExitCode {
    let (config, text) = match import_cmd::load(config_path) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let owner = match import::resolve_owner_system(
        config.server.run_user.as_deref(),
        config.server.run_group.as_deref(),
    ) {
        Ok(owner) => owner,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    match action {
        Action::New { name, save } => new_key(config_path, &config, &text, owner, name, save),
        Action::List => list(config_path, &config, owner),
        Action::Revoke(name) => revoke(&config, owner, &name),
        Action::SetHfToken => set_hf_token(config_path, &config, &text, owner),
    }
}

fn new_key(
    config_path: &Path,
    config: &Config,
    config_text: &str,
    owner: Option<Owner>,
    name: Option<String>,
    save: bool,
) -> ExitCode {
    let keys_file = config.server.api_keys_file.as_deref();
    let path = keys_file.map_or_else(|| beside(config_path, "keys"), Path::to_path_buf);
    let existing = match read_or_empty(&path) {
        Ok(text) => text,
        Err(e) => return fail(e),
    };
    let entries = keyfile::parse(&existing);
    let name = name.unwrap_or_else(|| keyfile::free_name(&entries));
    let key = match keyfile::generate() {
        Ok(key) => key,
        Err(e) => return fail(format!("No random numbers available: {e}")),
    };
    let new_text = match keyfile::add(&existing, &name, &key) {
        Ok(text) => text,
        Err(e) => return fail(e),
    };
    // The user's own copy comes first in the checks, so nothing is written if it cannot be placed.
    let user_file = if save {
        match invoker() {
            Some(who) => {
                let file = user_dir(&who.home).join("api_key");
                if file.exists() {
                    eprintln!(
                        "Note: {} exists and is not changed; --save keeps an existing key.",
                        file.display()
                    );
                    None
                } else {
                    Some((file, who))
                }
            }
            None => return fail("--save needs a home directory ($HOME or SUDO_USER)"),
        }
    } else {
        None
    };
    let stored = store(
        config_path,
        config_text,
        &Secret {
            section: "server",
            key: "api_keys_file",
            default: beside(config_path, "keys"),
            named: keys_file,
        },
        &new_text,
        owner,
    );
    let stored = match stored {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    println!(
        "Created the API key \"{name}\" ({}) in {}.",
        keyfile::mask(&key),
        stored.display()
    );
    if let Some((file, who)) = user_file {
        let made = file
            .parent()
            .map_or(Ok(()), |dir| {
                std::fs::create_dir_all(dir)?;
                std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
                if who.elevated {
                    chown(dir, Some(who.uid), Some(who.gid))?;
                }
                Ok::<(), std::io::Error>(())
            })
            .and_then(|()| {
                write_secret(
                    &file,
                    &format!("{key}\n"),
                    who.elevated.then_some((Some(who.uid), Some(who.gid))),
                )
            });
        match made {
            Ok(()) => println!(
                "Saved a copy for the command line client in {}.",
                file.display()
            ),
            Err(e) => eprintln!("The copy in {} could not be saved: {e}", file.display()),
        }
    }
    println!("\nApply it to the running service with: systemctl reload eliezer");
    println!(
        "\nThe key, shown only now. Keep it safe; enter it as the API key (OpenAI token) in the client, for example MultiGPT:\n\n{key}\n"
    );
    ExitCode::SUCCESS
}

fn mode_warning(path: &Path, owner: Option<Owner>) -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).ok()?;
    if meta.permissions().mode() & 0o077 != 0 {
        return Some(format!(
            "{} is readable by others (mode {:o}); the service refuses it. Run: chmod 600 {}",
            path.display(),
            meta.permissions().mode() & 0o777,
            path.display()
        ));
    }
    match owner.and_then(|o| o.uid) {
        Some(uid) if uid != meta.uid() => Some(format!(
            "{} does not belong to the service account (uid {uid}), so the service cannot read it",
            path.display()
        )),
        _ => None,
    }
}

fn list(config_path: &Path, config: &Config, owner: Option<Owner>) -> ExitCode {
    let path = config
        .server
        .api_keys_file
        .clone()
        .unwrap_or_else(|| beside(config_path, "keys"));
    println!("API keys ({})", path.display());
    if config.server.api_keys_file.is_none() {
        println!(
            "  The configuration names no api_keys_file: the API is open without authentication."
        );
    }
    match read_or_empty(&path) {
        Ok(text) => {
            let entries = keyfile::parse(&text);
            if entries.is_empty() {
                println!("  (no key)");
            }
            for entry in &entries {
                println!(
                    "  {:<20} {}",
                    entry.name.as_deref().unwrap_or("(unnamed)"),
                    keyfile::mask(&entry.key)
                );
            }
            if let Some(warning) = mode_warning(&path, owner) {
                println!("  Warning: {warning}");
            }
        }
        Err(e) => println!("  {e}"),
    }
    println!("\nHugging Face token");
    match &config.huggingface.token_file {
        None => println!("  not set (eliezer --set-hf-token)"),
        Some(file) => match read_or_empty(file) {
            Ok(text) if text.trim().is_empty() => {
                println!("  {} is empty or missing", file.display())
            }
            Ok(text) => {
                println!("  {} ({})", file.display(), keyfile::mask(text.trim()));
                if let Some(warning) = mode_warning(file, owner) {
                    println!("  Warning: {warning}");
                }
            }
            Err(e) => println!("  {e}"),
        },
    }
    ExitCode::SUCCESS
}

fn revoke(config: &Config, owner: Option<Owner>, name: &str) -> ExitCode {
    let Some(path) = config.server.api_keys_file.clone() else {
        return fail("The configuration names no api_keys_file; there is nothing to revoke.");
    };
    let text = match read_or_empty(&path) {
        Ok(text) => text,
        Err(e) => return fail(e),
    };
    let new_text = match keyfile::remove(&text, name) {
        Ok(text) => text,
        Err(e) => return fail(format!("{e}: '{name}'. See `eliezer --list-keys`.")),
    };
    if let Err(e) = write_secret(&path, &new_text, owner.map(|o| (o.uid, o.gid))) {
        return fail(format!("{}: {e}{}", path.display(), permission_hint(&e)));
    }
    println!("Revoked the API key \"{name}\".");
    if keyfile::parse(&new_text).is_empty() {
        println!(
            "Warning: no key is left. The service refuses to start with an empty key file \
             (it fails closed). Create a key with `eliezer --new-key`."
        );
    }
    println!("Apply it to the running service with: systemctl reload eliezer");
    ExitCode::SUCCESS
}

/// A Hugging Face token looks like `hf_` and letters and digits.
fn plausible_hf_token(token: &str) -> bool {
    token.starts_with("hf_")
        && (20..=200).contains(&token.len())
        && token
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
}

/// Reads one secret line without echoing it on a terminal.
fn read_hidden(prompt: &str) -> std::io::Result<String> {
    let terminal = std::io::stdin().is_terminal();
    eprint!("{prompt}");
    let _ = std::io::stderr().flush();
    if terminal {
        let _ = Command::new("stty").arg("-echo").status();
    }
    let mut line = String::new();
    let read = std::io::stdin().read_line(&mut line);
    if terminal {
        let _ = Command::new("stty").arg("echo").status();
        eprintln!();
    }
    read?;
    Ok(line.trim().to_owned())
}

/// The first non-empty line of a token file.
fn token_in(file: &Path) -> Option<String> {
    std::fs::read_to_string(file)
        .ok()?
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
}

/// Where `huggingface-cli login` / `hf auth login` keeps the token: `$HF_TOKEN_PATH`,
/// `$HF_HOME/token`, `$XDG_CACHE_HOME/huggingface/token`, otherwise `~/.cache/huggingface/token`.
/// Environment variables are honored only for the person themselves: `sudo` resets them, and root's
/// settings say nothing about the invoking user's files.
fn hf_default_file(home: &Path, elevated: bool) -> PathBuf {
    let env = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty() && !elevated)
            .map(PathBuf::from)
    };
    if let Some(path) = env("HF_TOKEN_PATH") {
        return path;
    }
    if let Some(hf_home) = env("HF_HOME") {
        return hf_home.join("token");
    }
    env("XDG_CACHE_HOME")
        .unwrap_or_else(|| home.join(".cache"))
        .join("huggingface/token")
}

/// Finds the token, in this order: standard input when something is piped in, the user's own
/// `~/.config/eliezer/hf_key`, the default file of the Hugging Face tools, a hidden prompt. The
/// token is never taken from the command line. Returns the token and where it came from.
fn find_hf_token() -> Result<(String, String), String> {
    if !std::io::stdin().is_terminal() {
        let mut piped = String::new();
        std::io::stdin()
            .read_to_string(&mut piped)
            .map_err(|e| format!("The token cannot be read from standard input: {e}"))?;
        if !piped.trim().is_empty() {
            return Ok((piped.trim().to_owned(), "standard input".to_owned()));
        }
    }
    let who = invoker();
    let files = who.iter().flat_map(|who| {
        [
            user_dir(&who.home).join("hf_key"),
            hf_default_file(&who.home, who.elevated),
        ]
    });
    let mut searched = Vec::new();
    for file in files {
        if let Some(token) = token_in(&file) {
            if let Ok(meta) = std::fs::metadata(&file)
                && meta.permissions().mode() & 0o077 != 0
            {
                eprintln!(
                    "Warning: {} can be read by other users (mode {:o}). Run: chmod 600 {}",
                    file.display(),
                    meta.permissions().mode() & 0o777,
                    file.display()
                );
            }
            return Ok((token, file.display().to_string()));
        }
        searched.push(file.display().to_string());
    }
    if std::io::stdin().is_terminal() {
        return read_hidden("Hugging Face token (input is hidden): ")
            .map(|token| (token, "the prompt".to_owned()))
            .map_err(|e| format!("The token cannot be read: {e}"));
    }
    Err(format!(
        "No Hugging Face token found. Looked in: standard input (empty), {}. \
         Log in with `hf auth login`, put the token in ~/.config/eliezer/hf_key, or pipe it in.",
        if searched.is_empty() {
            "no home directory".to_owned()
        } else {
            searched.join(", ")
        }
    ))
}

fn set_hf_token(
    config_path: &Path,
    config: &Config,
    config_text: &str,
    owner: Option<Owner>,
) -> ExitCode {
    let (token, source) = match find_hf_token() {
        Ok(found) => found,
        Err(message) => return fail(message),
    };
    if !plausible_hf_token(&token) {
        return fail(
            "That does not look like a Hugging Face token (hf_ followed by letters and digits). \
             Create one at https://huggingface.co/settings/tokens and give it on standard input, \
             in ~/.config/eliezer/hf_key, or at the prompt.",
        );
    }
    let stored = store(
        config_path,
        config_text,
        &Secret {
            section: "huggingface",
            key: "token_file",
            default: beside(config_path, "hf-token"),
            named: config.huggingface.token_file.as_deref(),
        },
        &format!("{token}\n"),
        owner,
    );
    match stored {
        Ok(path) => {
            println!(
                "Stored the Hugging Face token ({}) from {source} in {}.",
                keyfile::mask(&token),
                path.display()
            );
            println!("It is used for model downloads. Apply it with: systemctl reload eliezer");
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

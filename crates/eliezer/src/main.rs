mod cli;
mod import_cmd;
mod keys_cmd;
mod runtime;

use eliezer_core::config::{Config, PathResolutionError, PathResolver};
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};
use tokio::{
    net::TcpListener,
    signal::unix::{SignalKind, signal},
    sync::oneshot,
};

pub(crate) struct FileSystem;
impl PathResolver for FileSystem {
    fn resolve(&self, path: &Path) -> Result<PathBuf, PathResolutionError> {
        let mut existing = path;
        let mut tail = Vec::new();
        loop {
            match std::fs::symlink_metadata(existing) {
                Ok(_) => break,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tail.push(existing.file_name().ok_or(PathResolutionError)?.to_owned());
                    existing = existing.parent().ok_or(PathResolutionError)?;
                }
                Err(_) => return Err(PathResolutionError),
            }
        }
        let mut resolved = std::fs::canonicalize(existing).map_err(|_| PathResolutionError)?;
        if !tail.is_empty() && !resolved.is_dir() {
            return Err(PathResolutionError);
        }
        for part in tail.into_iter().rev() {
            resolved.push(part);
        }
        Ok(resolved)
    }
}

struct Arguments {
    config: PathBuf,
    check: bool,
    import: Option<PathBuf>,
    copy: bool,
    dry_run: bool,
    no_config: bool,
    suggest: Option<PathBuf>,
    keys: Option<keys_cmd::Action>,
}
fn arguments() -> Result<Option<Arguments>, &'static str> {
    let mut args = std::env::args_os().skip(1).peekable();
    let mut config = std::env::var_os("ELIEZER_CONFIG")
        .map_or_else(|| PathBuf::from("/etc/eliezer/config.toml"), PathBuf::from);
    let mut check = false;
    let mut import = None;
    let (mut copy, mut dry_run, mut no_config) = (false, false, false);
    let mut suggest = None;
    let (mut new_key, mut list_keys, mut set_hf, mut save) = (false, false, false, false);
    let (mut key_name, mut revoke) = (None, None);
    let mut explicit = false;
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("{}", cli::usage());
            return Ok(None);
        } else if arg == "--version" {
            println!("Eliezer {}", env!("CARGO_PKG_VERSION"));
            return Ok(None);
        } else if arg == "--config" {
            if explicit {
                return Err("Configuration path may only be given once");
            }
            config = PathBuf::from(args.next().ok_or("Path missing after --config")?);
            explicit = true;
        } else if arg == "--check-config" {
            if check {
                return Err("--check-config may only be given once");
            }
            check = true;
            if args
                .peek()
                .is_some_and(|value| !value.as_encoded_bytes().starts_with(b"-"))
            {
                if explicit {
                    return Err("Configuration path may only be given once");
                }
                config = PathBuf::from(args.next().ok_or("Configuration path missing")?);
                explicit = true;
            }
        } else if arg == "--import" {
            if import.is_some() {
                return Err("--import may only be given once");
            }
            import = Some(PathBuf::from(
                args.next().ok_or("Directory missing after --import")?,
            ));
        } else if arg == "--new-key" {
            new_key = true;
        } else if arg == "--name" {
            key_name = Some(
                args.next()
                    .ok_or("Name missing after --name")?
                    .into_string()
                    .map_err(|_| "The key name must be plain text")?,
            );
        } else if arg == "--save" {
            save = true;
        } else if arg == "--list-keys" {
            list_keys = true;
        } else if arg == "--revoke-key" {
            revoke = Some(
                args.next()
                    .ok_or("Name missing after --revoke-key")?
                    .into_string()
                    .map_err(|_| "The key name must be plain text")?,
            );
        } else if arg == "--set-hf-token" {
            set_hf = true;
        } else if arg == "--suggest" {
            suggest = Some(PathBuf::from(
                args.next().ok_or("File missing after --suggest")?,
            ));
        } else if arg == "--copy" {
            copy = true;
        } else if arg == "--dry-run" {
            dry_run = true;
        } else if arg == "--no-config" {
            no_config = true;
        } else {
            return Err("Unknown argument; see --help");
        }
    }
    let chosen = [new_key, list_keys, set_hf, revoke.is_some()]
        .iter()
        .filter(|c| **c)
        .count();
    if chosen > 1 {
        return Err("Use only one of --new-key, --list-keys, --revoke-key and --set-hf-token");
    }
    if (key_name.is_some() || save) && !new_key {
        return Err("--name and --save only work together with --new-key");
    }
    if chosen == 1 && (import.is_some() || check || suggest.is_some()) {
        return Err(
            "The key commands cannot be combined with --import, --check-config or --suggest",
        );
    }
    let keys = if new_key {
        Some(keys_cmd::Action::New {
            name: key_name.clone(),
            save,
        })
    } else if list_keys {
        Some(keys_cmd::Action::List)
    } else if let Some(name) = revoke {
        Some(keys_cmd::Action::Revoke(name))
    } else if set_hf {
        Some(keys_cmd::Action::SetHfToken)
    } else {
        None
    };
    if import.is_none() && (copy || dry_run || no_config) {
        return Err("--copy, --dry-run and --no-config only work together with --import");
    }
    if import.is_some() && check {
        return Err("--import and --check-config cannot be combined");
    }
    Ok(Some(Arguments {
        config,
        check,
        import,
        copy,
        dry_run,
        no_config,
        suggest,
        keys,
    }))
}

fn load(path: &Path) -> Result<Config, String> {
    let input = std::fs::read_to_string(path)
        .map_err(|_| "Configuration file cannot be read".to_owned())?;
    let config = Config::parse(&input).map_err(|e| e.to_string())?;
    config
        .validate_paths(&input, &FileSystem)
        .map_err(|e| e.to_string())?;
    Ok(config)
}

async fn serve(config_path: PathBuf, config: Config) -> Result<(), String> {
    let mut terminate =
        signal(SignalKind::terminate()).map_err(|_| "SIGTERM cannot be registered")?;
    let mut hangup = signal(SignalKind::hangup()).map_err(|_| "SIGHUP cannot be registered")?;
    for problem in config.file_problems() {
        tracing::warn!(event = "file_problem", "{problem}");
    }
    let config = std::sync::Arc::new(config);
    let runtime = runtime::build(config.clone(), runtime::keys(&config)?)?;
    let supervisor = runtime.supervisor.clone();
    let router = runtime.router.clone();
    let ticker = {
        let supervisor = supervisor.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            while supervisor.tick().await.is_ok() {
                interval.tick().await;
            }
        })
    };
    let listener = TcpListener::bind(config.server.listen)
        .await
        .map_err(|error| {
            let hint = match error.kind() {
                std::io::ErrorKind::AddrInUse => format!(
                    " Another program, or an Eliezer that is already running (or stopped with Ctrl-Z), uses the port; find it with `ss -ltnp | grep :{}`.",
                    config.server.listen.port()
                ),
                std::io::ErrorKind::PermissionDenied => {
                    " Ports below 1024 need extra privileges.".to_owned()
                }
                _ => String::new(),
            };
            format!(
                "HTTP address {} cannot be bound ({error}).{hint}",
                config.server.listen
            )
        })?;
    let (stop_tx, stop_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stop_rx.await;
            })
            .await
    });
    tracing::info!(event = "start", address = %config.server.listen, models = config.model.len(), "Service started");
    let mut task = task;
    loop {
        tokio::select! {
            result = &mut task => {
                return match result {
                    Ok(Ok(())) => Ok(()),
                    _ => Err("HTTP framework terminated unexpectedly".into()),
                };
            }
            _ = terminate.recv() => break,
            result = tokio::signal::ctrl_c() => {
                result.map_err(|_| "Interrupt signal cannot be received")?;
                break;
            }
            _ = hangup.recv() => {
                let reloaded = match load(&config_path) {
                    Ok(new) => match runtime::keys(&new) {
                        Ok(keys) => {
                            for problem in new.file_problems() {
                                tracing::warn!(event = "file_problem", "{problem}");
                            }
                            runtime
                                .reload(std::sync::Arc::new(new), keys)
                                .await
                                .map_err(str::to_owned)
                        }
                        Err(error) => Err(error.to_owned()),
                    },
                    Err(error) => Err(error),
                };
                match reloaded {
                    Ok(()) => tracing::info!(event = "config_reloaded", "New configuration applied"),
                    Err(error) => tracing::warn!(event = "config_rejected", %error, "New configuration was rejected; previous state is kept"),
                }
            }
        }
    }
    tracing::info!(event = "stop", "Service is shutting down in an orderly way");
    let _ = stop_tx.send(());
    ticker.abort();
    supervisor.shutdown().await;
    match tokio::time::timeout(Duration::from_secs(30), &mut task).await {
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(_) => Err("HTTP framework could not be shut down in an orderly way".into()),
        Err(_) => {
            task.abort();
            Err("Timeout exceeded while shutting down the service".into())
        }
    }
}

fn main() -> ExitCode {
    // Client commands (`eliezer ls`, `ps`, ...) run synchronously, before any async runtime exists.
    if let Some(code) = cli::dispatch() {
        return code;
    }
    match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime.block_on(service_main()),
        Err(_) => {
            eprintln!("Async runtime cannot be started");
            ExitCode::FAILURE
        }
    }
}

async fn service_main() -> ExitCode {
    let args = match arguments() {
        Ok(Some(args)) => args,
        Ok(None) => return ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    if let Some(action) = args.keys {
        // Like the import, these load the configuration themselves.
        return keys_cmd::run(&args.config, action);
    }
    if let Some(file) = &args.suggest {
        return import_cmd::suggest(file);
    }
    if let Some(source) = &args.import {
        // The import loads the configuration itself: a broken model list must not block it.
        return import_cmd::run(&import_cmd::Options {
            config_path: args.config.clone(),
            source: source.clone(),
            copy: args.copy,
            dry_run: args.dry_run,
            no_config: args.no_config,
        });
    }
    let config = match load(&args.config) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    if args.check {
        let problems = config.file_problems();
        for problem in &problems {
            eprintln!("Warning: {problem}");
        }
        if problems.is_empty() {
            println!("Configuration OK");
        } else {
            println!(
                "Configuration OK, but {} file problem(s) (see the warnings)",
                problems.len()
            );
        }
        return ExitCode::SUCCESS;
    }
    if tracing_subscriber::fmt()
        .with_target(false)
        .with_ansi(false)
        .try_init()
        .is_err()
    {
        eprintln!("Logging cannot be started");
        return ExitCode::FAILURE;
    }
    match serve(args.config, config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(event = "failed", "{error}");
            ExitCode::FAILURE
        }
    }
}

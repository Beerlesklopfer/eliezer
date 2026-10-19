mod cli;
mod runtime;

use eliezer_core::{
    config::{Config, PathResolutionError, PathResolver},
    import,
};
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

struct FileSystem;
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
}
fn arguments() -> Result<Option<Arguments>, &'static str> {
    let mut args = std::env::args_os().skip(1).peekable();
    let mut config = std::env::var_os("ELIEZER_CONFIG")
        .map_or_else(|| PathBuf::from("/etc/eliezer/config.toml"), PathBuf::from);
    let mut check = false;
    let mut import = None;
    let (mut copy, mut dry_run) = (false, false);
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
        } else if arg == "--copy" {
            copy = true;
        } else if arg == "--dry-run" {
            dry_run = true;
        } else {
            return Err("Unknown argument; see --help");
        }
    }
    if import.is_none() && (copy || dry_run) {
        return Err("--copy and --dry-run only work together with --import");
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
    }))
}

/// Resolves `server.run_user` and `server.run_group` on the running system.
fn import_owner(config: &Config) -> Result<Option<import::Owner>, String> {
    import::resolve_owner_system(
        config.server.run_user.as_deref(),
        config.server.run_group.as_deref(),
    )
}

/// `eliezer --import DIR`: moves (or copies) GGUF files into the model store.
fn run_import(config: &Config, source: &Path, copy: bool, dry_run: bool) -> ExitCode {
    let owner = match import_owner(config) {
        Ok(owner) => owner,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let plan = match import::plan(source, &config.storage.models_dir) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    for (path, reason) in &plan.skipped {
        let why = match reason {
            import::Skip::Exists => "target exists",
            import::Skip::NotRegular => "not a regular file",
        };
        println!("skipped  {} ({why})", path.display());
    }
    if plan.items.is_empty() {
        println!("Nothing to import.");
        return ExitCode::SUCCESS;
    }
    let mode = if copy {
        import::Mode::Copy
    } else {
        import::Mode::Move
    };
    let verb = if copy { "copy" } else { "move" };
    // The header is read before the move, while the file is still where it was.
    let infos: Vec<_> = plan
        .items
        .iter()
        .map(|item| import::read_gguf(&item.source).ok())
        .collect();
    if dry_run {
        for item in &plan.items {
            println!(
                "would {verb}  {} -> {} ({:.1} GB)",
                item.source.display(),
                item.target.display(),
                item.size as f64 / 1e9
            );
        }
        println!("Dry run: nothing was changed.");
        return ExitCode::SUCCESS;
    }
    if owner.is_none() {
        eprintln!(
            "Note: neither server.run_user nor server.run_group is set; ownership stays as it is."
        );
    }
    let (mut done, mut failed, mut bytes) = (0_usize, 0_usize, 0_u64);
    let mut suggestions = String::new();
    for (item, info) in plan.items.iter().zip(&infos) {
        match import::execute_item(item, mode, owner) {
            Ok(how) => {
                done += 1;
                bytes += item.size;
                let how = match how {
                    import::How::Renamed => "moved ",
                    import::How::Copied => "copied",
                };
                println!(
                    "{how}  {} -> {}",
                    item.source.display(),
                    item.target.display()
                );
                let is_projector = item
                    .target
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj"));
                if !is_projector {
                    suggestions += &import::suggest(&item.target, item.size, info.as_ref());
                    suggestions.push('\n');
                }
            }
            Err(error) => {
                failed += 1;
                eprintln!("failed   {error}");
                if matches!(&error, import::ImportError::Io { error, .. } if error.kind() == std::io::ErrorKind::PermissionDenied)
                    || matches!(&error, import::ImportError::Io { error, .. } if error.raw_os_error() == Some(1))
                {
                    eprintln!(
                        "         Hint: importing and handing over files usually needs root (sudo)."
                    );
                }
            }
        }
    }
    println!(
        "{done} file(s), {:.1} GB; {failed} failed.",
        bytes as f64 / 1e9
    );
    if !suggestions.is_empty() {
        println!(
            "\nSuggested configuration (check the values, then add them to the configuration file):\n\n{suggestions}"
        );
    }
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
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

async fn serve(config_path: PathBuf, config: Config) -> Result<(), &'static str> {
    let mut terminate =
        signal(SignalKind::terminate()).map_err(|_| "SIGTERM cannot be registered")?;
    let mut hangup = signal(SignalKind::hangup()).map_err(|_| "SIGHUP cannot be registered")?;
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
        .map_err(|_| "HTTP address cannot be bound")?;
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
                    _ => Err("HTTP framework terminated unexpectedly"),
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
                        Ok(keys) => runtime.reload(std::sync::Arc::new(new), keys).await.map_err(str::to_owned),
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
        Ok(_) => Err("HTTP framework could not be shut down in an orderly way"),
        Err(_) => {
            task.abort();
            Err("Timeout exceeded while shutting down the service")
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
    let config = match load(&args.config) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    if args.check {
        println!("Configuration OK");
        return ExitCode::SUCCESS;
    }
    if let Some(source) = &args.import {
        return run_import(&config, source, args.copy, args.dry_run);
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

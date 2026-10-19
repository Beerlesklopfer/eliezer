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
}
fn arguments() -> Result<Option<Arguments>, &'static str> {
    let mut args = std::env::args_os().skip(1).peekable();
    let mut config = std::env::var_os("ELIEZER_CONFIG")
        .map_or_else(|| PathBuf::from("/etc/eliezer/config.toml"), PathBuf::from);
    let mut check = false;
    let mut explicit = false;
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!(
                "Eliezer – Verwalter lokaler Modelle\n\nAufruf: eliezer [--config PFAD] [--check-config [PFAD]]\n\n--config PFAD       Konfiguration (sonst ELIEZER_CONFIG oder /etc/eliezer/config.toml)\n--check-config      Nur prüfen, danach beenden\n--version           Versionsnummer ausgeben"
            );
            return Ok(None);
        } else if arg == "--version" {
            println!("Eliezer {}", env!("CARGO_PKG_VERSION"));
            return Ok(None);
        } else if arg == "--config" {
            if explicit {
                return Err("Konfigurationspfad darf nur einmal angegeben werden");
            }
            config = PathBuf::from(args.next().ok_or("Nach --config fehlt der Pfad")?);
            explicit = true;
        } else if arg == "--check-config" {
            if check {
                return Err("--check-config darf nur einmal angegeben werden");
            }
            check = true;
            if args
                .peek()
                .is_some_and(|value| !value.as_encoded_bytes().starts_with(b"-"))
            {
                if explicit {
                    return Err("Konfigurationspfad darf nur einmal angegeben werden");
                }
                config = PathBuf::from(args.next().ok_or("Konfigurationspfad fehlt")?);
                explicit = true;
            }
        } else {
            return Err("Unbekanntes Argument; siehe --help");
        }
    }
    Ok(Some(Arguments { config, check }))
}

fn load(path: &Path) -> Result<Config, String> {
    let input = std::fs::read_to_string(path)
        .map_err(|_| "Konfigurationsdatei kann nicht gelesen werden".to_owned())?;
    let config = Config::parse(&input).map_err(|e| e.to_string())?;
    config
        .validate_paths(&input, &FileSystem)
        .map_err(|e| e.to_string())?;
    Ok(config)
}

async fn serve(config_path: PathBuf, config: Config) -> Result<(), &'static str> {
    let mut terminate =
        signal(SignalKind::terminate()).map_err(|_| "SIGTERM kann nicht registriert werden")?;
    let mut hangup =
        signal(SignalKind::hangup()).map_err(|_| "SIGHUP kann nicht registriert werden")?;
    let keys = match &config.server.api_keys_file {
        Some(path) => runtime::read_keys(path)?,
        None => {
            tracing::warn!(
                event = "no_auth",
                "Keine Schlüsseldatei konfiguriert; die Schnittstelle ist ohne Anmeldung offen"
            );
            Vec::new()
        }
    };
    let config = std::sync::Arc::new(config);
    let (router, supervisor) = runtime::build(config.clone(), keys)?;
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
        .map_err(|_| "HTTP-Adresse kann nicht gebunden werden")?;
    let (stop_tx, stop_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stop_rx.await;
            })
            .await
    });
    tracing::info!(event="start", address=%config.server.listen, models=config.model.len(), "Dienst gestartet; Laufzeiten folgen in M4");
    let mut task = task;
    loop {
        tokio::select! {
            result = &mut task => {
                return match result {
                    Ok(Ok(())) => Ok(()),
                    _ => Err("HTTP-Programmrahmen wurde unerwartet beendet"),
                };
            }
            _ = terminate.recv() => break,
            result = tokio::signal::ctrl_c() => {
                result.map_err(|_| "Abbruchsignal kann nicht empfangen werden")?;
                break;
            }
            _ = hangup.recv() => {
                match load(&config_path) {
                    Ok(_) => tracing::info!(event="config_check", "Neue Konfiguration ist gültig; Übernahme folgt mit dem Verwalter in M3"),
                    Err(error) => tracing::warn!(event="config_rejected", %error, "Neue Konfiguration wurde abgelehnt; bisheriger Zustand bleibt erhalten"),
                }
            }
        }
    }
    tracing::info!(event = "stop", "Dienst wird geordnet beendet");
    let _ = stop_tx.send(());
    ticker.abort();
    supervisor.shutdown().await;
    match tokio::time::timeout(Duration::from_secs(30), &mut task).await {
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(_) => Err("HTTP-Programmrahmen konnte nicht geordnet beendet werden"),
        Err(_) => {
            task.abort();
            Err("Zeitlimit beim Beenden des Dienstes überschritten")
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
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
        println!("Konfiguration in Ordnung");
        return ExitCode::SUCCESS;
    }
    if tracing_subscriber::fmt()
        .with_target(false)
        .with_ansi(false)
        .try_init()
        .is_err()
    {
        eprintln!("Protokollierung kann nicht gestartet werden");
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

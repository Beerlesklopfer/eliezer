//! `eliezer --import DIR`: moves GGUF files into the model store and enters the models into the
//! configuration.

use crate::FileSystem;
use eliezer_core::{
    config::Config,
    import::{self, Imported},
};
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

pub struct Options {
    pub config_path: PathBuf,
    pub source: PathBuf,
    pub copy: bool,
    pub dry_run: bool,
    /// Print the model blocks instead of writing them into the configuration file.
    pub no_config: bool,
}

/// Loads the configuration for the import. A broken model list must not block the import (the
/// import is what writes working entries), so the models are ignored when they do not validate.
pub(crate) fn load(path: &Path) -> Result<(Config, String), String> {
    let text = std::fs::read_to_string(path).map_err(|_| {
        format!(
            "Configuration file {} cannot be read (is it there, and are you root?)",
            path.display()
        )
    })?;
    match Config::parse(&text) {
        Ok(config) => {
            config
                .validate_paths(&text, &FileSystem)
                .map_err(|e| e.to_string())?;
            Ok((config, text))
        }
        Err(first) => {
            let (config, stripped) =
                Config::parse_without_models(&text).map_err(|e| e.to_string())?;
            config
                .validate_paths(&stripped, &FileSystem)
                .map_err(|e| e.to_string())?;
            eprintln!("Note: the model list of the configuration has a problem ({first}).");
            eprintln!("      The import continues; models are added only if the result is valid.");
            Ok((config, text))
        }
    }
}

/// Warns about existing entries that cannot work, so they do not surprise at the next start.
fn warn_about_mismatches(text: &str) {
    for found in import::mismatches(text) {
        eprintln!(
            "Warning: model '{}' ({}): {}.\n         Remove the entry; image models are served once the stable-diffusion.cpp runtime is connected.",
            found.key,
            found.file.display(),
            found.problem
        );
    }
}

fn owner(config: &Config) -> Result<Option<import::Owner>, String> {
    import::resolve_owner_system(
        config.server.run_user.as_deref(),
        config.server.run_group.as_deref(),
    )
}

/// Tries to enter the models into the configuration text and writes it when the result is valid.
/// Returns the keys that were (or, in a dry run, would be) added.
fn enter(options: &Options, text: &str, imported: &[Imported]) -> Vec<String> {
    let update = match import::update_config(text, imported) {
        Ok(update) => update,
        Err(reason) => {
            eprintln!("The configuration was not changed: {reason}.");
            return Vec::new();
        }
    };
    for (path, why) in &update.skipped {
        println!("not added  {} ({why})", path.display());
    }
    for note in &update.notes {
        println!("\n{note}");
    }
    if update.added.is_empty() {
        return Vec::new();
    }
    // Exactly what the service checks at start, including the paths: a file that passes only
    // the syntax check would fail later when the service starts.
    let valid = Config::parse(&update.text).and_then(|config| {
        config
            .validate_paths(&update.text, &FileSystem)
            .map(|()| config)
    });
    if let Err(error) = valid {
        eprintln!(
            "The configuration was not changed: adding the models would make it invalid ({error}).\n\
             Fix that problem in the file, then run the import again or add these blocks yourself."
        );
        return Vec::new();
    }
    if options.dry_run {
        println!(
            "would add to {}: {}{}",
            options.config_path.display(),
            update.added.join(", "),
            if update.backend_enabled {
                " (and enable [backend.llamacpp])"
            } else {
                ""
            }
        );
        return update.added;
    }
    match import::write_config(&options.config_path, &update.text) {
        Ok(()) => {
            println!(
                "Added to {}: {}{}",
                options.config_path.display(),
                update.added.join(", "),
                if update.backend_enabled {
                    " (and enabled [backend.llamacpp])"
                } else {
                    ""
                }
            );
            println!(
                "The previous file is kept as {}.bak. Apply the change with: systemctl reload eliezer",
                options.config_path.display()
            );
            update.added
        }
        Err(error) => {
            eprintln!(
                "The configuration file could not be written ({error}); run the import as root, or add the blocks yourself."
            );
            Vec::new()
        }
    }
}

pub fn run(options: &Options) -> ExitCode {
    let (config, text) = match load(&options.config_path) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    warn_about_mismatches(&text);
    let owner = match owner(&config) {
        Ok(owner) => owner,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let plan = match import::plan(&options.source, &config.storage.models_dir) {
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
    let mode = if options.copy {
        import::Mode::Copy
    } else {
        import::Mode::Move
    };
    let verb = if options.copy { "copy" } else { "move" };
    // The header is read before the move, while the file is still where it was.
    let infos: Vec<_> = plan
        .items
        .iter()
        .map(|item| import::read_gguf(&item.source).ok())
        .collect();
    let describe = |item: &import::Item, info: &Option<import::GgufInfo>| Imported {
        target: item.target.clone(),
        size: item.size,
        info: info.clone(),
    };

    if options.dry_run {
        for (item, info) in plan.items.iter().zip(&infos) {
            let kind = match info.as_ref().map(import::GgufInfo::kind) {
                Some(import::Kind::Image) => "image model",
                Some(import::Kind::Embedding) => "embedding model",
                _ => "language model",
            };
            println!(
                "would {verb}  {} -> {} ({:.1} GB, {kind})",
                item.source.display(),
                item.target.display(),
                item.size as f64 / 1e9
            );
        }
        if !options.no_config {
            let imported: Vec<_> = plan
                .items
                .iter()
                .zip(&infos)
                .map(|(item, info)| describe(item, info))
                .collect();
            enter(options, &text, &imported);
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
    let mut imported = Vec::new();
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
                imported.push(describe(item, info));
            }
            Err(error) => {
                failed += 1;
                eprintln!("failed   {error}");
                if matches!(&error, import::ImportError::Io { error, .. }
                    if error.kind() == std::io::ErrorKind::PermissionDenied
                        || error.raw_os_error() == Some(1))
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

    if options.no_config {
        let blocks: String = imported
            .iter()
            .filter(|i| {
                !i.target
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj"))
            })
            .map(|i| import::suggest(&i.target, i.size, i.info.as_ref()) + "\n")
            .collect();
        if !blocks.is_empty() {
            println!(
                "\nSuggested configuration (check the values, then add them to the configuration file):\n\n{blocks}"
            );
        }
    } else if !imported.is_empty() {
        println!();
        enter(options, &text, &imported);
    }
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// `eliezer --suggest FILE.gguf`: prints the `[[model]]` block for one file without changing
/// anything. Handy for models that are already in the configuration and need a better `arch`.
pub fn suggest(file: &Path) -> ExitCode {
    let size = match std::fs::metadata(file) {
        Ok(meta) if meta.is_file() => meta.len(),
        _ => {
            eprintln!("{} is not a readable file", file.display());
            return ExitCode::from(2);
        }
    };
    let target = std::fs::canonicalize(file).unwrap_or_else(|_| file.to_owned());
    let info = match import::read_gguf(&target) {
        Ok(info) => Some(info),
        Err(error) => {
            eprintln!("Note: the GGUF header cannot be read ({error}); the figures are estimates.");
            None
        }
    };
    print!("{}", import::suggest(&target, size, info.as_ref()));
    ExitCode::SUCCESS
}

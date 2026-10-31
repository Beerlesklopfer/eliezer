//! Command line client: `eliezer ls|ps|load|unload|status|run|show|help` talk to the running
//! service over its HTTP API. Everything here is synchronous and runs before the async runtime
//! of the service is started.
mod format;
mod http;
mod mp3;
mod speak_cmd;
mod transcribe;
mod transcribe_cmd;
mod voices_cmd;

use eliezer_core::config::Config;
use http::{Client, Endpoint, HttpError};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    io::{BufRead, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

const DEFAULT_HOST: &str = "http://127.0.0.1:1234";
const SHORT: Duration = Duration::from_secs(15);
const LONG: Duration = Duration::from_secs(3600);

/// The commands that have their own files (`speak_cmd.rs`, `voices_cmd.rs`).
const AUDIO_COMMANDS: &[&str] = &["speak", "voices"];

const COMMANDS: &[&str] = &[
    "ls",
    "ps",
    "load",
    "unload",
    "status",
    "run",
    "show",
    "search",
    "inspect",
    "download",
    "pins",
    "get",
    "rm",
    "transcribe",
    "help",
];

/// Complete help text of the program: service flags and client commands.
pub fn usage() -> String {
    r#"Eliezer – manager for local models

Usage: eliezer [--config PATH] [--check-config [PATH]]      start the service
       eliezer [--config PATH] --import DIR [--copy] [--dry-run] [--no-config]
       eliezer COMMAND [OPTIONS]                              talk to the running service

Service options:
  --config PATH       Configuration (otherwise ELIEZER_CONFIG or /etc/eliezer/config.toml)
  --check-config      Only check, then exit
  --import DIR        Move the .gguf files below DIR into storage.models_dir, hand them to
                      server.run_user / server.run_group and add the language and embedding
                      models to the configuration (usually needs root)
  --copy              With --import: copy instead of move
  --dry-run           With --import: only show what would happen
  --no-config         With --import: print the [[model]] blocks, do not edit the configuration
  --new-key [--name N] [--save]   Create an API key (sk-...), enter it in server.api_keys_file and
                      show it once; --save also stores it in ~/.config/eliezer/api_key
  --list-keys         Show the API keys (masked) and whether a Hugging Face token is set
  --revoke-key NAME   Remove one API key
  --set-hf-token      Store the Hugging Face token in huggingface.token_file. Source: standard
                      input, ~/.config/eliezer/hf_key, the default file of the Hugging Face tools
                      (~/.cache/huggingface/token) or a hidden prompt; never an argument
  --suggest FILE      Print the [[model]] block for one .gguf file (memory figures from its header)
  --version           Print the version number

Commands (they use the HTTP API of the running service):
  ls                              List the configured models (key, type, size, context, state)
  ps                              List the loaded models (VRAM, context, parallel, idle, requests)
  load MODEL [--context N] [--parallel N]
                                  Load a model and print the load time
  unload [MODEL | --all]          Unload one model, all models, or the only loaded one
  status                          Service, GPU memory and queue
  show MODEL                      Details of one configured model and the endpoints serving it
  run MODEL [PROMPT]              Chat with a model; streams the answer. Without PROMPT on a terminal
                                  an interactive chat starts (empty line or Ctrl-D ends it); with
                                  piped input the prompt is read from stdin
      --system TEXT               System prompt
      --temperature F             Sampling temperature
      --max-tokens N              Limit the answer length
      --reasoning LEVEL           none, low, medium or high
      --stats                     Print token statistics at the end (to stderr)
  speak | say [OPTIONS]           Read a text aloud (speech synthesis) and write the audio while it is
                                  made. The text comes from --text, --file or standard input; the audio
                                  goes to standard output, so `| sox -t wav - -t mp3 out.mp3` or
                                  `| play -t wav -` take it over; --format mp3 makes an mp3 here
      --text TEXT                 The text
      --file FILE                 Read the text from a file (- is standard input)
      --model KEY                 Speech synthesis model (default: the only tts model of the service)
      --voice NAME                A voice of `eliezer voices` (default: the model's setting; a model
                                  without a speaker of its own, such as Chatterbox, needs one)
      --language CODE             Language code such as de (default: the model's setting)
      --speed F                   Speed from 0.25 to 4
      --seed N                    Seed for a repeatable reading
      --format wav|pcm|mp3        wav (default, with an open length), raw 16 bit mono samples, or mp3
                                  (96 kbit/s, encoded here while the audio arrives)
      --out PATH|-                A file (written as PATH.part and renamed; --replace overwrites), a
                                  named pipe (written into as it is), or - for standard output
                                  (the default; refused on a terminal unless --out - is given)
      --replace                   Overwrite an existing file
  voices | --list-voices          List the voices of speech synthesis
  voices add NAME FILE [OPTIONS]  Add a voice from a recording (any audio file the service can read)
  voices add --name NAME --file AUDIO [--file TEXTFILE] [OPTIONS]
                                  The same with options; --file is given for the recording and, with a
                                  .txt file, for the words spoken in it
      --name NAME                 The name of the voice (letters, digits, - and _)
      --file FILE                 The recording, or a .txt file with the words spoken in it
      --transcript TEXT           The words spoken, as text (instead of a .txt file)
      --start TIME                Cut the part that starts here out of a long recording, before the
                                  upload; TIME is seconds, MM:SS or HH:MM:SS
      --length TIME               The length of that part (the service takes up to 120 s by default)
      --replace                   Overwrite a voice of that name
  voices rm NAME                  Remove a voice
  search [TEXT] [--task T] [--format F] [--license L] [--limit N] [--details]
                                  Search Hugging Face for model candidates (nothing is downloaded).
                                  T: llm, vision, embedding or image; F: gguf (default) or safetensors;
                                  --details also judges support and fit of each hit
  inspect ORG/REPO                Files, quantizations, backend support and fit of one repository
  get ORG/REPO [--quant Q4_K_M] [--with FILE,FILE]
                                  Fetch a model: the weight file in the best quantization that fits
                                  (or the one named), its vision projector and the extra files named
                                  with --with; like `lms get`
  download ORG/REPO FILE... [--revision COMMIT]
                                  Fetch files of an allowed repository into the model store and show the
                                  progress (the repository must be on [huggingface] allow)
  pins [--verify]                 Files fetched earlier and whether they still match their pin;
                                  --verify hashes them (slow for big files)
  rm ORG/REPO                     Delete a repository fetched earlier from the model store (refused
                                  while a configured model still uses its files)
  transcribe PATH... --out DIR [OPTIONS]
                                  Transcribe audio files (or the audio files of directories) one
                                  after the other and write DIR/<name>.txt, .srt, .vtt and .json.
                                  Existing outputs are kept, so an interrupted run can be repeated.
      --out DIR                   Where the outputs go (required); with --recursive the
                                  sub-directories below each PATH are kept
      --model KEY                 Speech model (default: the only stt model of the service)
      --language CODE             Language code such as de, or auto (default: the model's setting)
      --format LIST               txt,srt,vtt,json (default: all four)
      --prompt TEXT               Hint words such as names, passed to the recognizer
      --prompt-file FILE          Read the hint from a file
      --recursive                 Descend into sub-directories of directories
      --extensions LIST           File extensions searched in directories
                                  (default: mp3,m4a,aac,ogg,oga,opus,flac,wav); hidden files are ignored
      --replace                   Overwrite existing outputs
      --dry-run                   List what would be done, send nothing
      --limit N                   Stop after N transcribed files
      --timeout-secs N            Wait at most N seconds for one file (default: 21600)
                                  Progress goes to stderr, one line per file, and a summary at the end;
                                  exit 1 if any file failed, 3 if the service went away
  help                            This text

Options for all commands:
  --host URL          Service address (otherwise ELIEZER_HOST, the server.listen of the
                      configuration, or http://127.0.0.1:1234)
  --key-file PATH     File with the API key (otherwise ELIEZER_API_KEY_FILE, ELIEZER_API_KEY,
                      ~/.config/eliezer/api_key or
                      the server.api_keys_file of the configuration). The key is never taken
                      from the command line.
  --json              Print the JSON of the API instead of a table
  --config PATH       Configuration file to read the address and the key file from

Exit codes: 0 ok, 1 request failed or server error, 2 usage error, 3 service not reachable"#
    .to_owned()
}

enum Failure {
    Usage(String),
    Request(String),
    Unreachable(String),
}
type Res<T> = Result<T, Failure>;

/// The lines of the help text that describe one command: its line and the indented lines that follow.
fn command_usage(command: &str) -> String {
    let indent = |line: &str| line.len() - line.trim_start().len();
    let text = usage();
    let mut lines = text
        .lines()
        .skip_while(|line| indent(line) != 2 || line.split_whitespace().next() != Some(command));
    let mut out = String::new();
    if let Some(first) = lines.next() {
        out.push_str(first);
        out.push('\n');
        // The indented lines that follow, and further entries of the same command (`voices add ...`).
        for line in lines.take_while(|l| {
            indent(l) > 2 || (indent(l) == 2 && l.split_whitespace().next() == Some(command))
        }) {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn usage_err<T>(message: impl Into<String>) -> Res<T> {
    Err(Failure::Usage(message.into()))
}

/// Recognizes `eliezer [--config X|--host X|--key-file X|--json]... COMMAND ...`.
/// Returns `None` when the arguments belong to the service (flags only).
pub fn dispatch() -> Option<ExitCode> {
    let mut args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|a| a.into_string())
        .collect::<Result<_, _>>()
        .ok()?;
    let mut i = 0;
    while let Some(arg) = args.get(i) {
        match arg.as_str() {
            "--config" | "--host" | "--key-file" => i += 2,
            "--json" => i += 1,
            a if a.starts_with("--config=")
                || a.starts_with("--host=")
                || a.starts_with("--key-file=") =>
            {
                i += 1;
            }
            _ => break,
        }
    }
    // `--list-voices` in the place of the command word is an alias for `voices`, `say` one for `speak`.
    match args.get(i).map(String::as_str) {
        Some("--list-voices") => args[i] = "voices".to_owned(),
        Some("say") => args[i] = "speak".to_owned(),
        _ => {}
    }
    let word = args.get(i).filter(|a| !a.starts_with('-'))?;
    let code = match run(word, &args, i) {
        Ok(()) => 0,
        Err(Failure::Usage(m)) => {
            eprintln!("{m}\nSee `eliezer help`.");
            2
        }
        Err(Failure::Request(m)) => {
            eprintln!("Error: {m}");
            1
        }
        Err(Failure::Unreachable(m)) => {
            eprintln!("Error: {m}");
            3
        }
    };
    Some(ExitCode::from(code))
}

struct Spec {
    values: &'static [&'static str],
    flags: &'static [&'static str],
    positional: (usize, usize),
}
#[derive(Default)]
struct Parsed {
    positional: Vec<String>,
    values: HashMap<String, String>,
    /// The further values of an option in [`REPEATABLE_VALUES`], in the order given.
    more: HashMap<String, Vec<String>>,
    flags: HashSet<String>,
}
impl Parsed {
    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }
    fn value(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }
    /// All values of an option that may be given more than once, in the order given.
    fn all_values(&self, name: &str) -> Vec<&str> {
        let first = self.values.get(name).map(String::as_str);
        let more = self
            .more
            .get(name)
            .into_iter()
            .flatten()
            .map(String::as_str);
        first.into_iter().chain(more).collect()
    }
    fn number(&self, name: &str) -> Res<Option<u64>> {
        self.value(name)
            .map(|v| match v.parse::<u64>() {
                Ok(n) if n > 0 => Ok(n),
                _ => usage_err(format!("{name} needs a positive whole number")),
            })
            .transpose()
    }
}

const GLOBAL_VALUES: &[&str] = &["--host", "--key-file", "--config"];
/// Options that may be given more than once (`voices add --file a.mp3 --file a.txt`); a command that takes one
/// value only checks the count itself.
const REPEATABLE_VALUES: &[&str] = &["--file"];

fn parse(words: &[String], spec: &Spec) -> Res<Parsed> {
    let mut out = Parsed::default();
    let mut iter = words.iter();
    let mut literal = false;
    while let Some(word) = iter.next() {
        if literal || !word.starts_with('-') || word == "-" {
            out.positional.push(word.clone());
            continue;
        }
        if word == "--" {
            literal = true;
            continue;
        }
        if word == "-h" || word == "--help" {
            out.flags.insert("--help".to_owned());
            continue;
        }
        let word = if word == "-c" { "--context" } else { word };
        let (name, inline) = match word.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v.to_owned())),
            _ => (word, None),
        };
        if GLOBAL_VALUES.contains(&name) || spec.values.contains(&name) {
            let value = match inline {
                Some(v) => v,
                None => iter
                    .next()
                    .cloned()
                    .ok_or_else(|| Failure::Usage(format!("Value missing after {name}")))?,
            };
            if out.values.contains_key(name) {
                if !REPEATABLE_VALUES.contains(&name) {
                    return usage_err(format!("{name} may only be given once"));
                }
                out.more.entry(name.to_owned()).or_default().push(value);
            } else {
                out.values.insert(name.to_owned(), value);
            }
        } else if name == "--json" || spec.flags.contains(&name) {
            if inline.is_some() {
                return usage_err(format!("{name} takes no value"));
            }
            out.flags.insert(name.to_owned());
        } else if name == "--key" || name == "--api-key" {
            return usage_err(
                "The API key is never accepted on the command line; use --key-file, ELIEZER_API_KEY_FILE or ELIEZER_API_KEY",
            );
        } else {
            return usage_err(format!("Unknown option {name}"));
        }
    }
    // Asking for help needs none of the arguments of the command.
    if out.flags.contains("--help") {
        return Ok(out);
    }
    let (min, max) = spec.positional;
    if out.positional.len() < min {
        return usage_err("An argument is missing; see `eliezer help`");
    }
    if out.positional.len() > max {
        return usage_err(format!("Unexpected argument {}", out.positional[max]));
    }
    Ok(out)
}

fn spec_for(command: &str) -> Spec {
    match command {
        "load" => Spec {
            values: &["--context", "--parallel"],
            flags: &[],
            positional: (1, 1),
        },
        "unload" => Spec {
            values: &[],
            flags: &["--all"],
            positional: (0, 1),
        },
        "show" => Spec {
            values: &[],
            flags: &[],
            positional: (1, 1),
        },
        "search" => Spec {
            values: &["--task", "--format", "--license", "--limit"],
            flags: &["--details"],
            positional: (0, 8),
        },
        "inspect" => Spec {
            values: &[],
            flags: &[],
            positional: (1, 1),
        },
        "get" => Spec {
            values: &["--quant", "--with"],
            flags: &[],
            positional: (1, 1),
        },
        "download" => Spec {
            values: &["--revision"],
            flags: &[],
            positional: (2, 17),
        },
        "speak" => Spec {
            values: &[
                "--model",
                "--voice",
                "--language",
                "--speed",
                "--seed",
                "--format",
                "--text",
                "--file",
                "--out",
            ],
            flags: &["--replace"],
            positional: (0, 0),
        },
        "voices" => Spec {
            values: &["--transcript", "--name", "--file", "--start", "--length"],
            flags: &["--replace"],
            positional: (0, 3),
        },
        "transcribe" => Spec {
            values: &[
                "--out",
                "--model",
                "--language",
                "--format",
                "--prompt",
                "--prompt-file",
                "--extensions",
                "--limit",
                "--timeout-secs",
            ],
            flags: &["--recursive", "--replace", "--dry-run"],
            positional: (1, usize::MAX),
        },
        "rm" => Spec {
            values: &[],
            flags: &[],
            positional: (1, 1),
        },
        "pins" => Spec {
            values: &[],
            flags: &["--verify"],
            positional: (0, 0),
        },
        "run" => Spec {
            values: &["--system", "--temperature", "--max-tokens", "--reasoning"],
            flags: &["--stats"],
            positional: (1, 2),
        },
        "help" => Spec {
            values: &[],
            flags: &[],
            positional: (0, 1),
        },
        _ => Spec {
            values: &[],
            flags: &[],
            positional: (0, 0),
        },
    }
}

fn run(command: &str, args: &[String], at: usize) -> Res<()> {
    if !COMMANDS.contains(&command) && !AUDIO_COMMANDS.contains(&command) {
        return usage_err(format!("Unknown command '{command}'"));
    }
    // Global options before the command word and everything after it are parsed alike.
    let mut words: Vec<String> = args[..at].to_vec();
    words.extend(args[at + 1..].iter().cloned());
    let parsed = parse(&words, &spec_for(command)).map_err(|failure| match failure {
        // A missing argument is answered with the usage of the command, the template to fill in.
        Failure::Usage(m) if m.starts_with("An argument is missing") => Failure::Usage(format!(
            "An argument is missing.\n\nUsage:\n{}",
            command_usage(command)
        )),
        other => other,
    })?;
    if command == "help" || parsed.flag("--help") {
        // `eliezer COMMAND --help` and `eliezer help COMMAND` show that command only; the whole text otherwise.
        let topic = if command == "help" {
            parsed.positional.first().map(String::as_str)
        } else {
            Some(command)
        };
        match topic.map(command_usage).filter(|text| !text.is_empty()) {
            Some(text) => print!("{text}"),
            None if topic.is_some() && command == "help" => {
                return usage_err(format!("No help for '{}'", parsed.positional[0]));
            }
            None => println!("{}", usage()),
        }
        return Ok(());
    }
    let json = parsed.flag("--json");
    let client = connect(&parsed)?;
    match command {
        "ls" => cmd_ls(&client, json),
        "ps" => cmd_ps(&client, json),
        "speak" => speak_cmd::cmd_speak(&client, &parsed),
        "voices" => voices_cmd::cmd_voices(&client, json, &parsed),
        "status" => cmd_status(&client, json),
        "show" => cmd_show(&client, json, &parsed.positional[0]),
        "load" => cmd_load(&client, json, &parsed),
        "unload" => cmd_unload(&client, json, &parsed),
        "search" => cmd_search(&client, json, &parsed),
        "inspect" => cmd_inspect(&client, json, &parsed.positional[0]),
        "download" => cmd_download(&client, json, &parsed),
        "get" => cmd_get(&client, json, &parsed),
        "rm" => cmd_rm(&client, json, &parsed.positional[0]),
        "transcribe" => transcribe_cmd::cmd_transcribe(&client, json, &parsed),
        "pins" => cmd_pins(&client, json, parsed.flag("--verify")),
        _ => cmd_run(&client, json, &parsed),
    }
}

// ---- connection settings -------------------------------------------------------------------

fn read_config(explicit: Option<&str>) -> Res<Option<Config>> {
    let path = explicit.map(PathBuf::from).or_else(|| {
        Some(
            std::env::var_os("ELIEZER_CONFIG")
                .map_or_else(|| PathBuf::from("/etc/eliezer/config.toml"), PathBuf::from),
        )
    });
    let Some(path) = path else { return Ok(None) };
    let read = std::fs::read_to_string(&path).map_err(|_| "Configuration file cannot be read");
    let config = read.and_then(|text| Config::parse(&text).map_err(|_| "Configuration is invalid"));
    match (config, explicit.is_some()) {
        (Ok(config), _) => Ok(Some(config)),
        (Err(message), true) => usage_err(format!("{message}: {}", path.display())),
        (Err(_), false) => Ok(None),
    }
}

fn first_key(path: &Path) -> Result<String, &'static str> {
    let text = std::fs::read_to_string(path).map_err(|_| "Key file cannot be read")?;
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .ok_or("Key file contains no key")
}

fn connect(parsed: &Parsed) -> Res<Client> {
    let config = read_config(parsed.value("--config"))?;
    let host = match parsed.value("--host") {
        Some(h) => h.to_owned(),
        None => match std::env::var("ELIEZER_HOST").ok().filter(|h| !h.is_empty()) {
            Some(h) => h,
            None => config.as_ref().map_or_else(
                || DEFAULT_HOST.to_owned(),
                |c| {
                    let mut addr = c.server.listen;
                    if addr.ip().is_unspecified() {
                        addr.set_ip(if addr.is_ipv6() {
                            std::net::Ipv6Addr::LOCALHOST.into()
                        } else {
                            std::net::Ipv4Addr::LOCALHOST.into()
                        });
                    }
                    format!("http://{addr}")
                },
            ),
        },
    };
    let endpoint = Endpoint::parse(&host).map_err(Failure::Usage)?;
    let explicit_file = parsed.value("--key-file").map(PathBuf::from).or_else(|| {
        std::env::var_os("ELIEZER_API_KEY_FILE")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    });
    let key = if let Some(path) = explicit_file {
        Some(first_key(&path).map_err(|m| Failure::Usage(format!("{m}: {}", path.display())))?)
    } else if let Some(key) = std::env::var("ELIEZER_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
    {
        Some(key)
    } else {
        // Silent fallbacks: the user's own key file, then the file named by the configuration.
        let own = std::env::var_os("HOME")
            .map(|home| crate::keys_cmd::user_dir(Path::new(&home)).join("api_key"))
            .and_then(|file| first_key(&file).ok());
        own.or_else(|| {
            config
                .and_then(|c| c.server.api_keys_file)
                .and_then(|p| first_key(&p).ok())
        })
    };
    if key
        .as_deref()
        .is_some_and(|k| k.chars().any(char::is_control))
    {
        return usage_err("The API key contains control characters");
    }
    Ok(Client::new(endpoint, key))
}

// ---- requests ------------------------------------------------------------------------------

/// What systemd says about the local service (`active`, `inactive`, `failed`...), if it is
/// installed here. Only asked for loopback addresses.
fn systemd_state(client: &Client) -> Option<String> {
    let host = client.endpoint.host.as_str();
    if !(host == "localhost" || host == "::1" || host.starts_with("127.")) {
        return None;
    }
    let output = std::process::Command::new("systemctl")
        .args(["is-active", "eliezer"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let state = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    // "unknown" and an empty answer mean there is no such unit.
    (!state.is_empty() && state != "unknown").then_some(state)
}

fn transport(client: &Client, error: HttpError) -> Failure {
    match error {
        HttpError::Unreachable(reason) => Failure::Unreachable(match systemd_state(client) {
            Some(state) if state != "active" => format!(
                "The Eliezer service is not running (systemd: {state}); nothing answers at {} ({reason}).\n\
                 Start it with `sudo systemctl start eliezer` (and `sudo systemctl enable --now eliezer` to\n\
                 start it at boot), or run it in the foreground with `eliezer --config /etc/eliezer/config.toml`.\n\
                 If it fails, see `journalctl -u eliezer`.",
                client.endpoint.url()
            ),
            _ => format!(
                "The Eliezer service is not reachable at {} ({reason}).\n\
                 Start it with `systemctl start eliezer` or `eliezer --config /etc/eliezer/config.toml`,\n\
                 or name the address with --host or ELIEZER_HOST.",
                client.endpoint.url()
            ),
        }),
        HttpError::Protocol(m) => {
            Failure::Request(format!("Connection to the service failed: {m}"))
        }
        HttpError::NotEliezer(found) => Failure::Unreachable(format!(
            "The server at {} is not Eliezer ({}). Another program, for example\n\
             LM Studio, may be using this address. Give Eliezer its own port with `listen` in the\n\
             configuration, or name the right address with --host or ELIEZER_HOST.",
            client.endpoint.url(),
            found.map_or_else(
                || "it does not identify itself as Eliezer".to_owned(),
                |name| format!("it identifies as '{name}'")
            )
        )),
    }
}

/// Turns an error answer of the service into the message for the user.
fn api_error(client: &Client, status: u16, body: &str) -> Failure {
    let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let error = &parsed["error"];
    let message = error["message"]
        .as_str()
        .or_else(|| error.as_str())
        .map_or_else(|| format!("HTTP {status}"), str::to_owned);
    let kind = error["type"]
        .as_str()
        .map(|t| format!(", {t}"))
        .unwrap_or_default();
    let hint = if status == 401 {
        if client.has_key() {
            "\nThe service did not accept the API key. Check --key-file, ELIEZER_API_KEY_FILE or ELIEZER_API_KEY."
        } else {
            "\nThe service requires an API key. Name its file with --key-file or ELIEZER_API_KEY_FILE, or set ELIEZER_API_KEY."
        }
    } else {
        ""
    };
    Failure::Request(format!("{message} [HTTP {status}{kind}]{hint}"))
}

fn call(
    client: &Client,
    method: &str,
    path: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Res<Value> {
    let payload = body.map(Value::to_string);
    let response = client
        .send(method, path, payload.as_deref(), timeout)
        .map_err(|e| transport(client, e))?;
    let status = response.status;
    let text = response
        .text()
        .map_err(|e| Failure::Request(format!("Reading the answer failed: {e}")))?;
    if !(200..300).contains(&status) {
        return Err(api_error(client, status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|_| Failure::Request("The answer is not valid JSON".to_owned()))
}

fn print_json(value: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(value).unwrap_or_default()
    );
}

// ---- commands ------------------------------------------------------------------------------

fn cmd_ls(client: &Client, json: bool) -> Res<()> {
    let models = call(client, "GET", "/api/v1/models", None, SHORT)?;
    if json {
        print_json(&models);
        return Ok(());
    }
    let status = call(client, "GET", "/api/v1/status", None, SHORT).ok();
    print!("{}", format::ls_text(&models, status.as_ref()));
    Ok(())
}

fn cmd_ps(client: &Client, json: bool) -> Res<()> {
    let status = call(client, "GET", "/api/v1/status", None, SHORT)?;
    if json {
        print_json(&status);
        return Ok(());
    }
    let models = call(client, "GET", "/api/v1/models", None, SHORT)?;
    print!("{}", format::ps_text(&models, &status));
    Ok(())
}

fn cmd_status(client: &Client, json: bool) -> Res<()> {
    let health = client
        .send("GET", "/healthz", None, SHORT)
        .map_err(|e| transport(client, e))?;
    let ok = health.status == 200;
    let health = health.text().unwrap_or_default().trim().to_owned();
    if !ok {
        return Err(Failure::Request(format!(
            "The service answers, but its health check failed: {health}"
        )));
    }
    let status = call(client, "GET", "/api/v1/status", None, SHORT)?;
    if json {
        print_json(&json!({"url": client.endpoint.url(), "health": health, "status": status}));
    } else {
        print!(
            "{}",
            format::status_text(&client.endpoint.url(), &health, &status)
        );
    }
    Ok(())
}

fn cmd_show(client: &Client, json: bool, key: &str) -> Res<()> {
    let models = call(client, "GET", "/api/v1/models", None, SHORT)?;
    let Some(model) = models["models"]
        .as_array()
        .and_then(|l| l.iter().find(|m| m["key"] == key))
    else {
        return Err(Failure::Request(format!(
            "Model '{key}' is not configured. `eliezer ls` lists the configured models."
        )));
    };
    let status = call(client, "GET", "/api/v1/status", None, SHORT).ok();
    let entry = status
        .as_ref()
        .and_then(|s| s["models"].as_array())
        .and_then(|l| l.iter().find(|m| m["key"] == key));
    if json {
        print_json(&json!({"model": model, "status": entry}));
    } else {
        print!(
            "{}",
            format::show_text(model, entry, &client.endpoint.url())
        );
    }
    Ok(())
}

/// Percent-encodes everything but the unreserved characters, so user text cannot add parameters.
fn urlencode(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn cmd_search(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let mut query = vec![format!("q={}", urlencode(&parsed.positional.join(" ")))];
    for (flag, name) in [
        ("--task", "task"),
        ("--format", "format"),
        ("--license", "license"),
        ("--limit", "limit"),
    ] {
        if let Some(value) = parsed.value(flag) {
            query.push(format!("{name}={}", urlencode(value)));
        }
    }
    if parsed.flag("--details") {
        query.push("details=true".to_owned());
    }
    let path = format!("/api/v1/hub/search?{}", query.join("&"));
    let result = call(client, "GET", &path, None, SHORT)?;
    if json {
        print_json(&result);
    } else {
        print!("{}", format::search_text(&result));
    }
    Ok(())
}

fn cmd_inspect(client: &Client, json: bool, repo: &str) -> Res<()> {
    let path = format!(
        "/api/v1/hub/models/{}",
        repo.split('/').map(urlencode).collect::<Vec<_>>().join("/")
    );
    let result = call(client, "GET", &path, None, SHORT)?;
    if json {
        print_json(&result);
    } else {
        print!("{}", format::inspect_text(&result));
    }
    Ok(())
}

fn cmd_download(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    run_download(
        client,
        json,
        &parsed.positional[0],
        &parsed.positional[1..],
        parsed.value("--revision"),
    )
}

/// `eliezer get ORG/REPO`: looks at the repository, picks the files and fetches them.
fn cmd_get(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let repo = &parsed.positional[0];
    let path = format!(
        "/api/v1/hub/models/{}",
        repo.split('/').map(urlencode).collect::<Vec<_>>().join("/")
    );
    let detail = call(client, "GET", &path, None, SHORT)?;
    let with: Vec<String> = parsed
        .value("--with")
        .map(|v| v.split(',').map(str::to_owned).collect())
        .unwrap_or_default();
    let files =
        format::choose_files(&detail, parsed.value("--quant"), &with).map_err(Failure::Request)?;
    if !json {
        println!("Fetching {} file(s) of {repo}:", files.len());
        for name in &files {
            println!("  {name}");
        }
    }
    // The commit of the listing is pinned, so what was looked at is what is fetched.
    run_download(client, json, repo, &files, detail["sha"].as_str())
}

fn run_download(
    client: &Client,
    json: bool,
    repo: &str,
    files: &[String],
    revision: Option<&str>,
) -> Res<()> {
    let mut body = json!({"repo": repo, "files": files});
    if let Some(revision) = revision {
        body["revision"] = json!(revision);
    }
    let started = call(
        client,
        "POST",
        "/api/v1/models/download",
        Some(&body),
        SHORT,
    )?;
    let id = started["id"].as_str().unwrap_or_default().to_owned();
    if !json {
        println!(
            "Download {id}: {} files, {} from {repo} at commit {}",
            started["files"].as_array().map_or(0, Vec::len),
            format::bytes(started["total_bytes"].as_u64().unwrap_or(0)),
            started["revision"]
                .as_str()
                .unwrap_or("?")
                .get(..8)
                .unwrap_or("?")
        );
    }
    let mut last = String::new();
    // On a terminal the line is rewritten in place; in a pipe or log it changes in steps of 5 %.
    let live = std::io::IsTerminal::is_terminal(&std::io::stdout());
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let job = call(
            client,
            "GET",
            &format!("/api/v1/models/download/{id}"),
            None,
            SHORT,
        )?;
        let line = format::download_line(&job, live);
        if !json && line != last {
            if live {
                print!("\r\x1b[2K{line}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            } else {
                println!("{line}");
            }
            last = line;
        }
        match job["state"].as_str() {
            Some("done") => {
                if json {
                    print_json(&job);
                } else {
                    if live {
                        println!();
                    }
                    print!("{}", format::download_done(&job));
                }
                return Ok(());
            }
            Some("failed") => {
                if json {
                    print_json(&job);
                }
                return Err(Failure::Request(format!(
                    "The download failed: {}",
                    job["error"].as_str().unwrap_or("unknown reason")
                )));
            }
            _ => {}
        }
    }
}

fn cmd_rm(client: &Client, json: bool, repo: &str) -> Res<()> {
    if repo.split('/').count() != 2 {
        return Err(Failure::Usage(format!(
            "'{repo}' is not a repository: write ORG/REPO as `eliezer pins` lists it (a model key from \
             `eliezer ls` names a configured model, not the files in the store)"
        )));
    }
    let result = call(
        client,
        "DELETE",
        &format!("/api/v1/models/pins/{repo}"),
        None,
        SHORT,
    )?;
    if json {
        print_json(&result);
    } else {
        println!(
            "Deleted {repo}, freed {}.",
            format::bytes(result["freed_bytes"].as_u64().unwrap_or(0))
        );
    }
    Ok(())
}

fn cmd_pins(client: &Client, json: bool, verify: bool) -> Res<()> {
    let path = if verify {
        "/api/v1/models/pins?verify=true"
    } else {
        "/api/v1/models/pins"
    };
    let result = call(client, "GET", path, None, LONG)?;
    if json {
        print_json(&result);
    } else {
        print!("{}", format::pins_text(&result));
    }
    Ok(())
}

fn cmd_load(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let key = &parsed.positional[0];
    let mut body = json!({"model": key, "echo_load_config": true});
    if let Some(n) = parsed.number("--context")? {
        body["context_length"] = json!(n);
    }
    if let Some(n) = parsed.number("--parallel")? {
        body["parallel"] = json!(n);
    }
    if !json {
        eprintln!("Loading {key} ...");
    }
    let answer = call(client, "POST", "/api/v1/models/load", Some(&body), LONG)?;
    if json {
        print_json(&answer);
        return Ok(());
    }
    let seconds = answer["load_time_seconds"].as_f64().unwrap_or(f64::NAN);
    let config = &answer["load_config"];
    let mut details = Vec::new();
    if let Some(c) = config["context_length"].as_u64() {
        details.push(format!("context {c}"));
    }
    if let Some(p) = config["parallel"].as_u64() {
        details.push(format!("parallel {p}"));
    }
    let details = if details.is_empty() {
        String::new()
    } else {
        format!(" ({})", details.join(", "))
    };
    println!(
        "Model loaded successfully in {}: {}{details}",
        format::duration_secs(seconds),
        answer["instance_id"].as_str().unwrap_or(key)
    );
    Ok(())
}

fn cmd_unload(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let all = parsed.flag("--all");
    if all && !parsed.positional.is_empty() {
        return usage_err("Give either a model or --all, not both");
    }
    let targets: Vec<String> = if let Some(key) = parsed.positional.first() {
        vec![key.clone()]
    } else {
        let status = call(client, "GET", "/api/v1/status", None, SHORT)?;
        let loaded: Vec<String> = status["models"]
            .as_array()
            .map(|l| {
                l.iter()
                    .filter(|m| matches!(m["state"].as_str(), Some("loaded" | "loading")))
                    .filter_map(|m| m["key"].as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        if !all && loaded.len() > 1 {
            return usage_err(format!(
                "Several models are loaded ({}); name one or use --all",
                loaded.join(", ")
            ));
        }
        loaded
    };
    if targets.is_empty() {
        if json {
            print_json(&json!({"unloaded": []}));
        } else {
            println!("No models are currently loaded.");
        }
        return Ok(());
    }
    let (mut done, mut first_error) = (Vec::new(), None);
    for key in &targets {
        match call(
            client,
            "POST",
            "/api/v1/models/unload",
            Some(&json!({"instance_id": key})),
            LONG,
        ) {
            Ok(_) => {
                if !json {
                    println!("Model unloaded: {key}");
                }
                done.push(key.clone());
            }
            Err(Failure::Request(m)) if targets.len() > 1 => {
                eprintln!("Error: {key}: {m}");
                first_error.get_or_insert(m);
            }
            Err(other) => return Err(other),
        }
    }
    if json {
        print_json(&json!({"unloaded": done}));
    }
    match first_error {
        Some(m) => Err(Failure::Request(format!(
            "Not all models could be unloaded ({m})"
        ))),
        None => Ok(()),
    }
}

// ---- run (chat) ----------------------------------------------------------------------------

struct ChatOptions {
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    reasoning: Option<String>,
    stats: bool,
    json: bool,
}

/// How reasoning text is made visible: dim on a terminal, a prefix without colors, stderr when piped.
struct Styler {
    tty: bool,
    color: bool,
}
impl Styler {
    fn begin(&self) -> &'static str {
        match (self.tty, self.color) {
            (true, true) => "\x1b[2m",
            (true, false) => "[thinking] ",
            _ => "",
        }
    }
    fn end(&self) -> &'static str {
        match (self.tty, self.color) {
            (true, true) => "\x1b[0m\n\n",
            _ => "\n\n",
        }
    }
}

/// One streamed answer: returns the answer text (without reasoning).
fn stream_chat(client: &Client, body: &Value, options: &ChatOptions) -> Res<String> {
    let payload = body.to_string();
    let response = client
        .send("POST", "/v1/chat/completions", Some(&payload), LONG)
        .map_err(|e| transport(client, e))?;
    if response.status != 200 {
        let status = response.status;
        let text = response.text().unwrap_or_default();
        return Err(api_error(client, status, &text));
    }
    let stdout_tty = std::io::stdout().is_terminal();
    let styler = Styler {
        tty: stdout_tty,
        color: std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty()),
    };
    let (mut out, mut err) = (std::io::stdout(), std::io::stderr());
    let (mut content, mut thinking) = (String::new(), false);
    let (mut usage, mut stats) = (None, None);
    for line in response.lines() {
        let line =
            line.map_err(|e| Failure::Request(format!("The answer was interrupted: {e}")))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue; // comments (": ping"), event names and blank lines
        };
        if data == "[DONE]" {
            break;
        }
        if options.json {
            println!("{data}");
            continue;
        }
        let Ok(frame) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if frame.get("error").is_some() {
            let message = frame["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(Failure::Request(format!(
                "The service reported an error: {message}"
            )));
        }
        if !frame["usage"].is_null() {
            usage = Some(frame["usage"].clone());
        }
        if !frame["eliezer"].is_null() {
            stats = Some(frame["eliezer"].clone());
        }
        let delta = &frame["choices"][0]["delta"];
        if let Some(text) = delta["reasoning_content"]
            .as_str()
            .filter(|t| !t.is_empty())
        {
            if stdout_tty {
                if !thinking {
                    let _ = write!(out, "{}", styler.begin());
                }
                let _ = write!(out, "{text}");
                let _ = out.flush();
            } else {
                let _ = write!(err, "{text}");
                let _ = err.flush();
            }
            thinking = true;
        }
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            if thinking {
                let _ = write!(
                    if stdout_tty {
                        &mut out as &mut dyn Write
                    } else {
                        &mut err
                    },
                    "{}",
                    styler.end()
                );
                thinking = false;
            }
            let _ = write!(out, "{text}");
            let _ = out.flush();
            content.push_str(text);
        }
    }
    if options.json {
        return Ok(content);
    }
    if thinking {
        let _ = write!(
            if stdout_tty {
                &mut out as &mut dyn Write
            } else {
                &mut err
            },
            "{}",
            styler.end()
        );
    }
    if !content.ends_with('\n') {
        let _ = writeln!(out);
    }
    if options.stats {
        eprint!("{}", format::stats_text(usage.as_ref(), stats.as_ref()));
    }
    Ok(content)
}

fn cmd_run(client: &Client, json: bool, parsed: &Parsed) -> Res<()> {
    let model = &parsed.positional[0];
    let reasoning = parsed.value("--reasoning").map(str::to_owned);
    if let Some(level) = &reasoning
        && !["none", "low", "medium", "high"].contains(&level.as_str())
    {
        return usage_err("--reasoning must be none, low, medium or high");
    }
    let temperature = parsed
        .value("--temperature")
        .map(|t| match t.parse::<f64>() {
            Ok(v) if v.is_finite() && v >= 0.0 => Ok(v),
            _ => usage_err("--temperature needs a number of at least 0"),
        })
        .transpose()?;
    let options = ChatOptions {
        system: parsed.value("--system").map(str::to_owned),
        temperature,
        max_tokens: parsed.number("--max-tokens")?,
        reasoning,
        stats: parsed.flag("--stats"),
        json,
    };
    let mut messages = Vec::new();
    if let Some(system) = &options.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    let request = |messages: &[Value]| {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if let Some(t) = options.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(n) = options.max_tokens {
            body["max_tokens"] = json!(n);
        }
        if let Some(r) = &options.reasoning {
            body["reasoning_effort"] = json!(r);
        }
        body
    };
    let stdin_tty = std::io::stdin().is_terminal();
    let prompt = match parsed.positional.get(1) {
        Some(p) => Some(p.clone()),
        None if !stdin_tty => {
            let mut text = String::new();
            std::io::stdin()
                .lock()
                .take(16 << 20)
                .read_to_string(&mut text)
                .map_err(|_| Failure::Usage("The prompt cannot be read from stdin".to_owned()))?;
            let text = text.trim_end().to_owned();
            if text.is_empty() {
                return usage_err("No prompt: give PROMPT or pipe text into stdin");
            }
            Some(text)
        }
        None => None,
    };
    if let Some(prompt) = prompt {
        messages.push(json!({"role": "user", "content": prompt}));
        stream_chat(client, &request(&messages), &options)?;
        return Ok(());
    }
    if json {
        return usage_err("--json needs a PROMPT or piped input");
    }
    eprintln!("Chatting with {model}. An empty line or Ctrl-D ends the conversation.");
    let stdin = std::io::stdin();
    loop {
        print!(">>> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            println!();
            return Ok(());
        }
        messages.push(json!({"role": "user", "content": line.trim_end()}));
        match stream_chat(client, &request(&messages), &options) {
            Ok(answer) => messages.push(json!({"role": "assistant", "content": answer})),
            Err(failure) => {
                // The failed turn is dropped so the history stays consistent.
                messages.pop();
                return Err(failure);
            }
        }
        println!();
    }
}

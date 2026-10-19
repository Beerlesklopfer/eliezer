//! Pure output formatting of the command line client: tables, sizes, durations and the text
//! of each command. No I/O here, so that tests can include this file directly.
use serde_json::Value;

/// Bytes as GB (decimal, one digit); below 0.1 GB as MB, like `lms ls` does for small models.
pub fn bytes(n: u64) -> String {
    if n < 100_000_000 {
        format!("{:.1} MB", n as f64 / 1e6)
    } else {
        format!("{:.1} GB", n as f64 / 1e9)
    }
}

/// Milliseconds as a short readable duration: `850 ms`, `12.3 s`, `4m 05s`, `2h 03m`, `1d 02h`.
pub fn duration_ms(ms: u64) -> String {
    if ms < 1000 {
        return format!("{ms} ms");
    }
    let secs = ms / 1000;
    if secs < 60 {
        return format!("{:.1} s", ms as f64 / 1000.0);
    }
    let (mins, s) = (secs / 60, secs % 60);
    if mins < 60 {
        return format!("{mins}m {s:02}s");
    }
    let (hours, m) = (mins / 60, mins % 60);
    if hours < 24 {
        return format!("{hours}h {m:02}m");
    }
    format!("{}d {:02}h", hours / 24, hours % 24)
}

/// Seconds (as the API reports load times) as a duration.
pub fn duration_secs(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "-".to_owned();
    }
    duration_ms((secs * 1000.0).round() as u64)
}

/// Aligned table: upper-case header, columns separated by four spaces, no trailing blanks.
/// Columns listed in `right` are right-aligned (numbers).
pub fn table(headers: &[&str], rows: &[Vec<String>], right: &[usize]) -> String {
    let columns = headers.len();
    let mut width: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate().take(columns) {
            width[i] = width[i].max(cell.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        let mut out = String::new();
        for (i, cell) in cells.iter().enumerate() {
            let pad = width[i].saturating_sub(cell.chars().count());
            if i > 0 {
                out.push_str("    ");
            }
            if right.contains(&i) {
                out.push_str(&" ".repeat(pad));
                out.push_str(cell);
            } else {
                out.push_str(cell);
                if i + 1 < columns {
                    out.push_str(&" ".repeat(pad));
                }
            }
        }
        out.trim_end().to_owned()
    };
    let mut out = line(headers.to_vec());
    out.push('\n');
    for row in rows {
        out.push_str(&line(row.iter().map(String::as_str).collect()));
        out.push('\n');
    }
    out
}

fn text(v: &Value) -> Option<&str> {
    v.as_str().filter(|s| !s.is_empty())
}
fn opt_bytes(v: &Value) -> String {
    v.as_u64().map_or_else(|| "-".to_owned(), bytes)
}
fn opt_num(v: &Value) -> String {
    v.as_u64().map_or_else(|| "-".to_owned(), |n| n.to_string())
}

/// State of a model: the status report wins, the loaded instances of the model list are the fallback.
fn state_of(model: &Value, status: Option<&Value>) -> String {
    let key = model["key"].as_str().unwrap_or_default();
    if let Some(entry) = status
        .and_then(|s| s["models"].as_array())
        .and_then(|list| list.iter().find(|m| m["key"] == key))
        && let Some(state) = text(&entry["state"])
    {
        return state.to_owned();
    }
    if model["loaded_instances"]
        .as_array()
        .is_some_and(|l| !l.is_empty())
    {
        "loaded".to_owned()
    } else {
        "unloaded".to_owned()
    }
}

/// `eliezer ls`: the configured models.
pub fn ls_text(models: &Value, status: Option<&Value>) -> String {
    let list = models["models"].as_array().cloned().unwrap_or_default();
    if list.is_empty() {
        return "No models are configured.\n".to_owned();
    }
    let total: u64 = list.iter().filter_map(|m| m["size_bytes"].as_u64()).sum();
    let out = format!(
        "You have {} configured model{}, {} in total.\n\n",
        list.len(),
        if list.len() == 1 { "" } else { "s" },
        bytes(total)
    );
    let rows: Vec<Vec<String>> = list
        .iter()
        .map(|m| {
            vec![
                m["key"].as_str().unwrap_or("-").to_owned(),
                m["type"].as_str().unwrap_or("-").to_owned(),
                opt_bytes(&m["size_bytes"]),
                opt_num(&m["max_context_length"]),
                state_of(m, status),
            ]
        })
        .collect();
    out + &table(
        &["MODEL", "TYPE", "SIZE", "CONTEXT", "STATE"],
        &rows,
        &[2, 3],
    )
}

/// `eliezer ps`: models that occupy memory (loading, loaded, unloading).
pub fn ps_text(models: &Value, status: &Value) -> String {
    let empty = Vec::new();
    let config = models["models"].as_array().unwrap_or(&empty);
    let rows: Vec<Vec<String>> = status["models"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter(|m| {
            matches!(
                m["state"].as_str(),
                Some("loaded" | "loading" | "unloading")
            )
        })
        .map(|m| {
            let instance = config
                .iter()
                .find(|c| c["key"] == m["key"])
                .and_then(|c| c["loaded_instances"].get(0))
                .map(|i| &i["config"]);
            let get = |name: &str| instance.map_or_else(|| "-".to_owned(), |c| opt_num(&c[name]));
            vec![
                m["key"].as_str().unwrap_or("-").to_owned(),
                m["state"].as_str().unwrap_or("-").to_owned(),
                m["device"].as_str().unwrap_or("gpu").to_owned(),
                opt_bytes(&m["vram_bytes"]),
                get("context_length"),
                get("parallel"),
                m["last_used_ms_ago"]
                    .as_u64()
                    .map_or("-".to_owned(), duration_ms),
                opt_num(&m["in_flight"]),
                opt_num(&m["queue_len"]),
            ]
        })
        .collect();
    if rows.is_empty() {
        return "No models are currently loaded.\n\nTo load a model, run:\n\n    eliezer load <model>\n"
            .to_owned();
    }
    table(
        &[
            "MODEL",
            "STATE",
            "DEVICE",
            "MEMORY",
            "CONTEXT",
            "PARALLEL",
            "IDLE",
            "IN-FLIGHT",
            "QUEUED",
        ],
        &rows,
        &[3, 4, 5, 7, 8],
    )
}

/// `eliezer status`.
pub fn status_text(base: &str, health: &str, status: &Value) -> String {
    let gpu = &status["gpu"];
    let (total, used) = (gpu["total_bytes"].as_u64(), gpu["used_bytes"].as_u64());
    let percent = match (total, used) {
        (Some(t), Some(u)) if t > 0 => format!(" ({}%)", u * 100 / t),
        _ => String::new(),
    };
    let probe = gpu["probe"].as_str().unwrap_or("unknown");
    let method = if probe == "none" {
        "none (bookkeeping of the loaded models against the budget)".to_owned()
    } else {
        probe.to_owned()
    };
    let models = status["models"].as_array().cloned().unwrap_or_default();
    let loaded = models
        .iter()
        .filter(|m| m["state"].as_str() == Some("loaded"))
        .count();
    let queue = status["queue"].as_array().cloned().unwrap_or_default();
    let queue_text = if queue.is_empty() {
        "empty".to_owned()
    } else {
        let waiting: Vec<String> = queue
            .iter()
            .map(|q| {
                format!(
                    "{} ({})",
                    q["model"].as_str().unwrap_or("?"),
                    duration_secs(q["waiting_secs"].as_f64().unwrap_or(0.0))
                )
            })
            .collect();
        format!("{} waiting: {}", queue.len(), waiting.join(", "))
    };
    // The packaged default budget (8 GB) is meant for the smallest supported card; on a bigger
    // card it silently refuses every large model, so say so.
    let hint = match (gpu["budget_bytes"].as_u64(), total) {
        (Some(budget), Some(total)) if probe != "none" && budget * 10 < total * 6 => format!(
            "\nNote: the budget ({}) is far below the card's memory ({}). Raise `vram_budget_gb` in the\n\
             configuration (a little less than the card's memory), then `systemctl reload eliezer`,\n\
             or models above the budget are refused.\n",
            opt_bytes(&gpu["budget_bytes"]),
            opt_bytes(&gpu["total_bytes"]),
        ),
        _ => String::new(),
    };
    let cpu = &status["cpu"];
    let cpu_text = match cpu["budget_bytes"].as_u64() {
        Some(budget) if budget > 0 => format!(
            "CPU memory   {} used of {} budget\n",
            opt_bytes(&cpu["used_bytes"]),
            opt_bytes(&cpu["budget_bytes"])
        ),
        _ => String::new(),
    };
    format!(
        "Service      {base} is running (health: {health})\n\
         Client       eliezer {}\n\
         Measurement  {method}\n\
         GPU total    {}\n\
         GPU used     {}{percent}\n\
         GPU free     {}\n\
         GPU budget   {}\n\
         {cpu_text}Models       {loaded} loaded of {} configured\n\
         Queue        {queue_text}\n{hint}",
        env!("CARGO_PKG_VERSION"),
        opt_bytes(&gpu["total_bytes"]),
        opt_bytes(&gpu["used_bytes"]),
        opt_bytes(&gpu["free_bytes"]),
        opt_bytes(&gpu["budget_bytes"]),
        models.len(),
    )
}

/// `eliezer show`: details of one model plus the endpoints that serve it.
pub fn show_text(model: &Value, status_entry: Option<&Value>, base: &str) -> String {
    let mut rows: Vec<(&str, String)> = vec![
        ("Model", model["key"].as_str().unwrap_or("-").to_owned()),
        ("Type", model["type"].as_str().unwrap_or("-").to_owned()),
    ];
    for (label, field) in [
        ("Display name", "display_name"),
        ("Architecture", "architecture"),
        ("Parameters", "params_string"),
    ] {
        if let Some(v) = text(&model[field]) {
            rows.push((label, v.to_owned()));
        }
    }
    rows.push(("Size", opt_bytes(&model["size_bytes"])));
    rows.push(("Max context", opt_num(&model["max_context_length"])));
    if let Some(q) = text(&model["quantization"]["name"]) {
        rows.push(("Quantization", q.to_owned()));
    }
    let caps = &model["capabilities"];
    rows.push((
        "Vision",
        if caps["vision"].as_bool() == Some(true) {
            "yes"
        } else {
            "no"
        }
        .to_owned(),
    ));
    if let Some(options) = caps["reasoning"]["allowed_options"].as_array() {
        let list: Vec<&str> = options.iter().filter_map(Value::as_str).collect();
        let default = caps["reasoning"]["default"].as_str().unwrap_or("-");
        rows.push((
            "Reasoning",
            format!("{} (default: {default})", list.join(", ")),
        ));
    }
    let state = state_of(model, None);
    let state = status_entry
        .and_then(|e| text(&e["state"]))
        .map_or(state, str::to_owned);
    rows.push(("State", state));
    if let Some(i) = model["loaded_instances"].get(0) {
        rows.push((
            "Loaded with",
            format!(
                "context {}, parallel {}",
                opt_num(&i["config"]["context_length"]),
                opt_num(&i["config"]["parallel"])
            ),
        ));
    }
    if let Some(e) = status_entry {
        if e["vram_bytes"].as_u64().is_some_and(|b| b > 0) {
            rows.push(("VRAM", opt_bytes(&e["vram_bytes"])));
        }
        if let Some(idle) = e["last_used_ms_ago"].as_u64() {
            rows.push(("Idle", duration_ms(idle)));
        }
        rows.push(("In flight", opt_num(&e["in_flight"])));
        rows.push(("Queued", opt_num(&e["queue_len"])));
    }
    let width = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
    let mut out = String::new();
    for (label, value) in rows {
        out += &format!("{label:<width$}  {value}\n");
    }
    let key = model["key"].as_str().unwrap_or("MODEL");
    let kind = model["type"].as_str().unwrap_or("llm");
    out += &format!("\nAPI at {base}\n");
    let endpoints: &[&str] = match kind {
        "embedding" => &["POST /v1/embeddings"],
        "image" => &["POST /v1/images/generations"],
        _ => &[
            "POST /v1/chat/completions    (OpenAI)",
            "POST /v1/completions         (OpenAI)",
            "POST /v1/messages            (Anthropic)",
        ],
    };
    for e in endpoints {
        out += &format!("  {e}\n");
    }
    out += &format!(
        "  POST /api/v1/models/load      {{\"model\": \"{key}\"}}\n  POST /api/v1/models/unload    {{\"instance_id\": \"{key}\"}}\n"
    );
    out
}

/// Token statistics of a chat response (`usage` plus the `eliezer` block of the last frame).
pub fn stats_text(usage: Option<&Value>, eliezer: Option<&Value>) -> String {
    let mut out = String::new();
    if let Some(u) = usage {
        out += &format!(
            "prompt tokens:        {}\ncompletion tokens:    {}\ntotal tokens:         {}\n",
            opt_num(&u["prompt_tokens"]),
            opt_num(&u["completion_tokens"]),
            opt_num(&u["total_tokens"]),
        );
    }
    if let Some(e) = eliezer {
        let ttft = e["ttft_ms"].as_u64();
        let gen_ms = e["gen_ms"].as_u64();
        let tokens = e["tokens"].as_u64();
        if let Some(t) = ttft {
            out += &format!("time to first token:  {}\n", duration_ms(t));
        }
        if let Some(g) = gen_ms {
            out += &format!("total time:           {}\n", duration_ms(g));
        }
        if let (Some(n), Some(t), Some(g)) = (tokens, ttft, gen_ms) {
            // gen_ms is measured from the start of the request: the speed counts after the first token.
            let after = g.saturating_sub(t);
            if n > 1 && after > 0 {
                out += &format!(
                    "speed:                {:.1} tokens/s\n",
                    (n - 1) as f64 / (after as f64 / 1000.0)
                );
            }
        }
    }
    if out.is_empty() {
        out = "No statistics were reported.\n".to_owned();
    }
    out
}

/// The best place a repository's variants fit: `gpu`, `cpu` or `too large`.
fn best_fit(detail: &Value) -> &'static str {
    let fits = |name: &str| {
        detail["variants"]
            .as_array()
            .is_some_and(|l| l.iter().any(|v| v["fit"] == name))
    };
    if fits("gpu") {
        "gpu"
    } else if fits("cpu") {
        "cpu"
    } else {
        "too large"
    }
}

/// `eliezer search`: the candidates; with details also the support by a runtime and the best fit.
pub fn search_text(result: &Value) -> String {
    let empty = Vec::new();
    let hits = result["data"].as_array().unwrap_or(&empty);
    if hits.is_empty() {
        return "No repositories found.\n".to_owned();
    }
    let detailed = hits.iter().any(|h| h["detail"].is_object());
    let rows: Vec<Vec<String>> = hits
        .iter()
        .map(|h| {
            let mut row = vec![
                h["repo"].as_str().unwrap_or("-").to_owned(),
                opt_num(&h["downloads"]),
                text(&h["license"]).unwrap_or("-").to_owned(),
                text(&h["task"]).unwrap_or("-").to_owned(),
            ];
            if detailed {
                let d = &h["detail"];
                row.push(text(&d["support"]["level"]).unwrap_or("-").to_owned());
                row.push(if d.is_object() {
                    best_fit(d).to_owned()
                } else {
                    "-".to_owned()
                });
            }
            row
        })
        .collect();
    let mut out = if detailed {
        table(
            &[
                "REPOSITORY",
                "DOWNLOADS",
                "LICENSE",
                "TASK",
                "SUPPORT",
                "FITS",
            ],
            &rows,
            &[1],
        )
    } else {
        table(&["REPOSITORY", "DOWNLOADS", "LICENSE", "TASK"], &rows, &[1])
    };
    out.push_str("\nNothing is downloaded. `eliezer inspect ORG/REPO` lists files and fit of one repository.\n");
    out
}

/// `eliezer inspect ORG/REPO`.
pub fn inspect_text(detail: &Value) -> String {
    let mut out = format!(
        "Repository   {}\nCommit       {}\nLicense      {}{}\nTask         {}\nArchitecture {}\nSupport      {}{}\n",
        detail["repo"].as_str().unwrap_or("-"),
        text(&detail["sha"]).unwrap_or("-"),
        text(&detail["license"]).unwrap_or("-"),
        if detail["gated"] == true {
            "  (gated: access must be requested on Hugging Face)"
        } else {
            ""
        },
        text(&detail["task"]).unwrap_or("-"),
        text(&detail["architecture"]).unwrap_or("not reported"),
        text(&detail["support"]["level"]).unwrap_or("unknown"),
        text(&detail["support"]["backend"]).map_or_else(String::new, |b| format!("  ({b})")),
    );
    let empty = Vec::new();
    let rows: Vec<Vec<String>> = detail["variants"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .map(|v| {
            vec![
                v["file"].as_str().unwrap_or("-").to_owned(),
                text(&v["quantization"]).unwrap_or("-").to_owned(),
                opt_bytes(&v["bytes"]),
                opt_bytes(&v["estimate_bytes"]),
                v["fit"].as_str().unwrap_or("-").replace('_', " "),
            ]
        })
        .collect();
    if rows.is_empty() {
        out.push_str("\nNo weight files of a supported format.\n");
    } else {
        out.push('\n');
        out.push_str(&table(
            &["FILE", "QUANT", "SIZE", "NEEDS", "FITS"],
            &rows,
            &[2, 3],
        ));
        out.push_str("NEEDS is the file size plus 20 % (and the projector); the exact figure comes with `eliezer --suggest`.\n");
    }
    if let Some(p) = text(&detail["projector"]["file"]) {
        out.push_str(&format!(
            "\nVision projector: {p} ({})\n",
            opt_bytes(&detail["projector"]["bytes"])
        ));
    }
    if let Some(files) = detail["required_files"]
        .as_array()
        .filter(|l| !l.is_empty())
    {
        out.push_str("\nFurther files the model needs:\n");
        for f in files {
            out.push_str(&format!(
                "  {} ({})\n",
                f["file"].as_str().unwrap_or("-"),
                opt_bytes(&f["bytes"])
            ));
        }
    }
    out.push_str("\nNothing is downloaded. Downloading is not implemented yet: put the files in the model store and run `eliezer --import`.\n");
    out
}

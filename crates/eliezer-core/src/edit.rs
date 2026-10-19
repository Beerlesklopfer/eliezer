//! Targeted edits of the configuration text: set one key inside one section while everything else,
//! including comments, stays as it is. Used by the commands that store secrets (`--new-key`,
//! `--set-hf-token`) to enter the path of the secret file.

#[derive(Debug, PartialEq, Eq)]
pub enum EditError {
    /// The value would need escaping; paths and names used here never do.
    UnsafeValue,
}
impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("The value contains characters that cannot be written safely")
    }
}
impl std::error::Error for EditError {}

fn header(line: &str) -> Option<&str> {
    let t = line.trim();
    (t.starts_with('[') && !t.starts_with("[[") && t.ends_with(']'))
        .then(|| t[1..t.len() - 1].trim())
}

fn assigns(line: &str, key: &str) -> bool {
    let t = line.trim_start();
    t.strip_prefix(key)
        .is_some_and(|rest| rest.trim_start().starts_with('='))
}

/// Sets `key = "value"` in `[section]`: replaces an existing active line, otherwise inserts it
/// right after the section header, otherwise appends the section. Commented-out lines are left.
pub fn set_string(text: &str, section: &str, key: &str, value: &str) -> Result<String, EditError> {
    if value
        .chars()
        .any(|c| c == '"' || c == '\\' || c.is_control())
    {
        return Err(EditError::UnsafeValue);
    }
    let line = format!("{key} = \"{value}\"");
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let mut in_section = false;
    let mut header_at = None;
    for (index, current) in lines.iter().enumerate() {
        if let Some(name) = header(current) {
            in_section = name == section;
            if in_section && header_at.is_none() {
                header_at = Some(index);
            }
        } else if current.trim_start().starts_with("[[") {
            in_section = false;
        } else if in_section && assigns(current, key) {
            let replaced = lines.clone();
            let mut out = replaced;
            out[index] = line;
            return Ok(join(&out));
        }
    }
    match header_at {
        Some(at) => lines.insert(at + 1, line),
        None => {
            if lines.last().is_some_and(|l| !l.trim().is_empty()) {
                lines.push(String::new());
            }
            lines.push(format!("[{section}]"));
            lines.push(line);
        }
    }
    Ok(join(&lines))
}

fn join(lines: &[String]) -> String {
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

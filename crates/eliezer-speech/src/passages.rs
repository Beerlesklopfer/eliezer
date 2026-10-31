//! Cutting a text into passages for speech synthesis. A passage ends at a sentence boundary and is at most
//! `max_chars` characters long, so a synthesis model never gets more than it can read in one go and a chapter of any
//! length can be read. Paragraphs (separated by a blank line) are never merged; a sentence that is longer than a
//! passage is cut at a clause mark, then between words, and only as a last resort inside a word.

/// One piece of the text to synthesize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passage {
    pub text: String,
    /// The passage is the last one of its paragraph (a longer pause follows).
    pub ends_paragraph: bool,
}

/// Characters that end a sentence.
fn terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？')
}
/// Closing marks that stay with the sentence they close.
fn closer(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | '”' | '’' | '»' | '«' | ')' | ']' | '“' | '‘'
    )
}
/// Marks inside a sentence where it may be cut when it is too long.
fn clause_mark(c: char) -> bool {
    matches!(c, ',' | ';' | ':' | '–' | '—' | '，' | '；' | '：')
}

/// The sentences of one paragraph (whitespace folded to single spaces, nothing else changed).
fn sentences(paragraph: &str) -> Vec<String> {
    let folded: Vec<char> = paragraph
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < folded.len() {
        if !terminator(folded[i]) {
            i += 1;
            continue;
        }
        // The run of terminators and closing marks belongs to the sentence.
        let mut end = i + 1;
        while end < folded.len() && (terminator(folded[end]) || closer(folded[end])) {
            end += 1;
        }
        let at_end = end >= folded.len();
        let space_after = !at_end && folded[end] == ' ';
        if at_end || (space_after && boundary(&folded, start, i, end)) {
            out.push(folded[start..end].iter().collect::<String>());
            start = end;
            while start < folded.len() && folded[start] == ' ' {
                start += 1;
            }
            i = start;
        } else {
            i = end;
        }
    }
    if start < folded.len() {
        out.push(folded[start..].iter().collect());
    }
    out.retain(|s| !s.trim().is_empty());
    out
}

/// Is the full stop at `dot` (followed by a space at `end`) the end of a sentence? Not when it follows a number
/// (an ordinal or a date: "am 3. Mai"), a single letter (an initial or "z. B."), or when a lower-case letter follows
/// (after any terminator).
fn boundary(chars: &[char], start: usize, dot: usize, end: usize) -> bool {
    // Whatever ends the sentence, a lower-case word after it goes on with it ("Wirklich?" fragte sie).
    let next = chars[end..].iter().find(|c| **c != ' ');
    if next.is_some_and(|c| c.is_lowercase()) {
        return false;
    }
    if chars[dot] != '.' || end != dot + 1 {
        return true;
    }
    let token_start = chars[start..dot]
        .iter()
        .rposition(|c| *c == ' ')
        .map_or(start, |p| start + p + 1);
    let token = &chars[token_start..dot];
    if !token.is_empty() && token.iter().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if token.len() == 1 && token[0].is_alphabetic() {
        return false;
    }
    true
}

/// Pieces of `sentence` of at most `max` characters: cut after clause marks, else between words, else in a word.
fn split_long(sentence: &str, max: usize) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    // First the clauses.
    let mut clauses: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in sentence.chars() {
        current.push(c);
        if clause_mark(c) {
            clauses.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        clauses.push(current);
    }
    let mut line = String::new();
    let flush = |line: &mut String, pieces: &mut Vec<String>| {
        let text = line.trim();
        if !text.is_empty() {
            pieces.push(text.to_owned());
        }
        line.clear();
    };
    for clause in clauses {
        if clause.chars().count() > max {
            flush(&mut line, &mut pieces);
            // Between words, packed to the limit.
            for word in clause.split_whitespace() {
                let mut word = word;
                // A word longer than a passage: cut it.
                while word.chars().count() > max {
                    let cut = word.char_indices().nth(max).map_or(word.len(), |(i, _)| i);
                    if !line.is_empty() {
                        flush(&mut line, &mut pieces);
                    }
                    pieces.push(word[..cut].to_owned());
                    word = &word[cut..];
                }
                if line.chars().count() + 1 + word.chars().count() > max && !line.is_empty() {
                    flush(&mut line, &mut pieces);
                }
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(word);
            }
            flush(&mut line, &mut pieces);
        } else if line.chars().count() + clause.chars().count() > max {
            flush(&mut line, &mut pieces);
            line.push_str(&clause);
        } else {
            line.push_str(&clause);
        }
    }
    flush(&mut line, &mut pieces);
    pieces
}

/// Cuts `text` into passages of at most `max_chars` characters (at least 1). Empty passages and passages without a
/// letter or digit (a line of asterisks) are dropped.
pub fn passages(text: &str, max_chars: usize) -> Vec<Passage> {
    let max = max_chars.max(1);
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut out: Vec<Passage> = Vec::new();
    // A blank line (possibly with spaces) separates paragraphs.
    let mut paragraphs: Vec<String> = vec![String::new()];
    for line in text.split('\n') {
        if line.trim().is_empty() {
            if !paragraphs.last().is_some_and(String::is_empty) {
                paragraphs.push(String::new());
            }
        } else if let Some(last) = paragraphs.last_mut() {
            last.push(' ');
            last.push_str(line);
        }
    }
    for paragraph in paragraphs {
        let mut current = String::new();
        let mut items: Vec<String> = Vec::new();
        for sentence in sentences(&paragraph) {
            let pieces = if sentence.chars().count() > max {
                split_long(&sentence, max)
            } else {
                vec![sentence]
            };
            for piece in pieces {
                if !current.is_empty() && current.chars().count() + 1 + piece.chars().count() > max
                {
                    items.push(std::mem::take(&mut current));
                }
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&piece);
            }
        }
        if !current.is_empty() {
            items.push(current);
        }
        items.retain(|t| t.chars().any(char::is_alphanumeric));
        let count = items.len();
        for (index, text) in items.into_iter().enumerate() {
            out.push(Passage {
                text,
                ends_paragraph: index + 1 == count,
            });
        }
    }
    out
}

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{Result, invalid};

pub const MAX_CANDIDATES: usize = 64;

const ABBREVIATIONS: &[&str] = &[
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "ltd",
    "co", "corp", "fig", "no", "approx", "u.s", "u.k",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub source_id: String,
    #[serde(alias = "text")]
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sentence {
    pub source_id: String,
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

pub fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_abbreviation(before: &str) -> bool {
    let word = before
        .rsplit(|c: char| c.is_whitespace() || c == '(' || c == '"')
        .next()
        .unwrap_or("")
        .trim_end_matches('.')
        .to_lowercase();
    if word.chars().count() == 1 && word.chars().all(char::is_alphabetic) {
        return true;
    }
    ABBREVIATIONS.contains(&word.as_str())
}

fn push_span(spans: &mut Vec<(usize, usize)>, text: &str, start: usize, end: usize) {
    let piece = &text[start..end];
    let lead = piece.len() - piece.trim_start().len();
    let trimmed = piece.trim();
    if trimmed.chars().any(char::is_alphanumeric) {
        spans.push((start + lead, start + lead + trimmed.len()));
    }
}

pub fn segment(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let (pos, c) = chars[i];
        if c == '\n' {
            push_span(&mut spans, text, start, pos);
            start = pos + 1;
            i += 1;
            continue;
        }
        if matches!(c, '.' | '!' | '?') {
            let mut j = i + 1;
            while j < chars.len()
                && matches!(
                    chars[j].1,
                    '.' | '!' | '?' | '"' | '\'' | ')' | ']' | '”' | '’'
                )
            {
                j += 1;
            }
            let end = chars.get(j).map_or(text.len(), |&(p, _)| p);
            let at_break = j >= chars.len() || chars[j].1.is_whitespace();
            if at_break && !(c == '.' && is_abbreviation(&text[start..pos])) {
                push_span(&mut spans, text, start, end);
                start = end;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    push_span(&mut spans, text, start, text.len());
    spans
}

pub fn validate_outputs(outputs: &[ToolOutput]) -> Result<()> {
    if outputs.is_empty() {
        return invalid("no tool outputs");
    }
    let mut seen = HashSet::new();
    for o in outputs {
        if o.source_id.trim().is_empty() {
            return invalid("tool output with empty source_id");
        }
        if !seen.insert(o.source_id.as_str()) {
            return invalid(format!("duplicate source_id {}", o.source_id));
        }
        if o.content.trim().is_empty() {
            return invalid(format!("tool output {} is empty", o.source_id));
        }
    }
    Ok(())
}

pub fn candidates(outputs: &[ToolOutput]) -> Result<(Vec<Sentence>, bool)> {
    validate_outputs(outputs)?;
    let mut out = Vec::new();
    let mut truncated = false;
    for o in outputs {
        for (start, end) in segment(&o.content) {
            if out.len() == MAX_CANDIDATES {
                truncated = true;
                break;
            }
            out.push(Sentence {
                source_id: o.source_id.clone(),
                index: out.len(),
                start,
                end,
                text: normalize(&o.content[start..end]),
            });
        }
    }
    if out.is_empty() {
        return invalid("tool outputs contain no sentences");
    }
    Ok((out, truncated))
}

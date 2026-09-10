use std::collections::{BTreeSet, HashSet};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::Deserialize;

use super::example::{Provenance, RamanujanExample};
use crate::core::input::normalize;
use crate::core::{
    MAX_CANDIDATES, MAX_SENTENCES, QaPair, Sentence, ToolOutput, candidates, segment,
};
use crate::{Result, invalid};

pub const QUESTION: &str = "What do you want to know?";
const CONTEXT_TURNS: [usize; 3] = [2, 1, 0];
const TURNS_PER_SOURCE: usize = 6;
const SENTENCES_PER_SOURCE: usize = 8;
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "that", "this", "with", "from", "has", "have",
    "had", "but", "not", "you", "his", "her", "its", "they", "their", "will", "would", "said",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dataset {
    Qmsum,
    CnnDm,
}

impl Dataset {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "qmsum" => Ok(Dataset::Qmsum),
            "cnndm" => Ok(Dataset::CnnDm),
            _ => invalid(format!("unknown dataset {name}; expected qmsum or cnndm")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConvertStats {
    pub read: usize,
    pub written: usize,
    pub skipped: usize,
}

#[derive(Deserialize)]
struct QmTurn {
    speaker: String,
    content: String,
}

#[derive(Deserialize)]
struct QmQuery {
    query: String,
    answer: String,
    #[serde(default)]
    relevant_text_span: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct QmMeeting {
    #[serde(default)]
    specific_query_list: Vec<QmQuery>,
    meeting_transcripts: Vec<QmTurn>,
}

#[derive(Deserialize)]
struct CnnRow {
    #[serde(default)]
    id: String,
    article: String,
    highlights: String,
}

pub fn truncate_sentences(text: &str, n: usize) -> String {
    segment(text)
        .into_iter()
        .take(n)
        .map(|(s, e)| normalize(&text[s..e]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| w.chars().count() >= 3 && !STOPWORDS.contains(&w.as_str()))
        .collect()
}

pub fn oracle(sentences: &[Sentence], target: &str, k: usize) -> Vec<usize> {
    let want = tokens(target);
    let sets: Vec<HashSet<String>> = sentences.iter().map(|s| tokens(&s.text)).collect();
    let mut covered = HashSet::new();
    let mut picked = Vec::new();
    while picked.len() < k {
        let best = sets
            .iter()
            .enumerate()
            .filter(|(i, _)| !picked.contains(i))
            .map(|(i, s)| {
                (
                    i,
                    s.iter()
                        .filter(|w| want.contains(*w) && !covered.contains(*w))
                        .count(),
                )
            })
            .max_by_key(|&(i, gain)| (gain, std::cmp::Reverse(i)));
        match best {
            Some((i, gain)) if gain > 0 => {
                covered.extend(sets[i].iter().filter(|w| want.contains(*w)).cloned());
                picked.push(i);
            }
            _ => break,
        }
    }
    picked.sort_unstable();
    picked
}

pub fn clean_transcript(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for c in text.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    let mut out = normalize(&out);
    for (from, to) in [
        (" .", "."),
        (" ,", ","),
        (" ?", "?"),
        (" !", "!"),
        (" 's", "'s"),
        (" 're", "'re"),
        (" 've", "'ve"),
        (" 'll", "'ll"),
        (" 'd", "'d"),
        (" 'm", "'m"),
        (" n't", "n't"),
        ("@ @", ""),
    ] {
        out = out.replace(from, to);
    }
    normalize(&out)
}

fn parse_spans(spans: &[Vec<String>], n_turns: usize) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for span in spans {
        let parsed: Vec<usize> = span.iter().filter_map(|s| s.trim().parse().ok()).collect();
        if let [a, b] = parsed[..] {
            for t in a.min(b)..=a.max(b).min(n_turns.saturating_sub(1)) {
                out.insert(t);
            }
        }
    }
    out
}

fn qmsum_sources(
    turns: &[QmTurn],
    window: &[usize],
) -> (Vec<ToolOutput>, Vec<Vec<(usize, usize)>>) {
    let mut outputs = Vec::new();
    let mut lines = Vec::new();
    let mut group: Vec<usize> = Vec::new();
    let flush = |group: &mut Vec<usize>,
                 outputs: &mut Vec<ToolOutput>,
                 lines: &mut Vec<Vec<(usize, usize)>>| {
        if group.is_empty() {
            return;
        }
        let mut content = String::new();
        let mut map = Vec::new();
        for &t in group.iter() {
            if !content.is_empty() {
                content.push('\n');
            }
            map.push((content.len(), t));
            let turn = &turns[t];
            content.push_str(&format!(
                "{}: {}",
                normalize(&turn.speaker),
                clean_transcript(&turn.content)
            ));
        }
        outputs.push(ToolOutput {
            source_id: format!("turns-{}-{}", group[0], group[group.len() - 1]),
            content,
        });
        lines.push(map);
        group.clear();
    };
    for &t in window {
        let gap = group.last().is_some_and(|&l| l + 1 != t);
        if gap || group.len() == TURNS_PER_SOURCE {
            flush(&mut group, &mut outputs, &mut lines);
        }
        group.push(t);
    }
    flush(&mut group, &mut outputs, &mut lines);
    (outputs, lines)
}

pub fn qmsum_meeting(json: &str, meeting_id: &str) -> Result<(Vec<RamanujanExample>, usize)> {
    let meeting: QmMeeting = serde_json::from_str(json)?;
    let turns = &meeting.meeting_transcripts;
    let queries = &meeting.specific_query_list;
    let mut out = Vec::new();
    let mut skipped = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let relevant = parse_spans(&q.relevant_text_span, turns.len());
        let target = truncate_sentences(&q.answer, MAX_SENTENCES);
        if relevant.is_empty() || target.is_empty() || q.query.trim().is_empty() {
            skipped += 1;
            continue;
        }
        let built = CONTEXT_TURNS.iter().find_map(|&ctx| {
            let window: Vec<usize> = relevant
                .iter()
                .flat_map(|&t| t.saturating_sub(ctx)..=(t + ctx).min(turns.len() - 1))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let (outputs, lines) = qmsum_sources(turns, &window);
            match candidates(&outputs) {
                Ok((sentences, false)) => Some((outputs, lines, sentences)),
                _ => None,
            }
        });
        let Some((outputs, lines, sentences)) = built else {
            skipped += 1;
            continue;
        };
        let evidence_labels: Vec<usize> = sentences
            .iter()
            .filter(|s| {
                let src = outputs
                    .iter()
                    .position(|o| o.source_id == s.source_id)
                    .unwrap();
                let turn = lines[src]
                    .iter()
                    .rev()
                    .find(|(off, _)| *off <= s.start)
                    .map(|&(_, t)| t);
                turn.is_some_and(|t| relevant.contains(&t))
            })
            .map(|s| s.index)
            .collect();
        if evidence_labels.is_empty() {
            skipped += 1;
            continue;
        }
        let qa_pairs = queries
            .iter()
            .enumerate()
            .map(|(j, other)| QaPair {
                question_id: Some(format!("q-{j}")),
                question: QUESTION.into(),
                answer: normalize(&other.query),
                selected: j == qi,
            })
            .collect();
        out.push(RamanujanExample {
            id: format!("qmsum-{meeting_id}-{qi}"),
            tool_outputs: outputs,
            qa_pairs,
            evidence_labels,
            target_summary: target,
            provenance: Provenance {
                dataset: "qmsum".into(),
                group: Some(meeting_id.into()),
            },
        });
    }
    Ok((out, skipped))
}

fn end_sentence(s: &str) -> String {
    let s = normalize(s);
    if s.ends_with(['.', '!', '?']) {
        s
    } else {
        format!("{s}.")
    }
}

pub fn cnndm_row(json: &str, fallback_id: &str) -> Result<Option<RamanujanExample>> {
    let row: CnnRow = serde_json::from_str(json)?;
    let article = normalize(&row.article);
    let spans: Vec<(usize, usize)> = segment(&article).into_iter().take(MAX_CANDIDATES).collect();
    let target = row
        .highlights
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(MAX_SENTENCES)
        .map(end_sentence)
        .collect::<Vec<_>>()
        .join(" ");
    if spans.is_empty() || target.is_empty() {
        return Ok(None);
    }
    let outputs: Vec<ToolOutput> = spans
        .chunks(SENTENCES_PER_SOURCE)
        .enumerate()
        .map(|(k, chunk)| ToolOutput {
            source_id: format!("chunk-{k}"),
            content: chunk
                .iter()
                .map(|&(s, e)| &article[s..e])
                .collect::<Vec<_>>()
                .join(" "),
        })
        .collect();
    let Ok((sentences, false)) = candidates(&outputs) else {
        return Ok(None);
    };
    let evidence_labels = oracle(&sentences, &target, MAX_SENTENCES);
    if evidence_labels.is_empty() {
        return Ok(None);
    }
    let id = if row.id.is_empty() {
        fallback_id.to_string()
    } else {
        row.id
    };
    Ok(Some(RamanujanExample {
        id: format!("cnndm-{id}"),
        tool_outputs: outputs,
        qa_pairs: vec![QaPair {
            question_id: Some("q-0".into()),
            question: QUESTION.into(),
            answer: "The main points.".into(),
            selected: true,
        }],
        evidence_labels,
        target_summary: target,
        provenance: Provenance {
            dataset: "cnndm".into(),
            group: None,
        },
    }))
}

pub fn convert_file(
    dataset: Dataset,
    src: impl AsRef<Path>,
    out: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<ConvertStats> {
    let src = src.as_ref();
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("data")
        .to_string();
    let reader = BufReader::new(std::fs::File::open(src)?);
    let mut w = BufWriter::new(std::fs::File::create(out)?);
    let mut stats = ConvertStats::default();
    for (n, line) in reader.lines().enumerate() {
        if limit.is_some_and(|l| stats.written >= l) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        stats.read += 1;
        let id = format!("{stem}-{n}");
        let examples = match dataset {
            Dataset::Qmsum => {
                let (examples, skipped) = qmsum_meeting(&line, &id)?;
                stats.skipped += skipped;
                examples
            }
            Dataset::CnnDm => match cnndm_row(&line, &id)? {
                Some(e) => vec![e],
                None => {
                    stats.skipped += 1;
                    Vec::new()
                }
            },
        };
        for e in examples {
            if limit.is_some_and(|l| stats.written >= l) {
                break;
            }
            e.validate()?;
            serde_json::to_writer(&mut w, &e)?;
            w.write_all(b"\n")?;
            stats.written += 1;
        }
    }
    w.flush()?;
    Ok(stats)
}

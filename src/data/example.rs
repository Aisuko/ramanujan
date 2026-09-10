use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::core::{Intent, MAX_SENTENCES, QaPair, ToolOutput, candidates, sentence_count};
use crate::{Result, invalid};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub dataset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamanujanExample {
    pub id: String,
    pub tool_outputs: Vec<ToolOutput>,
    pub qa_pairs: Vec<QaPair>,
    pub evidence_labels: Vec<usize>,
    pub target_summary: String,
    #[serde(default)]
    pub provenance: Provenance,
}

impl RamanujanExample {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return invalid("example with empty id");
        }
        let (sentences, truncated) = candidates(&self.tool_outputs)?;
        if truncated {
            return invalid(format!(
                "{}: more candidate sentences than allowed",
                self.id
            ));
        }
        Intent::from_pairs(&self.qa_pairs)?;
        if self.evidence_labels.windows(2).any(|w| w[0] >= w[1]) {
            return invalid(format!(
                "{}: evidence labels must be sorted and unique",
                self.id
            ));
        }
        if let Some(&bad) = self.evidence_labels.iter().find(|&&i| i >= sentences.len()) {
            return invalid(format!("{}: evidence label {bad} out of range", self.id));
        }
        let n = sentence_count(&self.target_summary);
        if n == 0 || n > MAX_SENTENCES {
            return invalid(format!("{}: target has {n} sentences", self.id));
        }
        Ok(())
    }
}

pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<RamanujanExample>> {
    let reader = BufReader::new(std::fs::File::open(path)?);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(&line)?);
        }
    }
    Ok(out)
}

pub fn write_jsonl(path: impl AsRef<Path>, examples: &[RamanujanExample]) -> Result<usize> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    for e in examples {
        serde_json::to_writer(&mut w, e)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(examples.len())
}

use serde::{Deserialize, Serialize};

use crate::core::{Constraints, Evidence, sentence_count};
use crate::{Result, invalid};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metrics {
    pub selected_evidence: usize,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub request_id: String,
    pub summary: String,
    pub sentence_count: usize,
    pub evidence: Vec<Evidence>,
    pub metrics: Metrics,
}

impl Response {
    pub fn new(
        request_id: String,
        summary: String,
        evidence: Vec<Evidence>,
        metrics: Metrics,
    ) -> Self {
        Response {
            request_id,
            sentence_count: sentence_count(&summary),
            summary,
            evidence,
            metrics,
        }
    }

    pub fn validate(&self, constraints: &Constraints) -> Result<()> {
        if !constraints.accepts(&self.summary) {
            return invalid(format!(
                "summary has {} sentences, outside {}..={}",
                sentence_count(&self.summary),
                constraints.min_sentences,
                constraints.max_sentences
            ));
        }
        if self.sentence_count != sentence_count(&self.summary) {
            return invalid("sentence_count does not match summary");
        }
        Ok(())
    }
}

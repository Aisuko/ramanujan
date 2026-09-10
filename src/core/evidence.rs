use serde::{Deserialize, Serialize};

use super::input::Sentence;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub source_id: String,
    pub text_span: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sentence_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

impl Evidence {
    pub fn from_sentence(s: &Sentence, score: Option<f32>) -> Self {
        Evidence {
            source_id: s.source_id.clone(),
            text_span: s.text.clone(),
            sentence_index: Some(s.index),
            start: Some(s.start),
            end: Some(s.end),
            score,
        }
    }
}

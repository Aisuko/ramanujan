use serde::{Deserialize, Serialize};

use super::input::segment;
use crate::{Result, invalid};

pub const MAX_SENTENCES: usize = 3;

fn one() -> usize {
    1
}

fn max_sentences() -> usize {
    MAX_SENTENCES
}

fn concise() -> String {
    "concise".into()
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraints {
    #[serde(default = "one")]
    pub min_sentences: usize,
    #[serde(default = "max_sentences")]
    pub max_sentences: usize,
    #[serde(default = "concise")]
    pub style: String,
    #[serde(default = "yes")]
    pub preserve_factuality: bool,
}

impl Default for Constraints {
    fn default() -> Self {
        Constraints {
            min_sentences: one(),
            max_sentences: max_sentences(),
            style: concise(),
            preserve_factuality: yes(),
        }
    }
}

impl Constraints {
    pub fn validate(&self) -> Result<()> {
        if self.min_sentences < 1 || self.min_sentences > self.max_sentences {
            return invalid(format!(
                "sentence range {}..={} is empty",
                self.min_sentences, self.max_sentences
            ));
        }
        if self.max_sentences > MAX_SENTENCES {
            return invalid(format!(
                "max_sentences {} exceeds {MAX_SENTENCES}",
                self.max_sentences
            ));
        }
        Ok(())
    }

    pub fn accepts(&self, summary: &str) -> bool {
        let n = sentence_count(summary);
        !summary.contains('\n') && (self.min_sentences..=self.max_sentences).contains(&n)
    }
}

pub fn sentence_count(text: &str) -> usize {
    segment(text).len()
}

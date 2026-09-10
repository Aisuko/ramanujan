use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{Result, invalid};

fn selected_default() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QaPair {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question_id: Option<String>,
    pub question: String,
    pub answer: String,
    #[serde(default = "selected_default")]
    pub selected: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Intent {
    pub selected: Vec<QaPair>,
    pub unselected: Vec<QaPair>,
}

impl Intent {
    pub fn from_pairs(pairs: &[QaPair]) -> Result<Self> {
        let mut ids = HashSet::new();
        for p in pairs {
            if p.question.trim().is_empty() {
                return invalid("QA pair with empty question");
            }
            if p.selected && p.answer.trim().is_empty() {
                return invalid(format!("selected question '{}' has no answer", p.question));
            }
            if let Some(id) = &p.question_id
                && !ids.insert(id.as_str())
            {
                return invalid(format!("duplicate question_id {id}"));
            }
        }
        let (selected, unselected): (Vec<_>, Vec<_>) =
            pairs.iter().cloned().partition(|p| p.selected);
        if selected.is_empty() {
            return invalid("no selected QA pairs");
        }
        Ok(Intent {
            selected,
            unselected,
        })
    }
}

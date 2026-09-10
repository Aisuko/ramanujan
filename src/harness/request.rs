use serde::{Deserialize, Serialize};

use crate::core::{Constraints, Intent, QaPair, Sentence, ToolOutput, candidates};
use crate::{Result, invalid};

fn focused() -> String {
    "focused_summarisation".into()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub request_id: String,
    #[serde(default = "focused")]
    pub task: String,
    pub tool_outputs: Vec<ToolOutput>,
    pub selected_qa: Vec<QaPair>,
    #[serde(default)]
    pub constraints: Constraints,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_request_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedRequest {
    pub request_id: String,
    pub sentences: Vec<Sentence>,
    pub truncated: bool,
    pub intent: Intent,
    pub constraints: Constraints,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_request_id: Option<String>,
}

impl Request {
    pub fn prepare(&self) -> Result<PreparedRequest> {
        if self.request_id.trim().is_empty() {
            return invalid("empty request_id");
        }
        if self.task != focused() {
            return invalid(format!("unsupported task {}", self.task));
        }
        self.constraints.validate()?;
        let intent = Intent::from_pairs(&self.selected_qa)?;
        let (sentences, truncated) = candidates(&self.tool_outputs)?;
        Ok(PreparedRequest {
            request_id: self.request_id.clone(),
            sentences,
            truncated,
            intent,
            constraints: self.constraints.clone(),
            previous_request_id: self.previous_request_id.clone(),
        })
    }
}

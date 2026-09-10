pub mod evidence;
pub mod input;
pub mod intent;
pub mod output;

pub use evidence::Evidence;
pub use input::{MAX_CANDIDATES, Sentence, ToolOutput, candidates, segment};
pub use intent::{Intent, QaPair};
pub use output::{Constraints, MAX_SENTENCES, sentence_count};

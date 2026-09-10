use ramanujan::core::{
    Constraints, Evidence, Intent, MAX_CANDIDATES, QaPair, ToolOutput, candidates, segment,
    sentence_count,
};

fn out(id: &str, content: &str) -> ToolOutput {
    ToolOutput {
        source_id: id.into(),
        content: content.into(),
    }
}

fn pieces(text: &str) -> Vec<&str> {
    segment(text)
        .into_iter()
        .map(|(s, e)| &text[s..e])
        .collect()
}

#[test]
fn segments_on_terminal_punctuation_and_newlines() {
    assert_eq!(
        pieces("Qwen supports multilingual tasks. Gemma runs locally!  Is it free?"),
        [
            "Qwen supports multilingual tasks.",
            "Gemma runs locally!",
            "Is it free?"
        ]
    );
    assert_eq!(
        pieces("- first item\n- second item\n\n"),
        ["- first item", "- second item"]
    );
    assert_eq!(
        pieces("He said \"done.\" Then left."),
        ["He said \"done.\"", "Then left."]
    );
}

#[test]
fn keeps_abbreviations_initials_and_decimals_together() {
    assert_eq!(
        pieces("Dr. Smith met J. Doe, e.g. at 3.5 GHz vs. 2.0 GHz. Done."),
        [
            "Dr. Smith met J. Doe, e.g. at 3.5 GHz vs. 2.0 GHz.",
            "Done."
        ]
    );
}

#[test]
fn candidate_spans_point_back_into_their_source() {
    let outputs = vec![
        out("tool-001", "Qwen supports multilingual tasks.  It is open."),
        out("tool-002", "Gemma supports\nlocal deployment."),
    ];
    let (sentences, truncated) = candidates(&outputs).unwrap();
    assert!(!truncated);
    assert_eq!(sentences.len(), 4);
    for (i, s) in sentences.iter().enumerate() {
        assert_eq!(s.index, i);
        let src = outputs.iter().find(|o| o.source_id == s.source_id).unwrap();
        assert_eq!(
            s.text,
            src.content[s.start..s.end]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    assert_eq!(sentences[2].source_id, "tool-002");
    assert_eq!(sentences[2].text, "Gemma supports");

    let e = Evidence::from_sentence(&sentences[1], Some(0.5));
    assert_eq!(e.source_id, "tool-001");
    assert_eq!(e.text_span, "It is open.");
    assert_eq!(e.sentence_index, Some(1));
}

#[test]
fn candidates_truncate_at_the_limit() {
    let text = (0..MAX_CANDIDATES + 10)
        .map(|i| format!("Fact number {i}."))
        .collect::<Vec<_>>()
        .join(" ");
    let (sentences, truncated) = candidates(&[out("a", &text)]).unwrap();
    assert!(truncated);
    assert_eq!(sentences.len(), MAX_CANDIDATES);
}

#[test]
fn rejects_invalid_tool_outputs() {
    assert!(candidates(&[]).is_err());
    assert!(candidates(&[out("", "x.")]).is_err());
    assert!(candidates(&[out("a", "x."), out("a", "y.")]).is_err());
    assert!(candidates(&[out("a", "   ")]).is_err());
    assert!(candidates(&[out("a", "... ---")]).is_err());
}

fn qa(id: Option<&str>, answer: &str, selected: bool) -> QaPair {
    QaPair {
        question_id: id.map(Into::into),
        question: "What deployment do you prefer?".into(),
        answer: answer.into(),
        selected,
    }
}

#[test]
fn intent_splits_selected_and_validates() {
    let intent = Intent::from_pairs(&[
        qa(Some("q-1"), "Self-hosted", true),
        qa(Some("q-2"), "", false),
    ])
    .unwrap();
    assert_eq!(intent.selected.len(), 1);
    assert_eq!(intent.unselected.len(), 1);
    assert!(Intent::from_pairs(&[qa(None, "x", false)]).is_err());
    assert!(Intent::from_pairs(&[qa(None, "", true)]).is_err());
    assert!(Intent::from_pairs(&[qa(Some("q"), "a", true), qa(Some("q"), "b", true)]).is_err());
}

#[test]
fn constraints_enforce_one_to_three_sentences() {
    let c = Constraints::default();
    assert!(c.validate().is_ok());
    assert!(c.accepts("Gemma fits. It is small."));
    assert!(!c.accepts("One. Two. Three. Four."));
    assert!(!c.accepts("One.\nTwo."));
    assert!(!c.accepts(""));
    assert!(
        Constraints {
            max_sentences: 4,
            ..c.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        Constraints {
            min_sentences: 0,
            ..c.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        Constraints {
            min_sentences: 3,
            max_sentences: 2,
            ..c
        }
        .validate()
        .is_err()
    );
    assert_eq!(
        sentence_count("Gemma is a suitable candidate for self-hosted deployment."),
        1
    );
}

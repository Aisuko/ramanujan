use ramanujan::core::Constraints;
use ramanujan::harness::{Metrics, Request, Response};
use ramanujan::model::ramanujan_base;

const REQUEST: &str = r#"{
  "request_id": "req-001",
  "task": "focused_summarisation",
  "tool_outputs": [
    { "source_id": "search-001", "content": "Qwen supports multilingual tasks. Gemma supports local deployment." },
    { "source_id": "search-002", "content": "Gemma can be self-hosted on one GPU." }
  ],
  "selected_qa": [
    { "question_id": "q-001", "question": "What deployment do you prefer?", "answer": "Self-hosted" }
  ],
  "constraints": { "min_sentences": 1, "max_sentences": 3, "style": "concise", "preserve_factuality": true }
}"#;

const RESPONSE: &str = r#"{
  "request_id": "req-001",
  "summary": "Gemma is a suitable candidate for self-hosted deployment.",
  "sentence_count": 1,
  "evidence": [ { "source_id": "search-001", "text_span": "Gemma supports local deployment." } ],
  "metrics": { "selected_evidence": 1, "input_tokens": 1200, "output_tokens": 12 }
}"#;

#[test]
fn roadmap_request_parses_and_prepares() {
    let req: Request = serde_json::from_str(REQUEST).unwrap();
    let back: Request = serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
    assert_eq!(back, req);
    let p = req.prepare().unwrap();
    assert_eq!(p.request_id, "req-001");
    assert_eq!(p.sentences.len(), 3);
    assert_eq!(p.sentences[2].source_id, "search-002");
    assert_eq!(p.intent.selected.len(), 1);
    assert!(p.intent.selected[0].selected);
    assert!(!p.truncated);
}

#[test]
fn request_defaults_and_refinement_link() {
    let req: Request = serde_json::from_str(
        r#"{"request_id":"req-002","previous_request_id":"req-001",
            "tool_outputs":[{"source_id":"s","content":"A fact."}],
            "selected_qa":[{"question":"Q?","answer":"A"}]}"#,
    )
    .unwrap();
    assert_eq!(req.task, "focused_summarisation");
    assert_eq!(req.constraints, Constraints::default());
    assert_eq!(
        req.prepare().unwrap().previous_request_id.as_deref(),
        Some("req-001")
    );
}

#[test]
fn request_rejects_bad_input() {
    let mut req: Request = serde_json::from_str(REQUEST).unwrap();
    req.selected_qa.clear();
    assert!(req.prepare().is_err());
    let mut req: Request = serde_json::from_str(REQUEST).unwrap();
    req.constraints.max_sentences = 5;
    assert!(req.prepare().is_err());
    let mut req: Request = serde_json::from_str(REQUEST).unwrap();
    req.task = "chat".into();
    assert!(req.prepare().is_err());
}

#[test]
fn roadmap_response_round_trips_and_validates() {
    let resp: Response = serde_json::from_str(RESPONSE).unwrap();
    let back: Response = serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();
    assert_eq!(back, resp);
    resp.validate(&Constraints::default()).unwrap();

    let built = Response::new(
        "req-001".into(),
        "One. Two. Three. Four.".into(),
        Vec::new(),
        Metrics::default(),
    );
    assert_eq!(built.sentence_count, 4);
    assert!(built.validate(&Constraints::default()).is_err());
}

#[test]
fn base_config_matches_the_roadmap() {
    let c = ramanujan_base();
    c.validate().unwrap();
    assert_eq!(
        (c.d_model, c.n_encoder_layer, c.n_decoder_layer, c.n_head),
        (384, 6, 4, 6)
    );
    assert_eq!(
        (c.d_ff, c.vocab_size, c.max_source, c.max_target),
        (1536, 32000, 2048, 96)
    );
}

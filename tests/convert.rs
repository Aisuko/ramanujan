use ramanujan::core::candidates;
use ramanujan::data::convert::{QUESTION, clean_transcript, cnndm_row, oracle, qmsum_meeting};
use ramanujan::data::{Dataset, convert_file, read_jsonl};

fn meeting() -> String {
    let turns: Vec<String> = (0..12)
        .map(|i| {
            let content = match i {
                5 => "We agreed the remote should use a rubber case. It costs more.".to_string(),
                6 => "The rubber case improves grip.".to_string(),
                9 => "Marketing wants a yellow colour scheme.".to_string(),
                _ => format!("Filler remark number {i}."),
            };
            format!(r#"{{"speaker":"Speaker {}","content":"{content}"}}"#, i % 3)
        })
        .collect();
    format!(
        r#"{{"topic_list":[],"general_query_list":[{{"query":"Summarise.","answer":"A meeting."}}],
        "specific_query_list":[
          {{"query":"What was decided about the case?","answer":"They chose a rubber case. It improves grip. It costs more. Extra sentence.","relevant_text_span":[["5","6"]]}},
          {{"query":"What colour did marketing want?","answer":"Yellow.","relevant_text_span":[["9","9"]]}},
          {{"query":"No span here?","answer":"Nothing.","relevant_text_span":[]}}
        ],
        "meeting_transcripts":[{}]}}"#,
        turns.join(",")
    )
}

#[test]
fn qmsum_meeting_builds_grounded_examples() {
    let (examples, skipped) = qmsum_meeting(&meeting(), "m0").unwrap();
    assert_eq!(examples.len(), 2);
    assert_eq!(skipped, 1);

    let e = &examples[0];
    e.validate().unwrap();
    assert_eq!(e.id, "qmsum-m0-0");
    assert_eq!(e.provenance.group.as_deref(), Some("m0"));
    assert_eq!(
        e.target_summary,
        "They chose a rubber case. It improves grip. It costs more."
    );
    assert_eq!(e.qa_pairs.len(), 3);
    assert!(e.qa_pairs[0].selected && !e.qa_pairs[1].selected);
    assert_eq!(e.qa_pairs[0].question, QUESTION);
    assert_eq!(e.qa_pairs[0].answer, "What was decided about the case?");

    let (sentences, _) = candidates(&e.tool_outputs).unwrap();
    let labelled: Vec<&str> = e
        .evidence_labels
        .iter()
        .map(|&i| sentences[i].text.as_str())
        .collect();
    assert_eq!(
        labelled,
        [
            "Speaker 2: We agreed the remote should use a rubber case.",
            "It costs more.",
            "Speaker 0: The rubber case improves grip."
        ]
    );
    assert!(
        e.tool_outputs
            .iter()
            .all(|o| o.source_id.starts_with("turns-"))
    );
    assert_eq!(e.tool_outputs[0].source_id, "turns-3-8");

    let e = &examples[1];
    e.validate().unwrap();
    assert!(e.qa_pairs[1].selected);
    assert_eq!(e.tool_outputs[0].source_id, "turns-7-11");
}

const CNN: &str = r#"{"id":"abc","article":"Gemma is a family of open models. It was released by Google. The weather was sunny that day. Gemma supports local deployment on laptops.","highlights":"Gemma is an open model family\nGemma runs on local laptops"}"#;

#[test]
fn cnndm_row_uses_highlights_and_overlap_oracle() {
    let e = cnndm_row(CNN, "fallback").unwrap().unwrap();
    e.validate().unwrap();
    assert_eq!(e.id, "cnndm-abc");
    assert_eq!(
        e.target_summary,
        "Gemma is an open model family. Gemma runs on local laptops."
    );
    assert_eq!(e.evidence_labels, vec![0, 3]);
    assert!(
        cnndm_row(r#"{"article":"","highlights":"x"}"#, "f")
            .unwrap()
            .is_none()
    );
}

#[test]
fn oracle_stops_when_nothing_new_is_covered() {
    let (s, _) = candidates(&[ramanujan::core::ToolOutput {
        source_id: "a".into(),
        content: "Cats purr loudly. Dogs bark loudly. Fish swim.".into(),
    }])
    .unwrap();
    assert_eq!(oracle(&s, "cats purr", 3), vec![0]);
    assert!(oracle(&s, "nothing matches", 3).is_empty());
}

#[test]
fn convert_file_writes_valid_jsonl() {
    let dir = std::env::temp_dir().join(format!("ramanujan_convert_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("qmsum_train.jsonl");
    std::fs::write(
        &src,
        format!(
            "{}\n{}\n",
            meeting().replace('\n', " "),
            meeting().replace('\n', " ")
        ),
    )
    .unwrap();
    let out = dir.join("out.jsonl");
    let stats = convert_file(Dataset::Qmsum, &src, &out, None).unwrap();
    assert_eq!((stats.read, stats.written, stats.skipped), (2, 4, 2));
    let back = read_jsonl(&out).unwrap();
    assert_eq!(back.len(), 4);
    assert_eq!(back[2].id, "qmsum-qmsum_train-1-0");

    let src = dir.join("cnn.jsonl");
    std::fs::write(&src, format!("{CNN}\n{CNN}\n")).unwrap();
    let stats = convert_file(Dataset::CnnDm, &src, &out, Some(1)).unwrap();
    assert_eq!(stats.written, 1);
    std::fs::remove_dir_all(&dir).ok();
    assert!(Dataset::parse("other").is_err());
}

#[test]
fn cleans_qmsum_transcript_artifacts() {
    assert_eq!(
        clean_transcript(
            "I {disfmarker} I 'll  talk {vocalsound} later , it 's fine . We do n't @ @ know ."
        ),
        "I I'll talk later, it's fine. We don't know."
    );
}

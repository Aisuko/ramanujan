use forge::Device;
use forge::serialization::fzm::{load_fzm_q4, save_fzm_q4};
use ramanujan::{Config, ReactModel};

#[test]
fn forward_and_fzm_roundtrip() {
    let device = Device::Cpu;
    let model = ReactModel::init_random(Config::react_6m(), &device, 42).unwrap();

    let ids: Vec<u32> = (0..32).map(|i| (i * 37) % 8192).collect();
    let hidden = model.encoder.forward(&ids).unwrap();
    assert_eq!(hidden.shape().dims(), &[32, 256]);
    let out = hidden.to_vec_f32().unwrap();
    assert!(out.iter().all(|v| v.is_finite()));

    let candidates: Vec<usize> = (1..32).collect();
    let logits = model.pointer.score(&hidden, 0, &candidates).unwrap();
    assert_eq!(logits.len(), candidates.len());
    assert!(logits.iter().all(|v| v.is_finite()));

    let path = std::env::temp_dir().join("react_6m_smoke.fzm");
    let entries = model.to_entries().unwrap();
    save_fzm_q4(&path, &entries).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let loaded = load_fzm_q4(&bytes).unwrap();
    let restored = ReactModel::from_entries(Config::react_6m(), &loaded, &device).unwrap();
    let hidden2 = restored.encoder.forward(&ids).unwrap();
    let out2 = hidden2.to_vec_f32().unwrap();
    let max_diff = out
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1.0, "max diff {max_diff}");
    std::fs::remove_file(&path).ok();
}

use forge::Device;
use ramanujan::train::train_batch_sanity;

#[test]
fn batched_training_reduces_loss() {
    let device = Device::Cpu;
    let (first, last) = train_batch_sanity(&device, 3, 8, 150, 7).unwrap();
    assert!(first.is_finite() && last.is_finite());
    assert!(last < first * 0.5, "first {first} last {last}");
}

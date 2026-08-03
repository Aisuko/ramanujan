use forge::{Device, Result};
use ramanujan::Config;
use ramanujan::train::{mask_token_id, train_mlm};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct Args {
    dataset: Option<String>,
    smoke: bool,
    batch_size: usize,
    seq_len: usize,
    epochs: usize,
    lr: f32,
    weight_decay: f32,
    mask_prob: f32,
    seed: u64,
    warmup: usize,
    max_steps: usize,
    clip: f32,
    ckpt_every: usize,
    ckpt_path: String,
    device: String,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            dataset: None,
            smoke: false,
            batch_size: 32,
            seq_len: 128,
            epochs: 1,
            lr: 3e-4,
            weight_decay: 0.01,
            mask_prob: 0.15,
            seed: 42,
            warmup: 1000,
            max_steps: 0,
            clip: 1.0,
            ckpt_every: 500,
            ckpt_path: "checkpoints/react_6m.fzm".into(),
            device: "gpu".into(),
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = argv
            .get(i + 1)
            .unwrap_or_else(|| panic!("{key} needs a value"));
        match key {
            "--dataset" => a.dataset = Some(val.clone()),
            "--smoke" => a.smoke = val == "yes",
            "--batch-size" => a.batch_size = val.parse().expect("--batch-size"),
            "--seq-len" => a.seq_len = val.parse().expect("--seq-len"),
            "--epochs" => a.epochs = val.parse().expect("--epochs"),
            "--lr" => a.lr = val.parse().expect("--lr"),
            "--weight-decay" => a.weight_decay = val.parse().expect("--weight-decay"),
            "--mask-prob" => a.mask_prob = val.parse().expect("--mask-prob"),
            "--seed" => a.seed = val.parse().expect("--seed"),
            "--warmup" => a.warmup = val.parse().expect("--warmup"),
            "--max-steps" => a.max_steps = val.parse().expect("--max-steps"),
            "--clip" => a.clip = val.parse().expect("--clip"),
            "--ckpt-every" => a.ckpt_every = val.parse().expect("--ckpt-every"),
            "--ckpt-path" => a.ckpt_path = val.clone(),
            "--device" => a.device = val.clone(),
            other => panic!("unknown arg {other}"),
        }
        i += 2;
    }
    a
}

fn load_dataset(path: &str, vocab_size: usize) -> Vec<u32> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let len = file.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    let n = len / 4;
    let mut ids: Vec<u32> = vec![0u32; n];
    let buf = unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, n * 4) };
    file.read_exact(buf)
        .unwrap_or_else(|e| panic!("reading {path}: {e}"));
    if !cfg!(target_endian = "little") {
        for id in ids.iter_mut() {
            *id = id.to_le();
        }
    }
    for &id in &ids {
        assert!(
            (id as usize) < vocab_size,
            "token id {id} >= vocab_size {vocab_size}"
        );
    }
    ids
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "gpu" => Device::wgpu()?,
        other => panic!("unknown --device {other}, want cpu|gpu"),
    };
    let config = Config::react_6m();

    if args.smoke {
        let batch = 4;
        let seq = 16;
        let epochs = 200;
        let mut rng = StdRng::seed_from_u64(args.seed);
        let n = batch * seq; // one fixed block, repeated every epoch
        let mask_id = mask_token_id(&config);
        let dataset: Vec<u32> = (0..n).map(|_| rng.random_range(0..mask_id)).collect();
        let ckpt_path = std::env::temp_dir()
            .join("ramanujan_smoke")
            .join("ckpt.fzm")
            .to_string_lossy()
            .into_owned();
        let (first, last, _) = train_mlm(
            &device,
            config,
            &dataset,
            batch,
            seq,
            epochs,
            1e-2,
            0.0,
            0.5,
            args.seed,
            20,
            0,
            args.clip,
            epochs,
            &ckpt_path,
            |step, loss| println!("smoke step {step} loss {loss:.4}"),
        )?;
        if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
            for e in entries.flatten() {
                if e.file_name()
                    .to_string_lossy()
                    .starts_with("ramanujan_smoke")
                {
                    std::fs::remove_dir_all(e.path()).ok();
                }
            }
        }
        if last.is_finite() && last < first {
            println!("SMOKE OK: loss {first:.4} -> {last:.4} on {}", args.device);
            Ok(())
        } else {
            eprintln!("SMOKE FAILED: loss {first:.4} -> {last:.4}");
            std::process::exit(1);
        }
    } else {
        let dataset_path = args
            .dataset
            .expect("--dataset is required unless --smoke yes");
        let dataset = load_dataset(&dataset_path, config.vocab_size);
        let (first, last, ckpt) = train_mlm(
            &device,
            config,
            &dataset,
            args.batch_size,
            args.seq_len,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.mask_prob,
            args.seed,
            args.warmup,
            args.max_steps,
            args.clip,
            args.ckpt_every,
            &args.ckpt_path,
            |step, loss| println!("step {step} loss {loss:.4}"),
        )?;
        println!("done: loss {first:.4} -> {last:.4}, checkpoint at {ckpt}");
        Ok(())
    }
}

use forge::serialization::fzm::load_fzm_q4_file;
use forge::{Device, Gpt2Tokenizer, Result};
use ramanujan::train::mask_token_id;
use ramanujan::{Config, Encoder};
use std::io::{BufRead, Write};

struct Args {
    ckpt: String,
    tokenizer: String,
    device: String,
    topk: usize,
    gen_len: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        ckpt: "checkpoints/react_6m.fzm".into(),
        tokenizer: "datasets/tokenizer".into(),
        device: "gpu".into(),
        topk: 5,
        gen_len: 16,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = argv
            .get(i + 1)
            .unwrap_or_else(|| panic!("{key} needs a value"));
        match key {
            "--ckpt" => a.ckpt = val.clone(),
            "--tokenizer" => a.tokenizer = val.clone(),
            "--device" => a.device = val.clone(),
            "--topk" => a.topk = val.parse().expect("--topk"),
            "--gen" => a.gen_len = val.parse().expect("--gen"),
            other => panic!("unknown arg {other}"),
        }
        i += 2;
    }
    a
}

/// Row-wise softmax over the `[t, vocab]` logits, as a flat host vector.
fn probs(enc: &Encoder, ids: &[u32]) -> Result<(Vec<f32>, usize)> {
    let logits = enc.logits(ids)?;
    let vocab = logits.shape().dims()[1];
    let p = forge::ops::softmax(&logits, false, 0)?.to_vec_f32()?;
    Ok((p, vocab))
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "gpu" => Device::wgpu()?,
        other => panic!("unknown --device {other}, want cpu|gpu"),
    };
    let config = Config::react_6m();
    let mask_id = mask_token_id(&config);
    let enc = Encoder::from_entries(config, &load_fzm_q4_file(&args.ckpt)?, &device)?;
    let tok = Gpt2Tokenizer::from_dir(&args.tokenizer)?;

    println!("checkpoint {}", args.ckpt);
    println!(
        "masked LM: write [MASK] where you want a prediction, or plain text to continue it. ctrl-d to quit."
    );

    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(());
        }
        let line = line.trim_end_matches('\n');
        if line.trim().is_empty() {
            continue;
        }

        let explicit = line.contains("[MASK]");
        let mut ids: Vec<u32> = Vec::new();
        if explicit {
            for (i, piece) in line.split("[MASK]").enumerate() {
                if i > 0 {
                    ids.push(mask_id);
                }
                ids.extend(tok.encode(piece)?);
            }
        } else {
            ids.extend(tok.encode(line)?);
            ids.extend(std::iter::repeat_n(mask_id, args.gen_len));
        }
        ids.truncate(config.max_seq);
        if ids.is_empty() {
            continue;
        }

        // One pass per mask: fill the most confident position first, so later
        // masks are predicted with the earlier ones already resolved.
        loop {
            let masked: Vec<usize> = ids
                .iter()
                .enumerate()
                .filter(|&(_, &t)| t == mask_id)
                .map(|(i, _)| i)
                .collect();
            if masked.is_empty() {
                break;
            }
            let (p, vocab) = probs(&enc, &ids)?;
            let best = |pos: usize| -> (u32, f32) {
                let row = &p[pos * vocab..pos * vocab + vocab - 1];
                let (id, pr) = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                (id as u32, *pr)
            };
            let (pos, id, _) = masked
                .iter()
                .map(|&i| {
                    let (id, pr) = best(i);
                    (i, id, pr)
                })
                .max_by(|a, b| a.2.total_cmp(&b.2))
                .unwrap();
            if explicit {
                let row = &p[pos * vocab..pos * vocab + vocab - 1];
                let mut order: Vec<usize> = (0..row.len()).collect();
                order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                let top: Vec<String> = order
                    .iter()
                    .take(args.topk)
                    .map(|&i| format!("{:?} {:.3}", tok.decode(&[i as u32]), row[i]))
                    .collect();
                println!("  [{pos}] {}", top.join("  "));
            }
            ids[pos] = id;
        }
        println!("{}", tok.decode(&ids));
    }
}

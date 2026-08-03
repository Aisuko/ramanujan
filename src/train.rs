use crate::{Config, Encoder};
use forge::autograd::{TVar, Tape};
use forge::ops::MatmulSpec;
use forge::optim::{AdamW, AdamWOpts};
use forge::serialization::fzm::save_fzm_q4;
use forge::{Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

impl Encoder {
    fn leaves(&self, tape: &mut Tape) -> Vec<TVar> {
        let mut v = vec![
            tape.leaf(self.wte.clone()),
            tape.leaf(self.wte_proj.w.clone()),
            tape.leaf(self.wte_proj.b.clone().unwrap()),
        ];
        for b in &self.blocks {
            v.push(tape.leaf(b.ln_1.gamma.clone()));
            v.push(tape.leaf(b.ln_1.beta.clone()));
            v.push(tape.leaf(b.qkv.w.clone()));
            v.push(tape.leaf(b.qkv.b.clone().unwrap()));
            v.push(tape.leaf(b.attn_proj.w.clone()));
            v.push(tape.leaf(b.attn_proj.b.clone().unwrap()));
            v.push(tape.leaf(b.ln_2.gamma.clone()));
            v.push(tape.leaf(b.ln_2.beta.clone()));
            v.push(tape.leaf(b.fc.w.clone()));
            v.push(tape.leaf(b.fc.b.clone().unwrap()));
            v.push(tape.leaf(b.fc_proj.w.clone()));
            v.push(tape.leaf(b.fc_proj.b.clone().unwrap()));
        }
        v.push(tape.leaf(self.ln_f.gamma.clone()));
        v.push(tape.leaf(self.ln_f.beta.clone()));
        v.push(tape.leaf(self.head_b.clone()));
        v
    }

    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v = vec![
            &mut self.wte,
            &mut self.wte_proj.w,
            self.wte_proj.b.as_mut().unwrap(),
        ];
        for b in &mut self.blocks {
            v.push(&mut b.ln_1.gamma);
            v.push(&mut b.ln_1.beta);
            v.push(&mut b.qkv.w);
            v.push(b.qkv.b.as_mut().unwrap());
            v.push(&mut b.attn_proj.w);
            v.push(b.attn_proj.b.as_mut().unwrap());
            v.push(&mut b.ln_2.gamma);
            v.push(&mut b.ln_2.beta);
            v.push(&mut b.fc.w);
            v.push(b.fc.b.as_mut().unwrap());
            v.push(&mut b.fc_proj.w);
            v.push(b.fc_proj.b.as_mut().unwrap());
        }
        v.push(&mut self.ln_f.gamma);
        v.push(&mut self.ln_f.beta);
        v.push(&mut self.head_b);
        v
    }
}

fn block_tape(
    tape: &mut Tape,
    x: &TVar,
    leaves: &[TVar],
    idx: usize,
    n_head: usize,
    batch: usize,
) -> Result<TVar> {
    let (ln1g, ln1b, qkvw, qkvb, apw, apb, ln2g, ln2b, fcw, fcb, fpw, fpb) = (
        &leaves[idx],
        &leaves[idx + 1],
        &leaves[idx + 2],
        &leaves[idx + 3],
        &leaves[idx + 4],
        &leaves[idx + 5],
        &leaves[idx + 6],
        &leaves[idx + 7],
        &leaves[idx + 8],
        &leaves[idx + 9],
        &leaves[idx + 10],
        &leaves[idx + 11],
    );
    let h = tape.layernorm(x, ln1g, ln1b, 1e-5)?;
    let qkv = tape.matmul(&h, qkvw, Some(qkvb), MatmulSpec::default())?;
    let (q, k, v) = tape.split_heads_batched(&qkv, n_head, batch)?;
    let hd = q.t.shape().dims()[2];
    let att = tape.matmul(
        &q,
        &k,
        None,
        MatmulSpec {
            trans_b: true,
            alpha: 1.0 / (hd as f32).sqrt(),
            ..Default::default()
        },
    )?;
    let att = tape.softmax(&att, false, 0)?;
    let y = tape.matmul(&att, &v, None, MatmulSpec::default())?;
    let y = tape.merge_heads_batched(&y, batch)?;
    let y = tape.matmul(&y, apw, Some(apb), MatmulSpec::default())?;
    let x = tape.add(x, &y)?;
    let h2 = tape.layernorm(&x, ln2g, ln2b, 1e-5)?;
    let f = tape.matmul(&h2, fcw, Some(fcb), MatmulSpec::default())?;
    let f = tape.gelu(&f)?;
    let f = tape.matmul(&f, fpw, Some(fpb), MatmulSpec::default())?;
    tape.add(&x, &f)
}

/// Per-step constants that do not depend on the batch's token ids: the tiled
/// sinusoidal position table, the fixed `emb_ln` affine, and the zero `wpe`.
/// Built once and reused so no step re-derives or re-uploads them.
struct Consts {
    pos: Tensor,
    ln_g: Tensor,
    ln_b: Tensor,
    wpe_zero: Tensor,
}

impl Consts {
    fn new(device: &Device, config: &Config, batch: usize, seq: usize) -> Result<Consts> {
        let (d, bt) = (config.d_model, batch * seq);
        let mut pos = Vec::with_capacity(bt * d);
        for _ in 0..batch {
            pos.extend_from_slice(&crate::sinusoidal(seq, d));
        }
        Ok(Consts {
            pos: Tensor::from_f32(&pos, [bt, d], device)?,
            ln_g: Tensor::from_f32(&vec![1.0f32; d], [d], device)?,
            ln_b: Tensor::zeros([d], device)?,
            wpe_zero: Tensor::zeros([seq, config.embed_rank], device)?,
        })
    }
}

/// [`Encoder::forward`], tape-recorded over `batch` stacked sequences of
/// `ids.len() / batch` tokens each. Sinusoidal positions have no gradient of
/// their own, so they ride through `Tape::embedding_batched` as a fixed,
/// all-zero `wpe` and get added back at `d_model` width afterward — matching
/// where [`Encoder::forward`] adds them.
fn forward_batched(
    tape: &mut Tape,
    leaves: &[TVar],
    config: &Config,
    ids_t: &Tensor,
    batch: usize,
    c: &Consts,
) -> Result<TVar> {
    let wpe_zero = tape.leaf(c.wpe_zero.clone());
    let mut x = tape.embedding_batched(ids_t, &leaves[0], &wpe_zero, 0, batch)?;
    x = tape.matmul(&x, &leaves[1], Some(&leaves[2]), MatmulSpec::default())?;
    let g = tape.leaf(c.ln_g.clone());
    let b = tape.leaf(c.ln_b.clone());
    x = tape.layernorm(&x, &g, &b, 1e-5)?;
    let pos = tape.leaf(c.pos.clone());
    x = tape.add(&x, &pos)?;
    let mut idx = 3;
    for _ in 0..config.n_layer {
        x = block_tape(tape, &x, leaves, idx, config.n_head, batch)?;
        idx += 12;
    }
    tape.layernorm(&x, &leaves[idx], &leaves[idx + 1], 1e-5)
}

/// Trains a fresh `React-6M` encoder for `steps` batched steps against a
/// fixed random target, folding `batch` sequences of `seq` tokens into the
/// same leading axis forge's attention heads use — see
/// `split_heads_batched`/`merge_heads_batched`. A well-posed loss (not
/// trivially minimized by zero weights) that decreases end to end is
/// evidence the forward/backward wiring has no bugs. Returns (loss at step
/// 0, loss at the last step).
pub fn train_batch_sanity(
    device: &Device,
    batch: usize,
    seq: usize,
    steps: usize,
    seed: u64,
) -> Result<(f32, f32)> {
    let mut enc = Encoder::init_random(Config::react_6m(), device, seed)?;
    let config = enc.config;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA5A5_A5A5);
    let bt = batch * seq;
    let target: Vec<f32> = (0..bt * config.d_model)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();

    // `head_b` is only on the MLM logits path, not on `hidden` — this sanity
    // loss never reaches it, so it is left out of the step entirely.
    let n_params = 3 + config.n_layer * 12 + 2;
    let mut opt = {
        let mut tape = Tape::new();
        let leaves = enc.leaves(&mut tape);
        let params: Vec<(&Tensor, bool)> =
            leaves[..n_params].iter().map(|l| (&l.t, false)).collect();
        AdamW::new(
            &params,
            AdamWOpts {
                lr: 1e-2,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?
    };
    debug_assert_eq!(n_params + 1, {
        let mut tape = Tape::new();
        enc.leaves(&mut tape).len()
    });

    let consts = Consts::new(device, &config, batch, seq)?;
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    for step in 0..steps {
        let ids: Vec<u32> = (0..bt)
            .map(|_| rng.random_range(0..config.vocab_size as u32))
            .collect();
        let ids_t = Tensor::from_u32(&ids, [bt], device)?;
        let mut tape = Tape::new();
        let leaves = enc.leaves(&mut tape);
        let hidden = forward_batched(&mut tape, &leaves, &config, &ids_t, batch, &consts)?;

        let h = hidden.t.to_vec_f32()?;
        let n = h.len() as f32;
        let diff: Vec<f32> = h.iter().zip(&target).map(|(a, b)| a - b).collect();
        let loss = diff.iter().map(|d| d * d).sum::<f32>() / n;
        let grad: Vec<f32> = diff.iter().map(|d| 2.0 * d / n).collect();
        let seed_grad = Tensor::from_f32(&grad, hidden.t.shape().clone(), device)?;

        let all = tape.backward(&hidden, seed_grad)?;
        let grads: Vec<Tensor> = leaves[..n_params]
            .iter()
            .map(|l| all[l.id].clone().expect("param received no gradient"))
            .collect();
        let mut params = enc.params_mut();
        params.truncate(n_params);
        opt.step(&mut params, &grads)?;

        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;
    }
    Ok((first_loss, last_loss))
}

/// Last vocab id, reserved as the `[MASK]` token — no tokenizer section
/// exists yet to declare special ids separately.
pub fn mask_token_id(config: &Config) -> u32 {
    config.vocab_size as u32 - 1
}

#[allow(clippy::too_many_arguments)]
pub fn train_mlm(
    device: &Device,
    config: Config,
    dataset: &[u32],
    batch: usize,
    seq: usize,
    epochs: usize,
    lr: f32,
    weight_decay: f32,
    mask_prob: f32,
    seed: u64,
    warmup: usize,
    max_steps: usize,
    clip: f32,
    ckpt_every: usize,
    ckpt_path: &str,
    mut on_step: impl FnMut(usize, f32),
) -> Result<(f32, f32, String)> {
    let bt = batch * seq;
    let n_blocks = dataset.len() / bt;
    if n_blocks == 0 {
        return Err(forge::ForgeError::Shape(format!(
            "dataset has {} tokens, need at least batch*seq={bt}",
            dataset.len()
        )));
    }
    let mask_id = mask_token_id(&config);
    let mut enc = Encoder::init_random(config, device, seed)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED_5EED);

    let mut opt = {
        let mut tape = Tape::new();
        let leaves = enc.leaves(&mut tape);
        let params: Vec<(&Tensor, bool)> = leaves.iter().map(|l| (&l.t, false)).collect();
        AdamW::new(
            &params,
            AdamWOpts {
                lr,
                weight_decay,
                clip: (clip > 0.0).then_some(clip),
                ..Default::default()
            },
        )?
    };

    // Linear warmup then cosine decay to zero. `max_steps` is what the schedule
    // is stretched over when the run is budgeted by wall clock rather than by
    // exhausting `epochs` — otherwise the cosine would barely move.
    let total = if max_steps > 0 {
        max_steps
    } else {
        epochs * n_blocks
    };
    let lr_at = |s: usize| -> f32 {
        if s < warmup {
            lr * (s + 1) as f32 / warmup as f32
        } else {
            let p = (s - warmup) as f32 / total.saturating_sub(warmup).max(1) as f32;
            0.5 * lr * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos())
        }
    };

    let consts = Consts::new(device, &config, batch, seq)?;
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    let mut step = 0usize;
    for epoch in 1..=epochs {
        let mut order: Vec<usize> = (0..n_blocks).collect();
        order.shuffle(&mut rng);
        for block_idx in order {
            let block = &dataset[block_idx * bt..(block_idx + 1) * bt];
            let mut masked_ids = block.to_vec();
            let mut sel: Vec<u32> = Vec::with_capacity((bt as f32 * mask_prob) as usize + 1);
            for (i, id) in masked_ids.iter_mut().enumerate() {
                if rng.random::<f32>() < mask_prob {
                    *id = mask_id;
                    sel.push(i as u32);
                }
            }
            if sel.is_empty() {
                let i = rng.random_range(0..bt);
                masked_ids[i] = mask_id;
                sel.push(i as u32);
            }
            let m = sel.len();
            let targets: Vec<u32> = sel.iter().map(|&i| block[i as usize]).collect();

            let mut tape = Tape::new();
            let leaves = enc.leaves(&mut tape);
            let ids_t = Tensor::from_u32(&masked_ids, [bt], device)?;
            let hidden = forward_batched(&mut tape, &leaves, &config, &ids_t, batch, &consts)?;
            // The head runs on the masked rows only. Gathering them with an
            // embedding whose "table" is `hidden` makes backward a scatter-add
            // into those rows, so unmasked positions get an exact zero gradient
            // without the host round-trip a `[bt, vocab]` mask would need — and
            // the logits/softmax cost drops by 1/mask_prob.
            let sel_t = Tensor::from_u32(&sel, [m], device)?;
            let wpe_zero = tape.leaf(Tensor::zeros([m, config.d_model], device)?);
            let hsel = tape.embedding(&sel_t, &hidden, &wpe_zero, 0)?;
            let proj = tape.matmul(
                &hsel,
                &leaves[1],
                None,
                MatmulSpec {
                    trans_b: true,
                    ..Default::default()
                },
            )?;
            let logits = tape.matmul(
                &proj,
                &leaves[0],
                Some(&leaves[leaves.len() - 1]),
                MatmulSpec {
                    trans_b: true,
                    ..Default::default()
                },
            )?;

            let probs = forge::ops::softmax(&logits.t, false, 0)?;
            let targets_t = Tensor::from_u32(&targets, [m], device)?;
            let nll = forge::ops::gather_nll(&probs, &targets_t)?.to_vec_f32()?;
            let loss: f32 = nll.iter().sum::<f32>() / m as f32;
            let dlogits = forge::ops::ce_bwd(&probs, &targets_t, 1.0 / m as f32)?;

            let all = tape.backward(&logits, dlogits)?;
            let grads: Vec<Tensor> = leaves
                .iter()
                .map(|l| all[l.id].clone().expect("param received no gradient"))
                .collect();
            opt.opts.lr = lr_at(step);
            opt.step(&mut enc.params_mut(), &grads)?;

            if step == 0 {
                first_loss = loss;
            }
            last_loss = loss;
            step += 1;
            on_step(step, loss);

            if ckpt_every > 0 && step.is_multiple_of(ckpt_every) {
                save_checkpoint(&enc, ckpt_path, epoch, step)?;
            }
            if max_steps > 0 && step >= max_steps {
                let ckpt = save_checkpoint(&enc, ckpt_path, epoch, step)?;
                return Ok((first_loss, last_loss, ckpt));
            }
        }
    }
    let ckpt = save_checkpoint(&enc, ckpt_path, epochs, step)?;
    Ok((first_loss, last_loss, ckpt))
}

/// `dir/name.fzm` -> `dir_epoch{e}_step{s}/name.fzm`, so every checkpoint keeps
/// its own directory instead of overwriting the previous one.
fn checkpoint_dest(path: &str, epoch: usize, step: usize) -> String {
    let p = std::path::Path::new(path);
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "ckpt.fzm".into());
    let dir = match p.parent().map(|d| d.to_string_lossy().into_owned()) {
        Some(d) if !d.is_empty() => d,
        _ => "checkpoints".into(),
    };
    format!("{dir}_epoch{epoch}_step{step}/{name}")
}

fn save_checkpoint(enc: &Encoder, path: &str, epoch: usize, step: usize) -> Result<String> {
    let dest = checkpoint_dest(path, epoch, step);
    if let Some(parent) = std::path::Path::new(&dest).parent() {
        std::fs::create_dir_all(parent).map_err(|e| forge::ForgeError::Fzm(e.to_string()))?;
    }
    save_fzm_q4(&dest, &enc.to_entries()?)?;
    Ok(dest)
}

use forge::nn::{LayerNorm, Linear};
use forge::ops::{self, MatmulSpec};
use forge::serialization::fzm::TensorEntries;
use forge::{Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub mod train;

#[derive(Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub embed_rank: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub max_seq: usize,
}

impl Config {
    pub fn react_6m() -> Self {
        Config {
            vocab_size: 8192,
            d_model: 256,
            embed_rank: 128,
            n_layer: 6,
            n_head: 4,
            d_ff: 1024,
            max_seq: 256,
        }
    }
}

struct Block {
    ln_1: LayerNorm,
    qkv: Linear,
    attn_proj: Linear,
    ln_2: LayerNorm,
    fc: Linear,
    fc_proj: Linear,
}

impl Block {
    fn forward(&self, x: &Tensor, n_head: usize) -> Result<Tensor> {
        let h = self.ln_1.forward(x)?;
        let qkv = self.qkv.forward(&h)?;
        let (q, k, v) = ops::split_heads(&qkv, n_head)?;
        let hd = q.shape().dims()[2];
        let att = ops::matmul(
            &q,
            &k,
            None,
            MatmulSpec {
                trans_b: true,
                alpha: 1.0 / (hd as f32).sqrt(),
                ..Default::default()
            },
        )?;
        let att = ops::softmax(&att, false, 0)?;
        let y = ops::matmul(&att, &v, None, MatmulSpec::default())?;
        let y = ops::merge_heads(&y)?;
        let y = self.attn_proj.forward(&y)?;
        let x = ops::add(x, &y)?;
        let h2 = self.ln_2.forward(&x)?;
        let f = ops::gelu(&self.fc.forward(&h2)?)?;
        let f = self.fc_proj.forward(&f)?;
        ops::add(&x, &f)
    }
}

pub struct Encoder {
    wte: Tensor,
    wte_proj: Linear,
    emb_ln: LayerNorm,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head_b: Tensor,
    config: Config,
}

fn sinusoidal(t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * d];
    for pos in 0..t {
        for i in 0..d / 2 {
            let freq = 1.0 / 10000f32.powf(2.0 * i as f32 / d as f32);
            out[pos * d + 2 * i] = (pos as f32 * freq).sin();
            out[pos * d + 2 * i + 1] = (pos as f32 * freq).cos();
        }
    }
    out
}

impl Encoder {
    pub fn init_random(config: Config, device: &Device, seed: u64) -> Result<Encoder> {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = 0.02f32;
        let d = config.d_model;
        let r = config.embed_rank;
        let resid_std = std / (2.0 * config.n_layer as f32).sqrt();
        let mut normal = |n: usize, std: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let u1: f32 = rng.random::<f32>().max(1e-7);
                    let u2: f32 = rng.random::<f32>();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
                })
                .collect()
        };
        let ones = |n: usize| Tensor::from_f32(&vec![1.0f32; n], [n], device);
        let zeros = |n: usize| Tensor::zeros([n], device);
        // The tied head makes `wte`/`wte_proj.w` set the logit scale as well as
        // the embedding scale; 1/sqrt(embed_rank) on both puts initial logits at
        // rms ~1 and leaves `emb_ln` with only the sqrt(d_model) gap to close.
        let emb_std = 1.0 / (r as f32).sqrt();
        let wte = Tensor::from_f32(
            &normal(config.vocab_size * r, emb_std),
            [config.vocab_size, r],
            device,
        )?;
        let wte_proj = Linear {
            w: Tensor::from_f32(&normal(r * d, emb_std), [r, d], device)?,
            b: Some(zeros(d)?),
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(Block {
                ln_1: LayerNorm {
                    gamma: ones(d)?,
                    beta: zeros(d)?,
                    eps: 1e-5,
                },
                qkv: Linear {
                    w: Tensor::from_f32(&normal(d * 3 * d, std), [d, 3 * d], device)?,
                    b: Some(zeros(3 * d)?),
                },
                attn_proj: Linear {
                    w: Tensor::from_f32(&normal(d * d, resid_std), [d, d], device)?,
                    b: Some(zeros(d)?),
                },
                ln_2: LayerNorm {
                    gamma: ones(d)?,
                    beta: zeros(d)?,
                    eps: 1e-5,
                },
                fc: Linear {
                    w: Tensor::from_f32(&normal(d * config.d_ff, std), [d, config.d_ff], device)?,
                    b: Some(zeros(config.d_ff)?),
                },
                fc_proj: Linear {
                    w: Tensor::from_f32(
                        &normal(config.d_ff * d, resid_std),
                        [config.d_ff, d],
                        device,
                    )?,
                    b: Some(zeros(d)?),
                },
            });
        }
        let ln_f = LayerNorm {
            gamma: ones(d)?,
            beta: zeros(d)?,
            eps: 1e-5,
        };
        let emb_ln = LayerNorm {
            gamma: ones(d)?,
            beta: zeros(d)?,
            eps: 1e-5,
        };
        let head_b = zeros(config.vocab_size)?;
        Ok(Encoder {
            wte,
            wte_proj,
            emb_ln,
            blocks,
            ln_f,
            head_b,
            config,
        })
    }

    pub fn to_entries(&self) -> Result<TensorEntries> {
        let mut out = Vec::new();
        out.push((
            "wte".into(),
            self.wte.shape().dims().to_vec(),
            self.wte.to_vec_f32()?,
        ));
        out.push((
            "wte_proj.w".into(),
            self.wte_proj.w.shape().dims().to_vec(),
            self.wte_proj.w.to_vec_f32()?,
        ));
        out.push((
            "wte_proj.b".into(),
            self.wte_proj.b.as_ref().unwrap().shape().dims().to_vec(),
            self.wte_proj.b.as_ref().unwrap().to_vec_f32()?,
        ));
        for (i, b) in self.blocks.iter().enumerate() {
            let push = |name: &str, t: &Tensor, out: &mut TensorEntries| -> Result<()> {
                out.push((
                    format!("h.{i}.{name}"),
                    t.shape().dims().to_vec(),
                    t.to_vec_f32()?,
                ));
                Ok(())
            };
            push("ln_1.gamma", &b.ln_1.gamma, &mut out)?;
            push("ln_1.beta", &b.ln_1.beta, &mut out)?;
            push("qkv.w", &b.qkv.w, &mut out)?;
            push("qkv.b", b.qkv.b.as_ref().unwrap(), &mut out)?;
            push("attn_proj.w", &b.attn_proj.w, &mut out)?;
            push("attn_proj.b", b.attn_proj.b.as_ref().unwrap(), &mut out)?;
            push("ln_2.gamma", &b.ln_2.gamma, &mut out)?;
            push("ln_2.beta", &b.ln_2.beta, &mut out)?;
            push("fc.w", &b.fc.w, &mut out)?;
            push("fc.b", b.fc.b.as_ref().unwrap(), &mut out)?;
            push("fc_proj.w", &b.fc_proj.w, &mut out)?;
            push("fc_proj.b", b.fc_proj.b.as_ref().unwrap(), &mut out)?;
        }
        out.push((
            "ln_f.gamma".into(),
            self.ln_f.gamma.shape().dims().to_vec(),
            self.ln_f.gamma.to_vec_f32()?,
        ));
        out.push((
            "ln_f.beta".into(),
            self.ln_f.beta.shape().dims().to_vec(),
            self.ln_f.beta.to_vec_f32()?,
        ));
        out.push((
            "head_b".into(),
            self.head_b.shape().dims().to_vec(),
            self.head_b.to_vec_f32()?,
        ));
        Ok(out)
    }

    pub fn from_entries(
        config: Config,
        entries: &TensorEntries,
        device: &Device,
    ) -> Result<Encoder> {
        let find = |name: &str| -> Result<Tensor> {
            let (_, shape, data) = entries
                .iter()
                .find(|(n, _, _)| n == name)
                .ok_or_else(|| forge::ForgeError::Fzm(format!("missing tensor {name}")))?;
            Tensor::from_f32(data, shape.clone(), device)
        };
        let eps = 1e-5;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block {
                ln_1: LayerNorm {
                    gamma: find(&format!("h.{i}.ln_1.gamma"))?,
                    beta: find(&format!("h.{i}.ln_1.beta"))?,
                    eps,
                },
                qkv: Linear {
                    w: find(&format!("h.{i}.qkv.w"))?,
                    b: Some(find(&format!("h.{i}.qkv.b"))?),
                },
                attn_proj: Linear {
                    w: find(&format!("h.{i}.attn_proj.w"))?,
                    b: Some(find(&format!("h.{i}.attn_proj.b"))?),
                },
                ln_2: LayerNorm {
                    gamma: find(&format!("h.{i}.ln_2.gamma"))?,
                    beta: find(&format!("h.{i}.ln_2.beta"))?,
                    eps,
                },
                fc: Linear {
                    w: find(&format!("h.{i}.fc.w"))?,
                    b: Some(find(&format!("h.{i}.fc.b"))?),
                },
                fc_proj: Linear {
                    w: find(&format!("h.{i}.fc_proj.w"))?,
                    b: Some(find(&format!("h.{i}.fc_proj.b"))?),
                },
            });
        }
        let wte = find("wte")?;
        let wte_proj = Linear {
            w: find("wte_proj.w")?,
            b: Some(find("wte_proj.b")?),
        };
        let ln_f = LayerNorm {
            gamma: find("ln_f.gamma")?,
            beta: find("ln_f.beta")?,
            eps,
        };
        let d = config.d_model;
        let emb_ln = LayerNorm {
            gamma: Tensor::from_f32(&vec![1.0f32; d], [d], device)?,
            beta: Tensor::zeros([d], device)?,
            eps,
        };
        let head_b = find("head_b")?;
        Ok(Encoder {
            wte,
            wte_proj,
            emb_ln,
            blocks,
            ln_f,
            head_b,
            config,
        })
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let device = self.wte.device();
        let t = ids.len();
        let ids_t = Tensor::from_u32(ids, [t], &device)?;
        let emb = ops::embedding(&ids_t, &self.wte, None, 0)?;
        let x = self.emb_ln.forward(&self.wte_proj.forward(&emb)?)?;
        let pos = Tensor::from_f32(
            &sinusoidal(t, self.config.d_model),
            [t, self.config.d_model],
            &device,
        )?;
        let mut x = ops::add(&x, &pos)?;
        for block in &self.blocks {
            x = block.forward(&x, self.config.n_head)?;
        }
        self.ln_f.forward(&x)
    }

    /// `[t, vocab_size]` MLM logits through the tied embedding, matching the
    /// head `train_mlm` optimises.
    pub fn logits(&self, ids: &[u32]) -> Result<Tensor> {
        let hidden = self.forward(ids)?;
        let proj = ops::matmul(
            &hidden,
            &self.wte_proj.w,
            None,
            MatmulSpec {
                trans_b: true,
                ..Default::default()
            },
        )?;
        ops::matmul(
            &proj,
            &self.wte,
            Some(&self.head_b),
            MatmulSpec {
                trans_b: true,
                ..Default::default()
            },
        )
    }
}

pub struct PointerHead {
    w: Tensor,
}

impl PointerHead {
    pub fn init_random(d_model: usize, device: &Device, seed: u64) -> Result<PointerHead> {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..d_model * d_model)
            .map(|_| {
                let u1: f32 = rng.random::<f32>().max(1e-7);
                let u2: f32 = rng.random::<f32>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.02
            })
            .collect();
        Ok(PointerHead {
            w: Tensor::from_f32(&data, [d_model, d_model], device)?,
        })
    }

    pub fn score(&self, hidden: &Tensor, sel: usize, candidates: &[usize]) -> Result<Vec<f32>> {
        let d = hidden.shape().dims()[1];
        let device = hidden.device();
        let h = hidden.to_vec_f32()?;
        let sel_t = Tensor::from_f32(&h[sel * d..(sel + 1) * d], [1, d], &device)?;
        let proj = ops::matmul(&sel_t, &self.w, None, MatmulSpec::default())?;
        let cand_rows: Vec<f32> = candidates
            .iter()
            .flat_map(|&i| h[i * d..(i + 1) * d].iter().copied())
            .collect();
        let cand_t = Tensor::from_f32(&cand_rows, [candidates.len(), d], &device)?;
        let logits = ops::matmul(
            &proj,
            &cand_t,
            None,
            MatmulSpec {
                trans_b: true,
                ..Default::default()
            },
        )?;
        logits.to_vec_f32()
    }
}

pub struct ReactModel {
    pub encoder: Encoder,
    pub pointer: PointerHead,
}

impl ReactModel {
    pub fn init_random(config: Config, device: &Device, seed: u64) -> Result<ReactModel> {
        let d_model = config.d_model;
        Ok(ReactModel {
            encoder: Encoder::init_random(config, device, seed)?,
            pointer: PointerHead::init_random(d_model, device, seed.wrapping_add(1))?,
        })
    }

    pub fn to_entries(&self) -> Result<TensorEntries> {
        let mut out = self.encoder.to_entries()?;
        out.push((
            "pointer.w".into(),
            self.pointer.w.shape().dims().to_vec(),
            self.pointer.w.to_vec_f32()?,
        ));
        Ok(out)
    }

    pub fn from_entries(
        config: Config,
        entries: &TensorEntries,
        device: &Device,
    ) -> Result<ReactModel> {
        let find = |name: &str| -> Result<Tensor> {
            let (_, shape, data) = entries
                .iter()
                .find(|(n, _, _)| n == name)
                .ok_or_else(|| forge::ForgeError::Fzm(format!("missing tensor {name}")))?;
            Tensor::from_f32(data, shape.clone(), device)
        };
        let pointer = PointerHead {
            w: find("pointer.w")?,
        };
        Ok(ReactModel {
            encoder: Encoder::from_entries(config, entries, device)?,
            pointer,
        })
    }
}

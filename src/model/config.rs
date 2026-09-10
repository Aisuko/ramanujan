use forge::EncoderDecoderConfig;

pub fn ramanujan_base() -> EncoderDecoderConfig {
    EncoderDecoderConfig {
        vocab_size: 32000,
        d_model: 384,
        n_encoder_layer: 6,
        n_decoder_layer: 4,
        n_head: 6,
        d_ff: 1536,
        max_source: 2048,
        max_target: 96,
        layer_norm_epsilon: 1e-5,
        pad_id: 0,
        bos_id: 1,
        eos_id: 2,
    }
}

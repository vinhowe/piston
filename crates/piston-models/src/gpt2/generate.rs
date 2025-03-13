use crate::gpt2::{GPT2Input, GPT2};
use maybe_async::maybe_async;
use ndarray::{Array3, Axis, Ix3};
use ndarray_stats::QuantileExt;
use piston::{shape, Device, Tensor};
use piston_nn::{Module, ModuleMode, ModuleModeGuard};

#[maybe_async]
pub async fn generate(
    model: &mut GPT2,
    prompt: Vec<i32>,
    // The callback now receives:
    //   - A Vec<i32> of the tokens fed in this pass (either the prompt, or the latest token)
    //   - An ndarray (shape: [1, seq_len, vocab_size]) containing the logits for that pass.
    //   - An ndarray (shape: [1, seq_len, seq_len]) containing the attention probabilities for that pass.
    callback: impl Fn(Vec<i32>, Array3<f32>, Vec<f32>),
    max_tokens: usize,
) -> anyhow::Result<Vec<Array3<f32>>> {
    let _eval_mode_guard = ModuleModeGuard::new(ModuleMode::Eval);

    // Preserve original cache setting and enable kv-cache for generation.
    let use_kv_cache = model.cache_mut().use_kv_cache();
    model.cache_mut().set_use_kv_cache(true);

    // This vector will accumulate the logits (as ndarray on CPU) from each model call.
    let mut all_logits_ndarray: Vec<Array3<f32>> = Vec::new();

    // all_tokens holds the entire context (prompt + generated tokens).
    let mut all_tokens = prompt.clone();
    // Count only the tokens that are generated (not in the original prompt)
    let mut generated_count = 0;

    while generated_count < max_tokens && all_tokens[all_tokens.len() - 1] != 256 {
        // For the first pass, feed the entire prompt.
        // For subsequent passes, feed only the latest generated token.
        let tokens_to_feed = if generated_count == 0 {
            &all_tokens[..]
        } else {
            &all_tokens[all_tokens.len() - 1..]
        };

        // Build the input tensor from tokens_to_feed.
        let input = Tensor::from_data(
            tokens_to_feed,
            shape![1, tokens_to_feed.len()],
            model.device.clone(),
        );

        // The index_pos is the total length of the context so far.
        let (result, attn_probs) = model.schedule(GPT2Input {
            x: input,
            index_pos: all_tokens.len(),
        })?;

        // Bring the logits to the CPU.
        let logits_cpu = result.to(&Device::CPU).await?;

        // Update the kv-cache:
        //   - For the first pass, update with the full length.
        //   - For subsequent passes, update only with the new token.
        if generated_count == 0 {
            model.cache_mut().update(all_tokens.len() + 1);
        } else {
            model.cache_mut().update(1);
        }

        // Convert the logits to an owned ndarray.
        // The logits have shape [1, seq_len, vocab_size] where:
        //   - seq_len == prompt.len() on first pass, or 1 on subsequent passes.
        let logits_nd = logits_cpu
            .to_ndarray_view::<f32>()
            .into_owned()
            .into_dimensionality::<Ix3>()
            .unwrap();
        // Store them in our accumulator.
        all_logits_ndarray.push(logits_nd.clone());

        let attn_probs_cpu = attn_probs
            .to(&Device::CPU)
            .await
            .map_err(|e| e.to_string())
            .unwrap();
        let attn_probs_shape = attn_probs_cpu.shape().to_vec();
        let attn_probs_data = attn_probs_cpu
            .to_vec::<f32>()
            .await
            .map_err(|e| e.to_string())
            .unwrap();

        // *** Stream the current pass via the callback ***
        // Pass the tokens that were fed (as a Vec) and the corresponding logits ndarray.
        callback(tokens_to_feed.to_vec(), logits_nd.clone(), attn_probs_data);

        // Extract the logits for the last token in this pass:
        // - For the first pass, that's at index (prompt.len() - 1).
        // - For later passes, the only token is at index 0.
        let seq_len_this_pass = tokens_to_feed.len();
        let last_logits = logits_nd.index_axis(Axis(1), seq_len_this_pass - 1);
        // last_logits has shape [1, vocab_size]; take the 0th row.
        let vocab_logits = last_logits.index_axis(Axis(0), 0);
        // Use argmax_skipnan from ndarray_stats to get the next token id.
        let next_token_id = vocab_logits.argmax_skipnan().unwrap() as i32;

        // Stop if the end-of-text token is generated.
        if next_token_id == 50256 {
            break;
        }

        // Append the generated token to the full context.
        all_tokens.push(next_token_id);
        generated_count += 1;
    }

    // Clean up: reset model state and restore the original kv-cache setting.
    model.reset();
    model.cache_mut().set_use_kv_cache(use_kv_cache);
    Ok(all_logits_ndarray)
}

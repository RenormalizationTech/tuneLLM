# Using the retained sarcasm controller

The retained local profile is [sarcasm_retained_controller.json](../configs/sarcasm_retained_controller.json). It identifies GCAD base coefficient 2, scale 1.5, `inverse=False`, zero-based blocks 9–19 on the pinned native-HF Qwen2.5-7B-Instruct checkpoint. The [closeout](sarcasm_gcad_low_strength_closeout.md) records a strong bounded sarcasm effect with a substantial coherence cost and a narrow gate margin. Retention does not guarantee correctness, helpfulness or independent trait control.

## Assets and runtime

Use the exact profile checkpoint revision, tokenizer, bank SHA256, tensor hashes, author-hook hash and runtime/environment manifest. The local bank remains an ignored retained artifact; the JSON does not contain model weights or make the assets publicly downloadable. It is a pointer, not a new runner.

The recorded environment is Python 3.11.18, torch 2.6.0+cu124, Transformers 4.52.3, accelerate 1.7.0, native BF16 with SDPA. Recorded device: physical GPU 0 UUID `GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`; the study required at least 24000 MiB free and utilization≤10%, leaving other workloads intact. Match the profile/runtime before operational reuse. Do not rerun completed study stages to obtain a conversation response.

This is separate from the physics LoRA workflow: load the unadapted pinned Qwen2.5-7B checkpoint, not a physics adapter, Qwen3 checkpoint or quantized substitute. No LoRA is part of this controller.

## Per-response sequence

1. Verify the model manifest and all profile-referenced bank/source hashes. Load the small bank with `torch.load(..., map_location="cpu", weights_only=True)`; require object `gcad`, trait/source_trait `sarcasm`, native-HF basis and exact tensor hashes/shapes: delta 28×3584, post-RoPE key 28×4×128 and demand 28. Do not normalize the deltas or replace them with profanity/impolite vectors.
2. Render the user's ordinary request with the pinned tokenizer's native default helpful system and assistant-generation suffix. Do not append the positive sarcasm elicitation system. That was used to fit the bank, and is a distinct prompted comparison arm.
3. In a fresh GCAD context, use the pinned `MultiLayerGatedSteerer` class and the profile's raw layer deltas/positive keys/average demands. The hook adds a demand-gated shift to attention output at all prefill and cached-decode positions. Its coefficient is `2 * 2.0 * sigmoid(1.5 * (demand - avg_demand))`, not a uniform displacement of 2.
4. Generate greedily with seed 42, batch 1, `max_new_tokens=160`, `use_cache=True`, preserving the pinned generation defaults/EOS handling. End the context and remove hooks even on failure. Start the next request with a fresh cache; do not carry a prior conversation's KV state or stack another trait controller.
5. Report cap/EOS separately from semantic completion. Avoid interpreting a coherent-sounding response as factually verified.

The following illustrates only the already-verified-model hook interface; it is not an executed command or a new model-loading/benchmark script:

```python
# model, tokenizer, inputs and assets have already been checked against the profile.
# factory is the hash-verified pinned author MultiLayerGatedSteerer.
# bank is the verified saved gcad object; tensors remain raw, not normalized.
layers = list(range(9, 20))
t = bank["tensors"]
with factory(
    model,
    v_vectors={i: t["delta"][i] for i in layers},
    k_pos_vectors={i: t["key"][i] for i in layers},
    avg_demands={i: t["demand"][i].item() for i in layers},
    layer_indices=layers, base_coef=2.0, scale=1.5,
    inverse=False, debug=False,
):
    with torch.no_grad():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=160, use_cache=True
        )
```

The audited code resolves only the pinned author class, avoiding the author's credential-setting pipeline imports. Its source hash is in the static profile. Restore the ordinary hook-free model after each context.

## Measured limits

Heldout selected sarcasm 63.78125 versus baseline 0.0625/random 0; 29/32 answers scored≥50. Coherence fell from 89.59375 to 70.71875: 18.875 points, only 1.125 inside the registered floor. Selected mean impoliteness 38.75 is a material descriptive coupling. 27/32 selected outputs were capped. Random matched raw layer norms/gates but had lower actual realized shift magnitude. No accuracy, capability, uncapped, joint-control or cross-model validation was performed.

The [original-Mistral SAPS result](../results/author_saps_mistral_01.json) is a separate original-model complaint-rewriting reference (TransformerLens, different hook basis and cap 60). Its fixed/adaptive settings cannot be substituted into this Qwen ordinary-response profile. The closed higher-strength parent null remains unchanged; no additional sweep or automatic follow-up is part of this workflow.

"""Synthetic cache mechanics only; descriptive token agreement, no quality gate."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

from scripts.qwen38_science_runtime import BASE, sha, write, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', choices=['native', 'peft'], required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous mechanics attempts')
    args.output.mkdir(parents=True)
    prompts = ['Count aloud from one to twenty.',
               'A box contains three red balls and two blue balls. Explain how many balls it contains.']
    write(args.output / 'registration.json', {
        'scope': 'synthetic mechanics; no development or final examples',
        'path': args.path, 'gpu': args.gpu, 'prompts': prompts,
        'max_new_tokens': 16, 'cached_vs_full_recompute': 'descriptive only; known strict logit check failed',
        'source_sha256': sha(__file__), 'versions': versions(),
        'runtime_sha256': sha(Path(__file__).with_name('qwen38_science_runtime.py')),
    })
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed
    from scripts.qwen38_science_runtime import load_runtime
    from scripts.qwen38_nf4_smoke import validate_loading
    set_seed(42)
    started = time.monotonic()
    if args.path == 'peft':
        model, _ = load_runtime()
        context = model.disable_adapter()
    else:
        model, loading = Qwen3_5ForCausalLM.from_pretrained(BASE, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='sdpa', output_loading_info=True)
        validate_loading(loading)
        model.config.use_cache = True
        context = nullcontext()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    observations = []
    def observe(module, positional, kwargs, output):
        cache = getattr(output, 'past_key_values', None)
        observations.append({'input_length': int(kwargs['input_ids'].shape[-1]),
                             'cache_class': type(cache).__name__ if cache is not None else None})
    # Observe the underlying text model output, before the PEFT wrapper forwards it.
    core = model.get_base_model() if args.path == 'peft' else model
    hook = core.register_forward_hook(observe, with_kwargs=True)
    rows = []
    try:
        with context, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for prompt in prompts:
                ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                batch = torch.tensor([ids], device='cuda:0')
                variants = {}
                for cached in [True, False]:
                    observations.clear()
                    tick = time.monotonic()
                    output = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                        use_cache=cached, do_sample=False, max_new_tokens=16, logits_to_keep=1,
                        temperature=None, top_p=None, top_k=None,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    variants[str(cached)] = {'tokens': output[0, len(ids):].tolist(),
                        'seconds': time.monotonic()-tick, 'forward_observations': list(observations)}
                rows.append({'prompt': prompt, 'input_tokens': len(ids), 'variants': variants,
                             'token_agreement': variants['True']['tokens'] == variants['False']['tokens']})
                write(args.output / 'partial.json', rows)
    finally:
        hook.remove()
    result = {'completed': True, 'path': args.path, 'records': rows,
              'elapsed_seconds': time.monotonic()-started,
              'interpretation': 'cache presence and token agreement are descriptive; strict cached/full-prefix logit equivalence is not established'}
    write(args.output / 'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()

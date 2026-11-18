"""Stage evaluated text-only NF4 or Q4_K_M payloads; never uploads or edits models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile

from scripts.qwen38_science_runtime import sha, write
from scripts.stage_physics_release import json_bytes, private_preflight

ROOT = Path(__file__).resolve().parents[1]
REVISION = '37561816e66b5b9153b1b64c10968c18c1e6f4a0'
SOURCE = ROOT / '.hf/hub/models--Qwen--Qwen3.8-27B/snapshots' / REVISION
COMPUTE = {'chat_template.jinja', 'config.json', 'generation_config.json',
           'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'}
REPOS={'nf4':'RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4',
       'gguf':'RenormalizationTech/Qwen3.8-27B-tuneLLM-GGUF'}


def publisher_manifest(kind,folder,private=False):
    if kind not in REPOS:raise ValueError('only the approved base release destinations are supported')
    files={}
    for file in sorted(Path(folder).iterdir()):
        if file.stat().st_size<30_000_000:private_preflight(file.read_bytes(),ROOT)
        files[file.name]={'bytes':file.stat().st_size,'sha256':sha(file)}
    return {'repo_id':REPOS[kind],'private':private,'files':files}


def verified_file(source, expected):
    source = Path(source)
    if source.stat().st_size != expected['bytes'] or sha(source) != expected['sha256']:
        raise ValueError(f'bound artifact changed: {source.name}')
    return source


def quality():
    directory = ROOT / 'outputs/qwen38-quant-screen-01'
    registry = ROOT / 'results/protocols/qwen38-quant-screen-01.json'
    if registry.read_bytes() != (directory / 'registration.json').read_bytes(): raise ValueError('screen registration differs')
    protocol = json.loads(registry.read_text())
    if sha(directory / 'fixture.json') != protocol['fixture_sha256']: raise ValueError('screen fixture changed')
    fixture = json.loads((directory / 'fixture.json').read_text())
    report = {'scope': 'Fixed small descriptive screen, no adapter or fresh science final items.',
              'registration_sha256':sha(registry),'fixture_sha256':sha(directory / 'fixture.json'),
              'fixture_ids':{task:[r['id'] for r in entries] for task,entries in fixture.items()},
              'counts': {'general_mc': 32, 'gsm8k': 32, 'qasper_development': 35},
              'arms': {}, 'limitations': [
                  'NF4 uses native BF16 floating modules, without PEFT preparation; adapter baselines use FP32 casts.',
                  'Cross-backend NF4 versus Q4 is a workflow comparison, not an isolated quantization effect.',
                  'BF16 GGUF is a local same-backend reference, not another published quantization.',
                  'Small public subsets may have been in pretraining; no equivalence or broad improvement claim.',
                  'Custom QASPER text-only subset and paragraph-ID evidence metrics; not the full official benchmark.',
                  'GSM8K last-number fallback can reward format-noncompliant answers; strict scores also reported.']}
    for arm in ['nf4','q4','bf16_gguf']:
        folder = directory / arm
        result = json.loads((folder / 'result.json').read_text())
        if not result.get('completed'): raise ValueError(f'complete screen required: {arm}')
        launcher = json.loads((directory / f'{arm}-launcher.json').read_text())
        if launcher['protocol_sha256'] != sha(registry): raise ValueError('launcher registration differs')
        mc = json.loads((folder / 'general_mc.json').read_text())
        gsm = json.loads((folder / 'gsm8k_scores.json').read_text())
        qa = json.loads((folder / 'qasper_development_scores.json').read_text())
        if len(mc) != 32 or gsm['total'] != 32 or qa['sampled_questions'] != 35:
            raise ValueError('screen denominators differ')
        report['arms'][arm] = {'general_mc_correct':sum(r['correct'] for r in mc),
            'gsm8k_fallback_correct':gsm['fallback_correct'], 'gsm8k_strict_correct':gsm['strict_correct'],
            'gsm8k_hit_token_limit':gsm['hit_token_limit'],
            'qasper_strict_primary':qa['strict']['groups']['primary'],
            'qasper_strict_all':qa['strict']['groups']['all_sampled'],
            'elapsed_seconds':result['elapsed_seconds'], 'gpu_count':result['gpu_count'],
            'gpu_hours':result['gpu_hours'], 'result_sha256':sha(folder / 'result.json'),
            'launcher_sha256':sha(directory / f'{arm}-launcher.json'),
            'evidence_files':{p.name:sha(p) for p in sorted(folder.iterdir()) if p.suffix in ['.json','.jsonl']}}
    return report


def card(kind, report, file_bytes):
    nf4 = kind == 'nf4'
    suffix = 'NF4' if nf4 else 'GGUF'
    title = f'Qwen3.8-27B tuneLLM {suffix} (text only)'
    arm_names={'nf4':'HF/bitsandbytes NF4 (native)','q4':'llama.cpp Q4_K_M','bf16_gguf':'llama.cpp BF16 GGUF (local reference)'}
    rows = '\n'.join(f"| {arm_names[arm]} | {r['general_mc_correct']}/32 | {r['gsm8k_fallback_correct']}/32 | {r['gsm8k_strict_correct']}/32 |"
                     for arm,r in report['arms'].items())
    loading = '''```python
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

repo = "RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4"
revision = HfApi().model_info(repo).sha  # Resolve once; retain this immutable SHA.
print("Resolved release:", revision)
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
model = Qwen3_5ForCausalLM.from_pretrained(
    repo, revision=revision, dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=False,
).eval()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain a band gap in two sentences."}],
    add_generation_prompt=True, enable_thinking=False,
    return_dict=True, return_tensors="pt",
).to(model.device)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    ids = model.generate(**inputs, do_sample=False, use_cache=True,
                         max_new_tokens=128, logits_to_keep=1)
print(tokenizer.decode(ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

Load the saved quantization configuration directly; do not quantize these stored
weights again. Use the explicit text-only class shown above. PEFT preparation
casts floating modules to FP32 and defines a different measured runtime; the
separate adapters require that preparation. The baseline checkpoint has no adapter.
''' if nf4 else '''The only model binary here is `Qwen3.8-27B-text-Q4_K_M.gguf`.
Use the pinned llama.cpp build in `provenance.json`. After downloading the file
from an immutable release revision, start the checked completion server:

```bash
llama-server -m Qwen3.8-27B-text-Q4_K_M.gguf -ngl 99 -c 8192 \\
  --host 127.0.0.1 --port 8080 --parallel 1 -t 4 -fa on
```

Then render the original nonthinking prompt explicitly, as in the evaluation:

```python
import requests
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.8-27B", revision="37561816e66b5b9153b1b64c10968c18c1e6f4a0")
prompt = tokenizer.apply_chat_template(
    [{"role":"user", "content":"Explain a band gap in two sentences."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
token_check = requests.post("http://127.0.0.1:8080/tokenize", json={
    "content":prompt, "add_special":False, "parse_special":True}, timeout=30)
token_check.raise_for_status()
assert token_check.json()["tokens"] == ids
response = requests.post("http://127.0.0.1:8080/completion", json={
    "prompt":ids, "n_predict":128, "temperature":0,
    "seed":42, "cache_prompt":False, "repeat_penalty":1.0,
    "samplers":["temperature"]}, timeout=300)
response.raise_for_status()
print(response.json()["content"])
```

The checked evaluation used the llama.cpp server completion API with exact
HF/llama.cpp prompt-token parity and prompt-cache reuse disabled. The BF16 GGUF
reference remains local and is not part of this payload. No importance matrix
was used. This file comes from the original higher-precision model, independently
of the bitsandbytes NF4 checkpoint.

Scores apply to this explicit HF-rendered, thinking-off token route. Default
llama.cpp chat-template behavior can differ and was not evaluated here.
'''
    runtime=json.loads((ROOT / 'results/protocols/qwen38-quant-runtime-witness-01.json').read_text())
    v=runtime['versions']
    primary_counts=sorted(set(r['qasper_strict_primary']['denominator'] for r in report['arms'].values()))
    if len(primary_counts) != 1: raise ValueError('paired primary denominators differ')
    return f'''---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: {"transformers" if nf4 else "gguf"}
pipeline_tag: text-generation
tags:
- qwen3_5
- text-only
- {"bitsandbytes" if nf4 else "llama.cpp"}
- {"nf4" if nf4 else "gguf"}
---
# {title}

Unadapted, text-only quantization of `Qwen/Qwen3.8-27B`, pinned upstream revision
`{REVISION}`. The model weights occupy **{file_bytes:,} bytes**.
Vision and MTP components are excluded; this package provides text generation.
No task adapter, merged adapter, or activation steering is included.
Modification notice: upstream weights/configuration were modified by text-only
extraction, exclusion of vision/MTP components, and quantization.

## Measured scope

| Runtime representation | General MMLU | GSM8K numeric fallback | GSM8K strict |
| --- | ---: | ---: | ---: |
{rows}

These are fixed, small descriptive subsets, with batch one, greedy nonthinking
generation and a 512-token generation cap. MMLU uses custom zero-shot four-letter likelihoods.
The same screen includes 35 QASPER development questions; exact denominators,
custom text-evidence scores and caveats are in `evaluation.json`. These overlap
development/retention fixtures and are not an independent final benchmark.
The complete-text QASPER primary subset has {primary_counts[0]} questions per arm.
NF4 and llama.cpp differ in both representation and backend. Compare Q4 with
the local BF16 GGUF reference for the same-backend measurement; do not attribute
every NF4/Q4 difference to quantization.
No matched BF16 Transformers reference was run; NF4's own quantization effect
was not isolated.

The original NF4 cached/full-prefix strict logit diagnostic exceeded tolerance
(maximum observed absolute difference 0.125). Its failure is retained. Short
token replay and functional reload checks do not establish numerical equivalence.
Timing includes model load and the recorded tasks; concurrent jobs and a sharded
BF16 reference limit speed comparisons. No isolated TTFT benchmark is claimed.

## Use

{loading}

Recorded mid-study Python environment: PyTorch {v['torch']}, Transformers {v['transformers']},
bitsandbytes {v['bitsandbytes']}, PEFT {v['peft']}, Accelerate {v['accelerate']}.
The original screen omitted per-arm package-version fields; the separately
dated runtime witness and environment lock preserve that provenance limit.
The GGUF runtime is pinned by llama.cpp revision and binary hashes in provenance.
The HF NF4 path uses reference PyTorch GDN/causal-convolution kernels;
optimized FLA/causal-conv packages were not installed. Hardware: RTX A6000 48GB;
one GPU per NF4/Q4 arm, two GPUs for the local BF16 GGUF reference.

Apache 2.0 upstream license is included. The screen is an evaluation of public
datasets; this payload redistributes aggregate results and item IDs, not dataset
text. Model and dataset
provenance, licenses, hashes and reproduction commands are supplied separately.
This model may produce unsupported scientific statements; the small screen does
not establish scientific reliability.

Part of [tuneLLM](https://github.com/RenormalizationTech/tuneLLM).
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['nf4','gguf'],required=True)
    p.add_argument('--private',action='store_true')
    p.add_argument('--attempt',type=int,default=1)
    args=p.parse_args(); report=quality()
    if args.attempt<1:raise ValueError('positive staging attempt required')
    suffix='NF4' if args.kind=='nf4' else 'GGUF'
    name=f'Qwen3.8-27B-tuneLLM-{suffix}'
    out=ROOT / 'artifacts/releases' / name
    manifest_path=ROOT / f'outputs/qwen38-release-{args.kind}-{args.attempt:02d}.json'
    if out.exists() or manifest_path.exists(): raise FileExistsError('preserve previous staging')
    files={}
    if args.kind=='nf4':
        artifact=json.loads((ROOT / 'results/qwen38_27b_nf4_artifact_01.json').read_text())
        if set(artifact['files']) != COMPUTE | {'smoke_origin.json'}:
            raise ValueError('computational file inventory changed; review sharding and tokenizer assets')
        for filename in sorted(COMPUTE):
            source=ROOT / 'artifacts/qwen38-27b-nf4-01' / filename
            files[filename]=verified_file(source,artifact['files'][filename])
        weight_bytes=artifact['weight_bytes']
    else:
        filename='Qwen3.8-27B-text-Q4_K_M.gguf'
        conversion=json.loads((ROOT / 'outputs/qwen38-27b-gguf-01/result.json').read_text())
        if not conversion['completed']: raise ValueError('conversion incomplete')
        files[filename]=verified_file(ROOT / 'artifacts/qwen38-27b-gguf-01' / filename,conversion['files'][filename])
        weight_bytes=files[filename].stat().st_size
    # Validate small computational files without rewriting their bound bytes.
    for filename,source in files.items():
        if source.stat().st_size < 30_000_000: private_preflight(source.read_bytes(),ROOT)
    out.mkdir(parents=True)
    for filename,source in files.items(): os.link(source,out / filename)
    if args.kind=='nf4':
        import torch
        from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from safetensors import safe_open
        from scripts.qwen38_nf4_smoke import official_support
        config=Qwen3_5TextConfig.from_pretrained(out,local_files_only=True)
        tokenizer=AutoTokenizer.from_pretrained(out,local_files_only=True)
        if config.model_type != 'qwen3_5_text' or config.architectures != ['Qwen3_5ForCausalLM']:
            raise ValueError('staged payload configuration differs')
        with torch.device('meta'): meta=Qwen3_5ForCausalLM(config)
        if not official_support()['available']: raise ValueError('required official text-key conversion is unavailable')
        with safe_open(out / 'model.safetensors',framework='pt',device='cpu') as checkpoint:
            # save_pretrained preserves the original serialization prefix; the
            # pinned official qwen3_5_text loader removes language_model on reload.
            names={k.replace('model.language_model.','model.',1) if k.startswith('model.language_model.') else k for k in checkpoint.keys()}
            if not set(meta.state_dict()) <= names: raise ValueError('checkpoint parameter names incomplete')
        original_tokenizer=AutoTokenizer.from_pretrained(ROOT / 'artifacts/qwen38-27b-nf4-01',local_files_only=True)
        if tokenizer('A short serialization probe.') != original_tokenizer('A short serialization probe.'):
            raise ValueError('staged tokenizer differs')
    shutil.copyfile(SOURCE / 'LICENSE',out / 'LICENSE')
    shutil.copyfile(ROOT / 'outputs/qwen38-science-01/environment.freeze.txt',out / 'environment.lock.txt')
    for requirement_file in ['requirements-qwen38-science.txt','requirements-qwen38-nf4.txt']:
        shutil.copyfile(ROOT/requirement_file,out/requirement_file)
    (out / 'README.md').write_text(card(args.kind,report,weight_bytes))
    (out / 'evaluation.json').write_bytes(json_bytes(report,ROOT))
    provenance={'upstream':{'repo_id':'Qwen/Qwen3.8-27B','revision':REVISION},
        'llama_cpp_repository':'https://github.com/ggml-org/llama.cpp',
        'scope':'text only; unadapted; vision/MTP excluded','representation':args.kind,
        'quantization_registration':json.loads((ROOT / 'results/protocols/qwen38-quant-screen-01.json').read_text()),
        'model_verification':json.loads((ROOT / ('results/qwen38_27b_nf4_artifact_01.json' if args.kind=='nf4' else 'outputs/qwen38-27b-gguf-01/registration.json')).read_text()),
        'source_supplement':json.loads((ROOT / 'results/protocols/qwen38-quant-source-supplement-01.json').read_text()),
        'runtime_witness':json.loads((ROOT / 'results/protocols/qwen38-quant-runtime-witness-01.json').read_text()),
        'official_source_manifest':json.loads((ROOT / 'outputs/qwen38-27b-nf4-01/source.json').read_text()),
        'nf4_execution_record':json.loads((ROOT / 'results/qwen38_27b_nf4_execution_01.json').read_text()),
        'stored_quantization_config':json.loads((ROOT / 'artifacts/qwen38-27b-nf4-01/config.json').read_text())['quantization_config'],
        'dataset_attribution':{
            'mmlu':{'source':'https://huggingface.co/datasets/cais/mmlu','license':'MIT','revision':json.loads((ROOT / 'data/benchmark-screen-01/provenance.json').read_text())['mmlu_revision']},
            'gsm8k':{**json.loads((ROOT / 'data/gsm8k/provenance.json').read_text())['source'],'url':'https://huggingface.co/datasets/openai/gsm8k'},
            'qasper':json.loads((ROOT / 'data/qasper-v03-01/manifest.json').read_text())},
        'qasper_preparation_manifest':json.loads((ROOT / 'outputs/qwen38-science-01/prepared-qasper-tokenizer-01/manifest.json').read_text()),
        'payload_inventory_note':'The historical NF4 manifest includes smoke_origin.json. That metadata file is omitted from the compute payload; its identity remains in provenance. The publication manifest defines the actual released files.',
        'staging_source_sha256':sha(__file__)}
    if args.kind=='gguf':
        provenance['conversion_result']=json.loads((ROOT / 'outputs/qwen38-27b-gguf-01/result.json').read_text())
        provenance['embedded_chat_template_sha256']='af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375'
    (out / 'provenance.json').write_bytes(json_bytes(provenance,ROOT))
    source_files=['scripts/'+name+'.py' for name in ['common','qwen38_quant_execute','qwen38_quant_screen',
        'benchmark_screen','batched_generation','gguf_eval','qwen38_nf4_smoke','qwen38_science_runtime',
        'qwen38_gguf','qwen38_gguf_check','qwen38_acquire_qasper','qwen38_prepare_inputs',
        'qwen38_science_data','qwen38_science_score','qwen38_qasper_protocol','qwen38_train_science',
        'training_data','qwen38_stage_base','stage_physics_release','prepare_benchmark_screen',
        'prepare_gsm8k','download_model']]+['evaluation/gsm8k.py']
    with zipfile.ZipFile(out / 'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in source_files:
            data=(ROOT / filename).read_bytes();private_preflight(data,ROOT)
            archive.writestr(filename,data)
        archive.writestr('scripts/__init__.py','')
        archive.writestr('evaluation/__init__.py','')
        archive.writestr('LICENSE',(ROOT/'LICENSE').read_bytes())
        for source in (ROOT/'outputs/qwen38-science-01/prepared-qasper-tokenizer-01/source/scripts').glob('*.py'):
            data=source.read_bytes();private_preflight(data,ROOT)
            archive.writestr('frozen_qasper_preparation/scripts/'+source.name,data)
    creation='`python -B -m scripts.qwen38_nf4_smoke --execute-gpu --save-nf4 artifacts/qwen38-27b-nf4-01 --compare-reload`' if args.kind=='nf4' else '`python -B -m scripts.qwen38_gguf --prepare`, followed by `python -B -m scripts.qwen38_gguf --execute`'
    (out / 'REPRODUCE.md').write_text('The exact source snapshot is in source.zip. Use the pinned environment and registrations in provenance.json.\n\nPython3.13.3 was used. In a fresh isolated environment, install the observed Torch wheel with `python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130`, then `python -m pip install -r requirements-qwen38-science.txt`. The full installed versions are in environment.lock.txt. This preserves the project\'s older 4B/7B environment. The separate GGUF runtime requires the pinned llama.cpp revision and build described in provenance.json.\n\nThe scripts assume the tuneLLM directory layout and immutable run directories; this is a source snapshot, not a standalone one-command installer. Restore/acquire model assets at the pinned upstream revisions, build the pinned llama.cpp server, acquire the original QASPER data and benchmark fixtures, and prepare the frozen IDs recorded by the registrations. New execution directories must be registered rather than overwriting historical evidence.\n\nThe original artifact-creation entrypoint is '+creation+'. The checked screen entrypoint is `python -B -m scripts.qwen38_quant_execute --arm nf4 --gpus 0`, with q4 on one free GPU and bf16_gguf on two free GPUs. These commands consume pre-registered fixtures and artifact hashes; setup and path bindings must be adapted and recorded for another machine. The README quickstart can load the published model without those research assets.\n')
    manifest=publisher_manifest(args.kind,out,args.private)
    write(manifest_path,manifest)
    print(json.dumps({'staged':str(out.relative_to(ROOT)),'manifest':str(manifest_path.relative_to(ROOT)),
                      'repo_id':manifest['repo_id'],'private':args.private,'upload_performed':False},indent=2))


if __name__=='__main__':main()

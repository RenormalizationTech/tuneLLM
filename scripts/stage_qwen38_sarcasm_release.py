"""CPU-only, fresh explicit-file staging. Never uploads, loads a model, or edits evidence."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/qwen38-sarcasm-release-01'
REPO='RenormalizationTech/qwen38-27b-sarcasm-gcad-controls'
BASE='RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4'
REV='586bb02218d196f485c321bbde3348677c63fc6a'
BANK='4fd9f86bac07cd858c8f4177ce0e9f1095d30d105f0179713c78da671ab545de'
PORT='3d76d8147ffbe6f802b285d643b4363e09349f0dd14a5f6a2924545b5966abe3'
CANONICAL='61362b38ecb32581b58adb6f5e5caaaec7f16598d985ef1b834568210ca081ad'
RUNTIME=ROOT/'scripts/qwen38_sarcasm_release_runtime'
BASE_FILES=('model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja')
SITES=[23,27,31,35,39,43]


def sha(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(name):return json.loads((ROOT/name).read_text())


def save(file,value):
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def aggregate(result):
    if result['outcome']!='exploratory_null_through_b16' or result['old_outcome_unchanged']!='development_null':
        raise ValueError('canonical scientific outcomes changed')
    b=result['part_B'];a=result['part_A']
    public_b={k:b[k] for k in ['status','role','outcome','exploratory_candidate','metrics','gates',
        'random16_descriptive_gates','b16_direction_check','prompted_validity_concern','descriptives',
        'pooled_sarcasm_impoliteness','pooled_caveat','telemetry']}
    public_a={k:a[k] for k in ['status','role','question_clusters','cells','bootstrap','A1_alignment',
        'A3_residual','A4_gate','A5_nll_S_minus_L','denominator_note']}
    public_a['A2_magnitude']={site:{content:{k:v for k,v in values.items() if k!='paired_per_question'}
        for content,values in contents.items()} for site,contents in a['A2_magnitude'].items()}
    return {'outcome':result['outcome'],'old_outcome_unchanged':result['old_outcome_unchanged'],
        'canonical_result_sha256':CANONICAL,'scope':result['scope'],'part_A':public_a,'part_B':public_b,'limitations':result['limitations']}


def table(result):
    b=result['part_B'];floor=max(50,b['metrics']['baseline']['coherence']-20)
    lines=['| Strength / arm | Sarcasm | Coherence | Impoliteness | Coherence <50 | Registered gates |',
           '| --- | ---: | ---: | ---: | ---: | --- |']
    for arm,label in [('baseline','Baseline'),('prompted','Prompted reference'),('b2','2'),('b4','4'),('b8','8'),('b16','16'),('random16','Random16')]:
        m=b['metrics'][arm];d=b['descriptives'][arm]
        if arm in ('baseline','prompted'):gate='Reference'
        elif arm=='random16':gate='Descriptive only; setting-matched, not realised-shift-matched'
        elif arm=='b8':
            if b['gates']['8']['coherence_relative']:raise ValueError('b8 relative coherence failure changed')
            gate=f'FAIL relative coherence: {m["coherence"]:.2f} < {floor:.2f}; owner-chosen EXPERIMENTAL demonstration default'
        else:
            failed=[k for k,v in b['gates'][arm[1:]].items() if k!='all_pass' and not v]
            gate='FAIL: '+', '.join(failed)
            if arm=='b2':gate+='; essentially no judged sarcasm'
            if arm=='b16':gate+='; extreme setting'
        lines.append(f'| {label} | {m["trait_conservative"]:.2f} | {m["coherence"]:.2f} | {d["impoliteness_mean"]:.2f} | {d["coherence_below50"]}/32 | {gate} |')
    return '\n'.join(lines)


def card(draft,result):
    if '## What this controls package does' not in draft:raise ValueError('card draft section missing')
    body='## What this controls package does'+draft.split('## What this controls package does',1)[1]
    body=re.sub(r'\| Strength/arm \|.*?(?=\n\n)',table(result),body,flags=re.S)
    if table(result) not in body:raise ValueError('card numeric table replacement failed')
    body=body.replace('The planned CLI exposes','The CLI exposes').replace('The proposed packaged interface accepts','The interface accepts')
    body=body.replace('The controls package must preserve','This controls package preserves').replace('the proposed public payload','this public payload')
    header=f'''---
license: apache-2.0
base_model: {BASE}
pipeline_tag: text-generation
library_name: pytorch
tags: [gcad, steering, sarcasm, experimental, nf4]
---
# Experimental Qwen3.8-27B sarcasm GCAD controls

**Experimental research/demonstration only. No tested strength passed all registered acceptance gates
(`exploratory_null_through_b16`). This is not a validated controller.** The owner-chosen default
b=8 failed the relative-coherence gate. Strong settings can produce mocking or demeaning text.
This is not a production deployment recommendation or an official hosted Inference API integration.

## Standalone local use

Requires Python 3.13.3 and the recorded Torch 2.12.0+cu130 / CUDA 13.0 build, plus
the exact packages in `requirements.txt`. Install that Torch build first, then install the
remaining requirements. Optional FLA/causal-conv1d/kernels packages must remain absent, as studied.

Download this controls repository at a **specific public commit** and use the release-manifest
SHA256 printed in that commit message as an external trust anchor. Do not derive the trust
anchor from an unchecked local manifest. This verifies byte consistency, not publisher identity.
The actual controls commit is assigned at publication; it cannot be self-embedded in its own files.

Explicitly download the controls using that public commit, supplied as `HF_REV`:

```python
import os
from huggingface_hub import snapshot_download
snapshot_download("{REPO}", revision=os.environ["HF_REV"], local_dir="qwen38-controls")
```

Then enter `qwen38-controls` to run the commands below. Set `MANIFEST_SHA256` from the
same commit message. Downloads are explicit user actions, not runtime side effects.

The base snapshot is pinned to `{BASE}` at `{REV}`. One explicit download example:

```python
from huggingface_hub import snapshot_download
snapshot_download("{BASE}", revision="{REV}", local_dir="../qwen38-base")
```

From the downloaded controls directory, set `MANIFEST_SHA256` to the external release digest:

```bash
python -m runtime.example --snapshot ../qwen38-base --manifest-sha256 "$MANIFEST_SHA256" --strength 8 --prompt "How does a compass work?"
```

The runtime itself performs **no implicit downloads**. Real local_dir files are preferred.
Both legacy one-hop HF snapshot-to-repository-blob links and the pinned HF Hub2.4
two-hop shared-blob cache are supported and hash-verified. The shared store must have its
regular version-1 ownership marker and regular prefix directories/files; directory links,
third links and outside-cache targets are refused. All six base runtime
files and every controls payload file are checked. Consumer `.git`, `.cache`, `.gitattributes`
and generated runtime `__pycache__` metadata are excluded from the payload inventory only.
About 18 GB of model weights must fit on one BF16-capable CUDA device; no CPU/offload fallback.

Use `--strength 0` for ordinary generation without steering hooks, `8` for the experimental
demonstration, or `16` for extreme steering. Finite strengths in [0,16] are accepted; intermediate
values other than evaluated 2/4/8 are untested. Booleans/nonfinite/out-of-range strengths, padding,
batch sizes above one, nested or concurrent use, and strength changes mid-call are refused.
The supported example is one English user turn, no system prompt, greedy decoding, repetition
penalty 1.05, native caching, seed 42 and at most160 generated tokens.

```python
from runtime import load_runtime
controls = load_runtime("../qwen38-base", manifest_sha256="EXTERNAL_RELEASE_MANIFEST_SHA256")
answer = controls.generate("How does a compass work?", strength=8, max_new_tokens=160)
print(answer["text"])
```

Loading base weights alone does not enable GCAD. The bank is derived from that model's own
generations. Only the exact checkpoint/software/GPU stack was mechanically checked;
other hardware may generate different tokens. See `METHODS.md`, `evaluation.json`, and `metrics.csv`.
Functional package verification is recorded separately; scientific results and limitations appear below.

'''
    return header+body


def privacy_scan(folder):
    forbidden=re.compile(r'/home/|/Users/|/tmp/|[A-Za-z]:\\|angkunwu|outputs/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|session_id|sessionId|"argv"|total_cost|cost_usd')
    for f in folder.rglob('*'):
        if f.is_symlink():raise ValueError('staged payload must contain no symlinks')
        if f.is_file() and f.suffix!='.safetensors':
            if forbidden.search(f.read_text()):raise ValueError(f'private/raw-runtime detail in public text: {f.name}')


def environment_profile():
    import torch
    pins=dict(line.split('==') for line in (ROOT/'requirements-qwen38-nf4.txt').read_text().splitlines() if line and not line.startswith('#'))
    actual={name:importlib.metadata.version(name) for name in pins}
    if any(actual[name].split('+')[0]!=pin for name,pin in pins.items()) or platform.python_version()!='3.13.3':
        raise ValueError('stage from the studied isolated NF4 environment')
    source=Path(importlib.metadata.distribution('transformers').locate_file('transformers/models/qwen3_5/modeling_qwen3_5.py'))
    profile={'python':platform.python_version(),'torch_build':str(torch.__version__),'cuda_build':torch.version.cuda,
        'packages':actual,'optional_kernels':{name:importlib.util.find_spec(name) is not None for name in ['fla','causal_conv1d','kernels']},
        'qwen35_modeling_sha256':sha(source)}
    if profile['torch_build']!='2.12.0+cu130' or profile['cuda_build']!='13.0' or any(profile['optional_kernels'].values()) or profile['qwen35_modeling_sha256']!='1a475c45d900e0bfb06b9b245257d7f3234fd247474fcbd3363b03430e504a68':
        raise ValueError('exact studied Torch/CUDA/native-module/kernel profile required')
    return profile


def stage(destination=DEST):
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise FileExistsError('fresh staging root only')
    bound_sources=[Path(__file__),ROOT/'docs/qwen38_sarcasm_hf_card_draft.md',
                   *[RUNTIME/name for name in ['__init__.py','gcad.py','loader.py','example.py']]]
    source_hashes={str(p):sha(p) for p in bound_sources}
    environment=environment_profile()
    canonical=ROOT/'results/qwen38_sarcasm_followup_01.json'
    if sha(canonical)!=CANONICAL or sha(ROOT/'scripts/qwen38_gcad_port.py')!=PORT or sha(RUNTIME/'gcad.py')!=PORT:
        raise ValueError('canonical result or frozen/portable port identity changed')
    source_bank=ROOT/'outputs/qwen38-sarcasm-gcad-01/extraction/bank.safetensors'
    if sha(source_bank)!=BANK:raise ValueError('frozen bank identity changed')
    original=read('results/qwen38_27b_nf4_artifact_01.json');published=read('results/qwen38_nf4_public_verification_01.json')
    base={name:original['files'][name] for name in BASE_FILES}
    if published['repo_id']!=BASE or published['revision']!=REV or any(base[n]!=published['files'][n] for n in BASE_FILES):
        raise ValueError('public pinned base runtime bytes differ from evaluated artifact')
    result=read('results/qwen38_sarcasm_followup_01.json');public=aggregate(result)
    draft=(ROOT/'docs/qwen38_sarcasm_hf_card_draft.md').read_text();readme=card(draft,result)
    payload=destination/'payload';(payload/'runtime').mkdir(parents=True)
    for name in ('__init__.py','gcad.py','loader.py','example.py'):shutil.copyfile(RUNTIME/name,payload/'runtime'/name)
    shutil.copyfile(source_bank,payload/'bank.safetensors')
    (payload/'README.md').write_text(readme)
    (payload/'METHODS.md').write_text('''# Mechanics and measured scope

Six full-attention sites: 23,27,31,35,39,43 (zero-based). Linear-attention blocks are untouched.
The native per-head Q/gate split, Q/K norms, partial RoPE and GQA mapping follow the pinned runtime.
The intervention is added after native gating and o_proj: 2b * sigmoid(1.5*(demand-centre)) * delta.
The port source is a byte-identical copy of the reviewed research port; see release_manifest.json.
No weights are trained/edited by the runtime. Intermediate strengths are untested.

[Equations and tuning guide](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/qwen38_sarcasm_gcad_guide.md)
[Full experiment record](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/qwen38_sarcasm_followup_execution.md)

Both research outcomes remain null. No dose is asserted to have caused the historical null.
Random16 is setting-matched with one seed set, not realised-shift-matched. Its decode gate was
0.36–0.50 versus0.68–0.87 for b16, with smaller actual decode shifts; b8/lower strengths have no random control.
This package carries aggregate metrics, not fixtures, trajectories, activations, or judge sessions.
The coherence rubric is AI-judged and is not an independent factual-correctness test.
''')
    save(payload/'evaluation.json',public)
    with (payload/'metrics.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','n','sarcasm','coherence','impoliteness','coherence_below50','cap_hits','registered_gates_pass'])
        for arm,m in result['part_B']['metrics'].items():
            d=result['part_B']['descriptives'][arm]
            writer.writerow([arm,m['n'],m['trait_conservative'],m['coherence'],d['impoliteness_mean'],d['coherence_below50'],m['caps'],result['part_B']['gates'][arm[1:]]['all_pass'] if arm in ('b2','b4','b8','b16') else 'reference'])
    licence=ROOT/'.hf/hub/models--Qwen--Qwen3.8-27B/snapshots/37561816e66b5b9153b1b64c10968c18c1e6f4a0/LICENSE'
    if sha(licence)!=published['files']['LICENSE']['sha256']:raise ValueError('base licence bytes differ from public record')
    shutil.copyfile(licence,payload/'BASE_LICENSE')
    for name in ['qwen3_gcad_port_LICENSE','qwen3_gcad_port_NOTICE','qwen38_gcad_port_NOTICE']:shutil.copyfile(ROOT/'scripts'/name,payload/name)
    shutil.copyfile(ROOT/'scripts/qwen3_gcad_port_LICENSE',payload/'LICENSE')
    shutil.copyfile(ROOT/'requirements-qwen38-nf4.txt',payload/'requirements.txt')
    save(payload/'environment.lock.json',environment)
    privacy_scan(payload)
    files={p.relative_to(payload).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(payload.rglob('*')) if p.is_file()}
    registration=ROOT/'results/protocols/qwen38-sarcasm-followup-01.json';review=ROOT/'results/reviews/qwen38_sarcasm_followup_review_06.md'
    save(payload/'release_manifest.json',{'schema_version':1,'repo_id':REPO,'private':False,'status':'experimental_demonstration_not_validated',
        'base':{'repo_id':BASE,'revision':REV,'files':base},'bank_sha256':BANK,'sites':SITES,'scale':1.5,
        'default_strength':8,'extreme_strength':16,'supported_strength_range':[0,16],'evaluated_strengths':[2,4,8,16],
        'environment':environment,'frozen_port_sha256':PORT,'builder_sha256':sha(__file__),
        'canonical_result_sha256':CANONICAL,'followup_registration_sha256':sha(registration),'independent_review06_sha256':sha(review),
        'card_draft_sha256':sha(ROOT/'docs/qwen38_sarcasm_hf_card_draft.md'),'files':files,
        'consumer_inventory_exclusions':['.git/','.cache/','.gitattributes','runtime/**/__pycache__/*.pyc'],
        'verification_scope':'Externally pinned manifest and exact files prove byte consistency, not publisher identity. GPU/download checks are separate.'})
    privacy_scan(payload)
    mf_sha=sha(payload/'release_manifest.json')
    inventory={p.relative_to(payload).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(payload.rglob('*')) if p.is_file()}
    save(destination/'publication_manifest.json',{'repo_id':REPO,'private':False,'files':inventory,'release_manifest_sha256':mf_sha,
        'required_commit_message':f'Release manifest SHA256: {mf_sha}'})
    save(destination/'staging_receipt.json',{'status':'staged_CPU_only_not_uploaded','release_manifest_sha256':mf_sha,
        'publication_manifest_sha256':sha(destination/'publication_manifest.json'),'builder_sha256':sha(__file__),
        'runtime_sha256':{n:sha(RUNTIME/n) for n in ['__init__.py','gcad.py','loader.py','example.py']},
        'verifier_sha256':sha(ROOT/'scripts/verify_qwen38_sarcasm_release.py'),
        'CPU_tests_sha256':sha(ROOT/'tests/test_qwen38_sarcasm_release.py'),
        'source_result_sha256':CANONICAL,'no_GPU_model_judge_calls':True})
    if any(sha(p)!=h for p,h in source_hashes.items()):
        raise ValueError('builder/runtime/card changed during staging; preserve attempt and restage fresh')
    return {'payload':str(payload),'publication_manifest':str(destination/'publication_manifest.json'),'manifest_sha256':mf_sha}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=DEST)
    args=p.parse_args();print(json.dumps(stage(args.output_root),indent=2))


if __name__=='__main__':main()

"""CPU-only steering-bank packaging; no upload, model loads or network code."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess

from scripts import author_gcad_impolite_common as author
from scripts.save_nf4 import tensor_hash

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/steering-release-staging-02'
PUBLIC_DEST=ROOT/'outputs/steering-release-staging-03'
PUBLIC_UPLOAD_MANIFEST=ROOT/'outputs/steering-release-staging-03-publication-manifest.json'
PUBLIC_REPO='RenormalizationTech/qwen2.5-7b-gcad-steering-controls'
VALIDATED_02_SHA='05aea48f36a5365bbe55adad1ee5ef17d52df3711557f960c1f158fab44a853e'
RELOAD_RECORD='results/steering_release_real_reload_02.json'
FOURB_RESULT='results/qwen3_4b_fixed_transfer_01.json'
FOURB_CLOSEOUT='docs/qwen3_4b_fixed_transfer_closeout.md'
FOURB_RESULT_SHA='004c2de7bc24a8e6686fd6476964f0ea18ee1077fab50dfd07c2679816256d13'
PUBLIC_EXTRA={'METHODS.md','related_results/qwen3_4b_fixed_transfer_01.public.json','related_results/qwen3_4b_fixed_transfer_closeout.md'}
PROJECT_PATH=re.compile(r'(?<![A-Za-z0-9_])(?:configs|outputs|scripts|docs|results)/')
BASE='Qwen/Qwen2.5-7B-Instruct'
REV='a09a35458c702b33eeacc393d103063234e8bc28'
TRAITS=('profanity','impolite','sarcasm')
COEFS={'profanity':5.,'impolite':3.5,'sarcasm':2.}
SHAPES={'delta':(28,3584),'key':(28,4,128),'demand':(28,)}
PORT='scripts/qwen3_gcad_port.py'
AUTHOR_ROOT='outputs/persona-author-replication-audit-01/gcad'
PROFILE_FILES={'profanity':'configs/profanity_retained_controller.json','sarcasm':'configs/sarcasm_retained_controller.json'}
FORBIDDEN_KEYS={'question','answer','reason','reasons','rendered_prompt','input_ids','generated_ids','attention_mask','raw_text','raw_answers','judge_reasons'}
BAD_TEXT=re.compile(r'/home/|/Users/|/tmp/|[A-Za-z]:\\|Bearer\s+\S+|\bhf_[A-Za-z0-9]{15,}\b|\bsk-[A-Za-z0-9_-]{15,}\b')


def sha(file):
    digest=hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read(file):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    return json.loads(Path(file).read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def save(file,value):
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    file.chmod(0o600)


def local(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('source references must be explicit project-relative paths')
    file=ROOT/p
    if not file.resolve().is_relative_to(ROOT):raise ValueError('source path escapes project')
    return file


def checked(file,expected):
    if not re.fullmatch('[0-9a-f]{64}',expected) or sha(file)!=expected:raise ValueError('source hash mismatch before tensor load')
    return file


def committed(name):
    result=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,capture_output=True,check=True)
    if result.stdout!=local(name).read_bytes():raise ValueError('source profile/registration is not the committed version')


def refs(value):
    """Verify existing profile references without following unrelated manifests."""
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:yield value['path'],value['sha256']
        for v in value.values():yield from refs(v)
    elif isinstance(value,list):
        for v in value:yield from refs(v)


def validate_tensors(values,expected=None):
    import torch
    if set(values)!=set(SHAPES):raise ValueError('exact delta/key/demand required')
    for name,shape in SHAPES.items():
        v=values[name]
        if not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or v.device.type!='cpu' or tuple(v.shape)!=shape or not torch.isfinite(v).all():raise ValueError('28-layer FP32 CPU Qwen7 bank geometry required')
        if expected is not None and tensor_hash(v)!=expected[name]:raise ValueError('original tensor hash differs')
    return values


def load_source_profile(trait):
    import torch
    name=PROFILE_FILES[trait];committed(name);profile=read(local(name))
    for file,digest in refs(profile):checked(local(file),digest)
    model=profile['model'];control=profile['controller']
    if (model['id'],model['revision'],model['dtype'],model['attention'])!=(BASE,REV,'bfloat16','sdpa'):raise ValueError('wrong model/revision/runtime profile')
    if control['blocks']!=list(range(9,20)) or control['base_coef']!=COEFS[trait] or control['scale']!=1.5 or control['inverse'] is not False:raise ValueError('wrong fixed retained setting')
    file=checked(local(control['bank']['path']),control['bank']['sha256'])
    original=torch.load(file,map_location='cpu',weights_only=True)
    values=validate_tensors(original[control['object_key']]['tensors'],control['tensor_sha256'])
    evidence={'source_profile_sha256':sha(local(name)),'source_bank_sha256':sha(file),
        'references_sha256':{PurePosixPath(n).name+'_'+h[:8]:h for n,h in refs(profile)},
        'tokenizer_fingerprint':profile['basis']['tokenizer_sha256'],'rendering_fingerprint':profile['basis']['rendering_sha256'],
        'input_provenance':'Project reconstructed/generated contrast prompts; source template Apache-2.0. Same Claude model family generated/reviewed/judged; not human-ground-truth validation.'}
    measured=profile['measured_heldout']
    keep=('baseline','selected','random','selected_pf_gain_baseline','selected_pf_gain_random','selected_sarcasm_gain_baseline','selected_sarcasm_gain_random',
          'coherence_loss_vs_baseline','coherence_relative_gate_margin','all_registered_criteria_pass','descriptive_coupling','scope')
    return values,evidence,{k:measured[k] for k in keep if k in measured},profile['limits']


def load_impolite():
    import torch
    reg='results/protocols/author-gcad-impolite-01.json';committed(reg)
    run=ROOT/'outputs/author-gcad-impolite-01';m=read(checked(run/'manifest.json',read(local(reg))['manifest_sha256']))
    meta=read(checked(run/'inputs/vector_metadata.json',m['files']['inputs/vector_metadata.json']))
    values={};source={}
    for name in SHAPES:
        file=local(AUTHOR_ROOT)/author.VECTOR_FILES[name]
        digest=m['external_sha256'].get(str(file),m['external_sha256'].get(str(file.relative_to(ROOT))))
        if digest is None:raise ValueError('author vector absent from registered manifest')
        checked(file,digest);values[name]=torch.load(file,map_location='cpu',weights_only=True);source[name]=digest
    validate_tensors(values,{k:meta[k]['tensor_sha256'] for k in SHAPES})
    result='results/author_gcad_impolite_01.json';committed(result);data=read(local(result))
    if data['status']!='complete' or not data['metrics']['all_contrasts_pass']:raise ValueError('author effect not retained')
    evidence={'source_manifest_sha256':sha(run/'manifest.json'),'initial_registration_sha256':sha(local(reg)),
        'source_pickle_files_sha256':source,'vector_metadata_sha256':sha(run/'inputs/vector_metadata.json'),'aggregate_result_sha256':sha(local(result)),
        'registered_renderings_sha256':m['files']['inputs/rendered.json'],
        'author_commit':m['author_commit'],'input_provenance':'Released author impolite bank; author source/data Apache-2.0, including Persona Vectors attribution in NOTICE.'}
    evidence['extraction_inputs_sha256']={Path(n).name:h for n,h in m['external_sha256'].items() if n.endswith('trait_data_extract/impolite.json')}
    measured={'baseline':data['metrics']['conditions']['baseline'],'selected':data['metrics']['conditions']['gcad'],
        'author_contrasts':data['metrics']['directional_contrasts'],'source_question_coverage':data['source_question_coverage'],'random':'not measured in original author-reference study'}
    limits=['Author-reference first-reply task:60 sampled conversations on13 unique first questions, not our32-item held-out design.',
        'Shorter impolite replies; helpfulness and correctness were not tested. Original7B study has no random arm or newer4B qualification gate.',
        'Qualitative author contrasts passed under registered adaptations; this is not exact numerical paper reproduction or universal rudeness validation.']
    return values,evidence,measured,limits


LOADER='''"""Pinned local-bank loader; no model download, publication or pickle load."""
import hashlib,json,importlib.metadata
from pathlib import Path
import torch
from safetensors.torch import load_file
from .gcad import MultiLayerGatedSteerer,validate_geometry

BASE="Qwen/Qwen2.5-7B-Instruct"
REV="a09a35458c702b33eeacc393d103063234e8bc28"
TRAITS=("profanity","impolite","sarcasm")
SHAPES={"delta":(28,3584),"key":(28,4,128),"demand":(28,)}

def sha(file):
    h=hashlib.sha256()
    with Path(file).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def tensor_sha(v):return hashlib.sha256(v.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def profile_and_tensors(root,trait,*,expected_manifest_sha256):
    root=Path(root)
    if trait not in TRAITS:raise ValueError("only retained individual traits")
    if sha(root/"release_manifest.json")!=expected_manifest_sha256:raise ValueError("pin the immutable bundle manifest hash")
    manifest=json.loads((root/"release_manifest.json").read_text())
    for name in (trait+"/profile.json",trait+"/bank.safetensors","runtime/gcad.py","runtime/loader.py"):
        file=root/name
        if file.is_symlink() or sha(file)!=manifest["files_sha256"][name]:raise ValueError("bundle file hash differs")
    p=json.loads((root/trait/"profile.json").read_text())
    if p["base"]["id"]!=BASE or p["base"]["revision"]!=REV or p["trait"]!=trait:raise ValueError("base model/revision/trait differs")
    c=p["controller"]
    if c["blocks"]!=list(range(9,20)) or c["base_coef"]!={"profanity":5.,"impolite":3.5,"sarcasm":2.}[trait] or c["scale"]!=1.5 or c["inverse"] is not False:raise ValueError("retained fixed controller differs")
    if sha(root/trait/"bank.safetensors")!=p["bank"]["sha256"]:raise ValueError("bank file hash differs from profile")
    values=load_file(str(root/trait/"bank.safetensors"),device="cpu")
    if set(values)!=set(SHAPES):raise ValueError("exact tensor names required")
    for k,v in values.items():
        if tuple(v.shape)!=SHAPES[k] or v.dtype!=torch.float32 or not torch.isfinite(v).all() or tensor_sha(v)!=p["bank"]["tensor_sha256"][k]:raise ValueError("finite FP32 tensor shape/hash differs")
    return p,values

def validate_tokenizer(tokenizer,snapshot,metadata):
    from transformers import AutoTokenizer
    snapshot=Path(snapshot).resolve()
    required={"config.json","generation_config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"}
    if set(metadata)!=required:raise ValueError("complete pinned snapshot metadata required")
    for name,digest in metadata.items():
        if sha(snapshot/name)!=digest:raise ValueError("base config/tokenizer/generation metadata changed")
    if Path(tokenizer.name_or_path).resolve()!=snapshot:raise ValueError("actual tokenizer source differs")
    reference=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
    reference.pad_token=reference.eos_token;reference.padding_side="left"
    if tokenizer.eos_token_id!=151645 or tokenizer.pad_token_id!=151645 or tokenizer.padding_side!="left":raise ValueError("source tokenizer EOS-as-PAD left padding required")
    if reference.eos_token_id!=151645 or reference.pad_token_id!=151645 or reference.padding_side!="left":raise ValueError("reference tokenizer policy differs")
    if tokenizer.get_vocab()!=reference.get_vocab():raise ValueError("actual tokenizer vocabulary differs")
    if tokenizer.chat_template!=reference.chat_template:raise ValueError("actual tokenizer template differs")
    actual=json.loads(tokenizer.backend_tokenizer.to_str());expected=json.loads(reference.backend_tokenizer.to_str())
    for key in ("model","normalizer","pre_tokenizer","post_processor","decoder","added_tokens"):
        if actual.get(key)!=expected.get(key):raise ValueError("actual tokenizer backend differs: "+key)

def validate_base(model,tokenizer,snapshot,profile,*,base_id,revision):
    snapshot=Path(snapshot).resolve();c=model.config;b=profile["base"]
    if base_id!=BASE or revision!=REV or snapshot.name!=REV:raise ValueError("exact base ID/revision/snapshot required")
    if importlib.metadata.version("torch")!="2.6.0+cu124" or importlib.metadata.version("transformers")!="4.52.3":raise ValueError("registered runtime package versions required")
    if Path(c._name_or_path).resolve()!=snapshot or Path(tokenizer.name_or_path).resolve()!=snapshot:raise ValueError("loaded model/tokenizer must identify the checked local snapshot")
    if getattr(c,"_commit_hash",None) not in (None,REV):raise ValueError("loaded base commit differs")
    if (c.model_type,c.num_hidden_layers,c.hidden_size,c.num_attention_heads,c.num_key_value_heads)!=("qwen2",28,3584,28,4):raise ValueError("Qwen2.5-7B native geometry required")
    if c._attn_implementation!="sdpa" or any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError("native BF16 SDPA model required")
    validate_tokenizer(tokenizer,snapshot,b["snapshot_metadata_sha256"])
    validate_geometry(model)

def load_controller(root,trait,model,tokenizer,snapshot,*,base_id,revision,expected_manifest_sha256,zero=False):
    p,v=profile_and_tensors(root,trait,expected_manifest_sha256=expected_manifest_sha256)
    validate_base(model,tokenizer,snapshot,p,base_id=base_id,revision=revision)
    blocks=p["controller"]["blocks"]
    return MultiLayerGatedSteerer(model,{l:v["delta"][l] for l in blocks},{l:v["key"][l] for l in blocks},{l:v["demand"][l].item() for l in blocks},
        layer_indices=blocks,base_coef=0. if zero else p["controller"]["base_coef"],scale=1.5,inverse=False,debug=False)
'''

EXAMPLE='"""Deferred local-only reload/example. Never executed by staging."""\nimport argparse,json\nfrom pathlib import Path\nimport torch\nfrom transformers import AutoModelForCausalLM,AutoTokenizer\nfrom .loader import load_controller,profile_and_tensors,BASE,REV\n\ndef main():\n    a=argparse.ArgumentParser();a.add_argument("--snapshot",required=True);a.add_argument("--trait",choices=["profanity","impolite","sarcasm"],required=True)\n    a.add_argument("--manifest-sha256",required=True);a.add_argument("--device",default="cuda:0");a.add_argument("--zero-selftest",action="store_true");a.add_argument("--prompt",default="Describe a rainy afternoon in two sentences.")\n    args=a.parse_args();root=Path(__file__).resolve().parents[1]\n    p,_=profile_and_tensors(root,args.trait,expected_manifest_sha256=args.manifest_sha256)\n    tok=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True,trust_remote_code=False);tok.pad_token=tok.eos_token;tok.padding_side="left"\n    model=AutoModelForCausalLM.from_pretrained(args.snapshot,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map={"":args.device}).eval()\n    ids=tok.apply_chat_template([{"role":"user","content":args.prompt}],return_tensors="pt",add_generation_prompt=True).to(model.device)\n    context=load_controller(root,args.trait,model,tok,args.snapshot,base_id=BASE,revision=REV,expected_manifest_sha256=args.manifest_sha256,zero=args.zero_selftest)\n    torch.manual_seed(42)\n    with torch.no_grad():\n        if args.zero_selftest:\n            ordinary=model(ids,use_cache=False).logits\n            with context:zero=model(ids,use_cache=False).logits\n            if not torch.equal(ordinary,zero):raise ValueError("real base zero parity failed")\n            print("zero-coefficient parity PASS")\n        else:\n            g=p["decoding"];kwargs={"do_sample":g["do_sample"],"max_new_tokens":g["max_new_tokens"],"repetition_penalty":1.05,"use_cache":True}\n            if g["do_sample"]:kwargs.update(temperature=1.,top_p=.8,top_k=20)\n            with context:out=model.generate(ids,**kwargs)\n            print(tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True))\nif __name__=="__main__":main()\n'


def runtime_files():return {'runtime/gcad.py':local(PORT).read_bytes(),'runtime/loader.py':LOADER.encode(),'runtime/example.py':EXAMPLE.encode(),'runtime/__init__.py':b''}


def scan_json(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN_KEYS:raise ValueError('private raw field forbidden')
        for v in value.values():scan_json(v)
    elif isinstance(value,list):
        for v in value:scan_json(v)
    elif isinstance(value,str) and BAD_TEXT.search(value):raise ValueError('local absolute path or secret-like string forbidden')


def allowlist(public=False):
    return (PUBLIC_EXTRA if public else set()) | {'README.md','LICENSE','NOTICE','release_manifest.json',*runtime_files(),
        *{t+'/'+f for t in TRAITS for f in ('bank.safetensors','profile.json','provenance.json','README.md')}}


def prior_validation():
    return {'staging_01':'Real-7B load guard false positive from tokenizer serialization comparison; base loaded, zero forwards; original tree and log preserved.',
        'model_forwards':0,'zero_selftest_completed':False,
        'staging_01_manifest_sha256':sha(ROOT/'outputs/steering-release-staging-01/release_manifest.json'),
        'log_sha256':sha(ROOT/'outputs/steering-release-validation-01/profanity.log'),
        'differing_keys':['model.merges','model.ignore_merges'],
        'repair':'Fresh local reference from hash-checked snapshot; same EOS-as-PAD/left padding; actual vocabulary/template and semantic backend sections compared.'}


def history_readme(history):
    return ('\n## Prior validation history\n\nStaging01 loaded the base but its tokenizer-serialization guard gave a false\n'
        'positive before any forward: **0 forwards**, zero self-test not reached.\n'
        'The original01 tree/log are preserved; bank tensors and scientific results\n'
        'are unchanged. Staging02 uses the checked fresh-reference tokenizer guard.\n'
        f'01 manifest SHA256: `{history["staging_01_manifest_sha256"]}`.\n'
        f'Preserved failed-log SHA256: `{history["log_sha256"]}`.\n'
        'Real7B reload/zero validation remains deferred until separately released.\n')


def preflight(root,*,public=False):
    root=Path(root);wanted=allowlist(public);actual=set()
    for file in root.rglob('*'):
        if file.is_symlink():raise ValueError('no symlinks in release')
        if file.is_file():
            name=file.relative_to(root).as_posix()
            if name not in wanted or '..' in PurePosixPath(name).parts:raise ValueError('nonallowlisted/pickle/path payload')
            actual.add(name)
            if file.suffix!='.safetensors':
                raw=file.read_text()
                if BAD_TEXT.search(raw):raise ValueError('private path/secret-like text in release')
                if file.suffix=='.json':scan_json(read(file))
                if public and file.suffix in ('.json','.md'):
                    if PROJECT_PATH.search(raw):raise ValueError('private project path in public JSON/Markdown')
                    if file.suffix=='.md':validate_public_links(root,file,raw)
                if file.suffix=='.py':ast.parse(raw)
    if actual!=wanted:raise ValueError('incomplete release allowlist')
    m=read(root/'release_manifest.json')
    if public:validate_public_metadata(root,m)
    elif m.get('private') is not True or m.get('status')!='local_draft_not_uploaded':raise ValueError('explicit local private draft field required')
    history=m.get('prior_validation',{})
    if history.get('model_forwards')!=0 or history.get('zero_selftest_completed') is not False or any(not re.fullmatch('[0-9a-f]{64}',history.get(k,'')) for k in ('staging_01_manifest_sha256','log_sha256')):
        raise ValueError('prior zero-forward guard-failure history required')
    if set(m['files_sha256'])!=actual-{'release_manifest.json'}:raise ValueError('manifest path/inventory differs')
    for name,digest in m['files_sha256'].items():checked(root/name,digest)
    for name,raw in runtime_files().items():
        if (root/name).read_bytes()!=raw:raise ValueError('unreviewed runtime Python modification')
    from safetensors.torch import load_file
    for trait in TRAITS:
        p=read(root/trait/'profile.json');validate_tensors(load_file(str(root/trait/'bank.safetensors'),device='cpu'),p['bank']['tensor_sha256'])
    return {'pass':True,'file_count':len(actual),'manifest_sha256':sha(root/'release_manifest.json')}


def stage():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('staging requires CUDA_VISIBLE_DEVICES empty')
    if DEST.exists() or DEST.is_symlink():raise FileExistsError('fresh staging root only; no overwrite')
    # Resolve/verify every source before writing or unsafe-format deserialization.
    committed(PROFILE_FILES['profanity'])
    model_ref=read(local(PROFILE_FILES['profanity']))['model']['manifest']
    model_manifest=read(checked(local(model_ref['path']),model_ref['sha256']))
    model=next(x for x in model_manifest if x['model']==BASE and x['revision']==REV)
    metadata={r['path']:r['sha256'] for r in model['files'] if r['path'] in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')}
    staged={trait:(load_impolite() if trait=='impolite' else load_source_profile(trait)) for trait in TRAITS}
    history=prior_validation()
    DEST.mkdir(mode=0o700)
    from safetensors.torch import save_file,load_file
    import torch
    for name,raw in runtime_files().items():
        file=DEST/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(raw);file.chmod(0o600)
    for name in ('LICENSE','NOTICE'):shutil.copyfile(local(AUTHOR_ROOT)/name,DEST/name);(DEST/name).chmod(0o600)
    for trait,(values,evidence,measured,limits) in staged.items():
        folder=DEST/trait;folder.mkdir();file=folder/'bank.safetensors'
        save_file({k:v.contiguous() for k,v in values.items()},str(file));file.chmod(0o600);again=load_file(str(file),device='cpu')
        if any(not torch.equal(values[k],again[k]) for k in values):raise ValueError('conversion roundtrip failed')
        hashes={k:tensor_hash(v) for k,v in values.items()}
        profile={'schema_version':1,'kind':'provisional_working_individual_7b_gcad','trait':trait,
            'base':{'id':BASE,'revision':REV,'dtype':'bfloat16','attention':'sdpa','snapshot_metadata_sha256':metadata},
            'controller':{'blocks':list(range(9,20)),'base_coef':COEFS[trait],'scale':1.5,'inverse':False,'gate':'c = 2*b*sigmoid(scale*(demand-mean_demand))','update':'attention_out + c*raw_delta'},
            'bank':{'file':'bank.safetensors','sha256':sha(file),'shapes':{k:list(v.shape) for k,v in values.items()},'tensor_sha256':hashes,'all_layers':28,'dtype':'float32'},
            'decoding':{'do_sample':trait=='impolite','temperature':1. if trait=='impolite' else None,'top_p':.8,'top_k':20,'repetition_penalty':1.05,
                'max_new_tokens':500 if trait=='impolite' else 160,'seed':42,'main_batch_size':64 if trait=='impolite' else 1,'tokenizer_eos_as_pad':151645,'generation_pad':151643,'eos_ids':[151645,151643]},
            'evaluation_design':{'n':60 if trait=='impolite' else 32,'task':'author first-reply sampled conversations,13 unique questions; no random control' if trait=='impolite' else '32 held-out ordinary neutral questions; matched random control'},
            'measured':measured,'limits':limits,'runtime':{'torch':'2.6.0+cu124','transformers':'4.52.3','real_7b_reload_validation':'deferred_until_4b_jobs_release_gpu'}}
        provenance={'source':evidence,'source_tensor_sha256':hashes,'converted_tensor_sha256':{k:tensor_hash(v) for k,v in again.items()},'exact_fp32_roundtrip':True,
            'safetensors_sha256':sha(file),'author_repository':'NihiI-obstat/Gated-Cropped-Attention-Delta','author_revision':'6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de',
            'license':'Apache-2.0','notice_sha256':sha(DEST/'NOTICE'),'runtime_port_source_sha256':sha(local(PORT))}
        save(folder/'profile.json',profile);save(folder/'provenance.json',provenance)
        (folder/'README.md').write_text(f'# {trait}: provisional retained7B controller\n\nFixed b={COEFS[trait]}, scale1.5, blocks9–19. Raw dynamic GCAD, no weight edit.\nDesign n={profile["evaluation_design"]["n"]}: {profile["evaluation_design"]["task"]}.\nSee profile.json for measured outcomes and substantial limitations.\nStrength is bank/task/decoding specific; no cross-trait dose equivalence.\n4B pending; no retained refusal controller. Real7B reload is deferred.\n')
    (DEST/'README.md').write_text('''---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
---
# Provisional individual7B GCAD research controls — local staging only

This draft contains retained profanity, author impolite and sarcasm banks.
No base weights, physics adapter, raw answers, judge notes, token arrays or
pickle files are included. No upload or visibility action has occurred.
`private:true` is local draft metadata, not a new approval/permission flow.
4B is pending; there is no retained refusal controller. Do not infer model-size
causality, joint independence or disentanglement. Nulls are not hidden.

PF:31.125 vs0 trait; coherence82.6875 vs85.34375;14/32 lexical hits vs0.
SARC:63.78125 vs0.0625;29/32 clear; coherence70.71875 vs89.59375.
Its18.875-point loss leaves only1.125 margin; descriptive impoliteness38.75.
Impolite:77.3667 vs1.6167 numeric trait; coherence75.9 vs82.6833;
60 first replies on13 unique questions, not the PF/SARC held-out design.
It has no random arm; shorter replies and helpfulness were not validated.
See each profile for exact counts, truncation, random results and limitations.
The scales use one Claude judge family; coherence is not accuracy/helpfulness.
PF/SARC extraction inputs were reconstructed/generated and judged by that family.
Offensive style is intentional; research use only. Strengths are not comparable.

Runtime is the byte-identical reviewed Qwen3 GCAD port's Qwen2-compatible path;
its tinyQwen2 source parity/zero tests do not replace a real7B reload check.
Torch2.6.0/Transformers4.52.3 are the measured basis. The loader validates the
pinned local base revision, tokenizer metadata, tensor dimensions and hashes.
Tokenizer structure is compared with a fresh local reference loaded only after
its snapshot files pass the pinned hashes; EOS-as-PAD and left padding match.
All model/normalizer/pre-tokenizer/post-processor/decoder/added-token structures,
vocabulary and chat template must agree. No cache files are altered.
It cannot attest that a caller secretly modified in-memory model parameters.

Runtime requirements: torch2.6.0+cu124, transformers4.52.3, safetensors and tqdm.
Callers supply a tokenizer with EOS-as-PAD and left padding, as in the example.
PF and SARC are project-produced banks under this bundle's Apache-2.0 licence.
The upstream NOTICE is preserved verbatim.

After a separately authorized real7B reload, from the bundle root invoke:
`python -m runtime.example --snapshot YOUR_PINNED_LOCAL_SNAPSHOT --trait profanity --manifest-sha256 PINNED_BUNDLE_MANIFEST_SHA --zero-selftest`.
No network is needed: both base and tokenizer loads are local-files-only.
The snapshot basename must equal the pinned upstream commit. No model code
from a remote repository is executed. Generation example is a convenience,
not a claim that a new prompt matches the published small evaluation design.

LICENSE and upstream NOTICE are copied verbatim. The NOTICE credits Persona
Vectors for source data/code. The runtime port is derived Apache-2.0 code;
its changes are head geometry/norm support and fail-closed audit instrumentation.
'''+history_readme(history))
    files={p.relative_to(DEST).as_posix():sha(p) for p in DEST.rglob('*') if p.is_file()}
    save(DEST/'release_manifest.json',{'schema_version':1,'private':True,'status':'local_draft_not_uploaded','base_id':BASE,'revision':REV,
        'files_sha256':files,'staging_script_sha256':sha(Path(__file__)),'source_scope_review_sha256':sha(local('results/reviews/steering_release_scope_review.md')),
        'prior_validation':history,
        'gpu_or_model_or_native_or_hf_calls':0,'real_7b_reload_validation':'deferred; CPU tensor conversion and tiny tests only'})
    result=preflight(DEST)
    # Recheck every loaded source after conversion; do not repair on mismatch.
    for trait in TRAITS:
        after=load_impolite() if trait=='impolite' else load_source_profile(trait)
        if any(not torch.equal(after[0][k],staged[trait][0][k]) for k in SHAPES):raise ValueError('source bank changed during staging')
    return result


def resolve_source_ref(source_ref):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',source_ref):raise ValueError('explicit simple Git tag required')
    ref='refs/tags/'+source_ref
    commit=subprocess.run(['git','rev-parse','--verify',ref+'^{commit}'],cwd=ROOT,capture_output=True,check=True).stdout.decode().strip()
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('resolved source commit invalid')
    for name in ('scripts/stage_steering_release.py','tests/test_stage_steering_release.py',RELOAD_RECORD,
                 FOURB_RESULT,FOURB_CLOSEOUT,'docs/refusal_reduction_execution.md','docs/refusal_headroom_closeout.md'):
        data=subprocess.run(['git','show',ref+':'+name],cwd=ROOT,capture_output=True,check=True).stdout
        if data!=local(name).read_bytes():raise ValueError('source tag does not bind current committed release inputs: '+name)
    return {'tag':source_ref,'commit':commit}


def validated_parent():
    if sha(DEST/'release_manifest.json')!=VALIDATED_02_SHA:raise ValueError('reviewed staging02 manifest differs')
    preflight(DEST)
    return read(DEST/'release_manifest.json')


def reload_proof(parent):
    record=read(local(RELOAD_RECORD))
    if (record['status']!='real_7b_zero_reload_checks_passed' or record['bundle_manifest_sha256']!=VALIDATED_02_SHA
        or record['model']!={'id':BASE,'revision':REV,'dtype':'bfloat16','attention':'sdpa'}
        or record['model_forwards']!=6 or record['generated_answers']!=0 or set(record['traits'])!=set(TRAITS)):
        raise ValueError('real three-controller staging02 zero validation required')
    for trait in TRAITS:
        row=record['traits'][trait]
        if row['status']!='passed' or row['exit_code']!=0 or row['ordinary_and_zero_logits_exact_equal'] is not True or row['forwards']!=2:
            raise ValueError('zero validation trait receipt differs')
    if record['runtime_files_sha256']!={Path(n).name:h for n,h in parent['files_sha256'].items() if n.startswith('runtime/')}:
        raise ValueError('validated runtime hash basis differs')
    return {'method':'CPU byte identity of the inference core to the real-zero-validated staging02 bundle',
        'record':Path(RELOAD_RECORD).name,'record_sha256':sha(local(RELOAD_RECORD)),
        'validated_parent_manifest_sha256':VALIDATED_02_SHA,'traits_passed':list(TRAITS),
        'staging02_real_model_forwards':6,'staging02_generated_answers':0,'public03_gpu_rerun':False,
        'scope':'Operational zero-coefficient validation on staging02, not a new efficacy benchmark or a claimed staging03 GPU rerun.'}


def core_names():
    return allowlist()-{'release_manifest.json','README.md',*[t+'/README.md' for t in TRAITS]}


def validate_public_metadata(root,m):
    if m.get('private') is not False or m.get('status')!='public_release' or m.get('repo_id')!=PUBLIC_REPO:raise ValueError('exact public repository/visibility/status required')
    source=m.get('source_git',{})
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',source.get('tag','')) or not re.fullmatch('[0-9a-f]{40}',source.get('commit','')):raise ValueError('source Git tag/commit required')
    if m.get('builder_sha256')!=sha(Path(__file__)):raise ValueError('public builder hash differs')
    parent=validated_parent()
    if m.get('reviewed_parent_manifest_sha256')!=VALIDATED_02_SHA or m.get('reviewed_parent_files_sha256')!=parent['files_sha256']:
        raise ValueError('reviewed parent hashes differ')
    if m.get('validation')!=reload_proof(parent) or m.get('prior_validation')!=parent['prior_validation']:
        raise ValueError('validation/repair history differs')
    expected={n:parent['files_sha256'][n] for n in core_names()}
    if m.get('inference_core_sha256')!=expected:raise ValueError('inference core map differs')
    for name,digest in expected.items():
        if sha(root/name)!=digest or (root/name).read_bytes()!=(DEST/name).read_bytes():raise ValueError('public inference core differs from validated02')
    if read(root/'related_results/qwen3_4b_fixed_transfer_01.public.json')!=public_result_derivative():raise ValueError('public aggregate derivative differs')
    if (root/'related_results/qwen3_4b_fixed_transfer_closeout.md').read_text()!=public_closeout_derivative():raise ValueError('public closeout derivative differs')
    if (root/'METHODS.md').read_text()!=methods_text(source):raise ValueError('public method notes differ')
    for name in ('README.md',*[t+'/README.md' for t in TRAITS]):
        card=(root/name).read_text().lower()
        if any(v in card for v in ('local_draft_not_uploaded','private:true','private: true','research use only','research-use-only')):
            raise ValueError('obsolete draft/visibility/licence-restriction wording in public card')


def validate_public_links(root,file,text):
    from urllib.parse import urlsplit,unquote
    allowed=('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct',
        'https://huggingface.co/'+PUBLIC_REPO,
        'https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta',
        'https://github.com/safety-research/persona_vectors')
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
        u=urlsplit(target)
        if u.scheme or u.netloc:
            if not any(target==p or target.startswith(p+'/') or target.startswith(p+'#') for p in allowed):raise ValueError('unavailable/unreviewed public link')
        else:
            dest=(file.parent/unquote(u.path)).resolve()
            if not dest.is_relative_to(root.resolve()) or not dest.is_file():raise ValueError('relative Markdown link must target a payload file')
    if 'github.com/RenormalizationTech/tuneLLM' in text:raise ValueError('private project GitHub link forbidden')


def public_result_derivative():
    original=read(checked(local(FOURB_RESULT),FOURB_RESULT_SHA))
    if original['status']!='complete' or any(v['gates']['all_pass'] for v in original['traits'].values()):raise ValueError('expected closed zero-quality-qualified4B outcome')
    result=json.loads(json.dumps(original))
    for key in ('runtime_config','audit'):
        result['native_identity'][key]=Path(result['native_identity'][key]).name
    result['source_private_record_sha256']=FOURB_RESULT_SHA
    result['derivative_scope']='Only two native runtime/audit references reduced to basenames; their existing hashes and all scientific numbers/decisions retained. Full private pipeline and item data are not included.'
    return result


def public_closeout_derivative():
    text=local(FOURB_CLOSEOUT).read_text()
    text=text.replace('All 4B outcomes are released as aggregates on GitHub','All 4B outcomes are released as aggregates in this bundle (`related_results/`)')
    text=text.replace('`'+FOURB_RESULT+'`','[public aggregate](qwen3_4b_fixed_transfer_01.public.json)')
    def artifact(match):
        name=match[1]
        return '`'+Path(name).name+'` (private provenance artifact SHA256: `'+sha(local(name))+'`)'
    text=re.sub(r'`((?:configs|outputs|scripts|docs|results)/[^`]+)`',artifact,text)
    def link(match):
        label,target=match[1],match[2]
        if target=='qwen3_4b_fixed_transfer_01.public.json':return match[0]
        return ('private protocol (not included)' if label=='plan' else 'private interpretation review (not included)' if label=='interpretation' else label+' (private source; not included)')
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    return ('> Public derivative of the private project closeout; source SHA256: `'+sha(local(FOURB_CLOSEOUT))+'`. Full private protocols/reviews are not included. See [public method notes](../METHODS.md#qwen3-4b-fixed-transfer) and the included aggregate.\n\n'+text)


def methods_text(source):
    return f'''# Methods and related results

## Artifact and reproducibility scope

This package contains dynamic GCAD inference banks, the reviewed standalone
runtime and aggregate outcomes. It contains no base weights, LoRA adapter,
private physics adapter, exact custom evaluation questions, generation prompts,
answers, grades or full experimental pipeline. The original project repository
is private. Its tag `{source['tag']}` and commit `{source['commit']}` are opaque
provenance identifiers, not public source links. Inference is fully runnable
with this package and the pinned local base; reproducing the full private
measurement pipeline is outside this release.

The original public author sources are
[GCAD](https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta/tree/6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de) and
[Persona Vectors](https://github.com/safety-research/persona_vectors/tree/b8e0f044fe2410a6fad579f38324f03f13b4e917).
Apache-2.0 LICENSE and author NOTICE are preserved verbatim.

## Dynamic inference control

The exact base is Qwen/Qwen2.5-7B-Instruct revision `{REV}`, native BF16/SDPA,
Python3.11, torch2.6.0+cu124, Transformers4.52.3 and Accelerate1.7.0. The bank contains all28 layers:
raw delta[28,3584], post-RoPE key[28,4,128], mean demand[28], stored FP32.
Blocks9..19 are hooked. Per layer/token the native attention output receives
`2*b*sigmoid(1.5*(demand-mean_demand))*delta`, inverseFalse. There is no weight
edit. The runtime port's Qwen2 path retains the pinned author arithmetic.
PF uses b5, impolite3.5, SARC2. These are bank/task-specific settings, not
comparable doses, independent components or jointly validated controls.

PF/SARC use project-reconstructed five instruction pairs and20 fitting
questions, fresh model elicitation/filtering and system-attention extraction.
The same Claude model family generated/reviewed/judged those artifacts.
Impolite uses the author's released Qwen7 bank and broad impoliteness construct.
Trait and coherence were scored separately0..100 with tools-off pinned Claude;
semantic trait REFUSAL contributes0 to conservative fixed-denominator means.
Coherence is not factual accuracy or helpfulness. The impolite author rule uses
numeric means excluding REFUSAL and a qualitative+30 contrast; its original
study had no random arm or the later4B qualification guards.

PF and SARC each used32 ordinary held-out questions, greedy B1 cap160 seed42.
Impolite used60 sampled first replies on13 unique first questions, temperature1,
cap500 B64 seed42. Both retain repetition penalty1.05, top-p.8/top-k20 where
sampling applies. EOS-as-PAD/left padding match the measured tokenizer setup.
PF/SARC random controls match raw delta norms/sites/keys/demand, not realized
shift energy. See the archival per-trait profiles for exact counts and limits.

## Operational validation and repair history

Staging01 loaded the base but rejected a tokenizer serialization comparison
before any model forward; no zero self-test ran. The snapshot stored151387
BPE merges as strings; the loaded backend represented identical ordered pairs
and added default ignore_mergesFalse. The repaired guard hashes metadata first,
loads a fresh local tokenizer reference with EOS-as-PAD/left padding, and
compares vocabulary, template and all semantic backend sections. Injected
merge, normalizer, added-token, template and padding changes are rejected.

The repaired staging02 passed one ordinary-versus-zero logit test for each
controller:3 real pinned base reloads,6 forwards,0 generated answers. This
public package's inference core is CPU byte-identical to that validated02 core.
There was no staging03 GPU rerun, no new efficacy test and no new research call.
The release manifest records the02 validation receipt hash, parent hashes,
repair history and byte-identity proof. Historical per-trait profile fields
saying validation was deferred are archival snapshots; the current manifest
and this card supersede their operational availability statement, not their
scientific measurements.

## Qwen3-4B fixed transfer

[Public aggregate](related_results/qwen3_4b_fixed_transfer_01.public.json) and
[closeout](related_results/qwen3_4b_fixed_transfer_closeout.md) are sanitized
derivatives, with their private original hashes recorded. The aggregate keeps
all numeric values and decisions, reducing two private runtime/audit references
to basenames plus their original hashes. No raw item data is included.

All3 style traits were strongly induced;0/3 met quality qualification.
PF: trait37.34375/coherence20.5625/invalid6; failed absolute/relative coherence
and invalid gates. SARC:56.125/coherence63.90625; missed relative coherence by
3.09375. Impolite:80.18333/coherence52.01667; author-effect rule passed but the
new qualification missed relative coherence by5.30. Fresh4B banks, fixed
proportional blocks12..25 and the carried-over b values were used, with no
retuning. No4B bank or new4B steering repository is published. Outcomes are
model-specific; differences between checkpoints do not establish an effect of
model size, and a4B quality failure does not show that the method fails on small
models. This was not a failure to induce the traits.

## Refusal closures and private adapter

No refusal controller is retained or packaged. The conventional4B record is
carried forward: fixed subtraction block23, coefficient-.5, reduced source-
unsafe refusals29->22/30 and increased valid full answers1->7/30, but benign
usefulness fell29/42->23/42 and the quality rule failed. All-block projection
was not selected (6/30 valid answers in development). These are historical
aggregate results, not a new boundary-removal experiment.
The7B political screen found coverage-mode headroom but no explicit-refusal
headroom on the target baseline; its extension was closed for priority before
fit/steering. That closure is not a method null or hard-refusal reduction.
The physics adapter remains private and untouched. The separate original-
Mistral rewriting reference is a different model/task, not a Qwen bank here.

## Limits

One judge family, small designs, reconstructed artifacts, truncation and source
adaptations limit generalization. SARC's18.875-point coherence loss leaves only
1.125 margin within its relative gate. Its descriptive impoliteness mean38.75,
within-selected r~.6371 and pooled condition-confounded r~.9319 are not causal or
independence evidence. Impolite produced much shorter replies; helpfulness was
not established. Intentional profanity/impoliteness are properties of these
style controls. No joint, disentanglement, universal quality or capability claim.
'''


def public_card(source,trait=None):
    header=('---\nlicense: apache-2.0\nbase_model: '+BASE+'\n---\n') if trait is None else ''
    methods='METHODS.md' if trait is None else '../METHODS.md'
    if trait:
        parent=read(DEST/trait/'profile.json');n=parent['evaluation_design']['n']
        return (f'# {trait}: Qwen2.5-7B GCAD controller\n\nFixed b = {COEFS[trait]}, scale 1.5, blocks 9..19, inverse False.\n'
            f'Its own evaluation design has n = {n}; see [methods and related results]({methods}).\n'
            'The unchanged profile/provenance files are archival snapshots. Current\n'
            'manifest validation supersedes their historical deferred-validation field.\n'
            'The bank/profile/runtime bytes equal the repaired 02 package with3 real zero\n'
            'checks (6 forwards). No staging 03 GPU rerun is claimed.\n'
            'Coherence is not accuracy/helpfulness; strengths are not comparable across\n'
            'traits. No 4B bank or refusal controller is included. Apache-2.0.\n')
    return header+f'''# Qwen2.5-7B GCAD steering controls

Dynamic inference-time banks for individual profanity, author impoliteness and
sarcasm controls on the pinned
[Qwen2.5-7B-Instruct base](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
These are not base-model weights or LoRA adapters. No private physics adapter,
raw answers, grader reasons, token arrays or full private project code/data is
included. The original project is private; source tag `{source['tag']}` and
commit `{source['commit']}` are provenance identifiers, not access-required links.

| 7B controller | Own evaluation design | Trait baseline → GCAD (random) | Coherence baseline → GCAD (random) |
|---|---|---|---|
| PF, b = 5 | 32 ordinary held-out questions, greedy cap 160 | 0 → 31.125 (0); 14/32 lexicon hits | 85.34375 → 82.6875 (85.6875) |
| SARC, b = 2 | 32 ordinary held-out questions, greedy cap 160 | .0625 → 63.78125 (0); 29/32 ≥ 50 | 89.59375 → 70.71875 (87.71875) |
| Impolite, b = 3.5 | 60 author first replies on 13 unique questions, sampled cap 500 | 1.61667 → 77.36667 (not measured) | 82.68333 → 75.9 (not measured) |

The designs and gates differ. Impolite passed its author qualitative contrasts,
with no original random arm; PF/SARC passed their registered held-out gates.
SARC has a substantial 18.875-point coherence loss and only 1.125 points of gate margin. Its
impoliteness mean 38.75 and correlations are descriptive, not independence or
causality evidence. One Claude judge family and reconstructed PF/SARC inputs
limit validation; coherence is not correctness or helpfulness. These controls
intentionally induce potentially offensive style. Apache-2.0 is the licence;
no additional use restriction is imposed here.

## Validation

The inference core is byte-identical to validated staging 02: 3 real pinned
base zero-coefficient checks, 6 forwards total. No staging 03 GPU rerun is
claimed. Current manifest validation supersedes historical deferred fields in
archival profiles. [METHODS](METHODS.md) records the repair history and scope.

## Pinned local inference

Requirements: Python3.11, torch==2.6.0+cu124, transformers==4.52.3,
accelerate==1.7.0, safetensors and tqdm.
Use base `{BASE}` revision `{REV}` from a local snapshot whose basename equals
that commit. The example uses local-files-only and no remote model code.
Tokenizer must use EOS151645 as PAD and left padding. Base weights are fetched
separately from their upstream repository, not supplied here.

Choose an immutable **HF_REV** of this bank repository. Read that revision's
commit message `Release manifest SHA256: H` and use H as the expected
manifest hash. This verifies file consistency for the pinned revision you choose;
it does not independently prove who published them. Publisher authenticity
rests on the HF account. Download this repository at HF_REV, never an unpinned moving
revision. No private GitHub record is needed. Use real control files (the loader
rejects control symlinks); a pinned local-directory download provides them:

```python
import os
from huggingface_hub import snapshot_download
PINNED_HF_SHA = os.environ["HF_REV"]
snapshot_download("RenormalizationTech/qwen2.5-7b-gcad-steering-controls",
                  revision=PINNED_HF_SHA, local_dir="controls")
```

Then from the downloaded bundle root, with H from that commit message in
BANK_MANIFEST_SHA and the pinned upstream base snapshot in LOCAL_BASE_SNAPSHOT:

```bash
cd controls
python -m runtime.example --snapshot "$LOCAL_BASE_SNAPSHOT" --trait profanity --manifest-sha256 "$BANK_MANIFEST_SHA" --zero-selftest
python -m runtime.example --snapshot "$LOCAL_BASE_SNAPSHOT" --trait profanity --manifest-sha256 "$BANK_MANIFEST_SHA" --prompt "Describe a rainy afternoon in two sentences."
```

Use `impolite` or `sarcasm` for the other isolated banks. The saved settings are
bank/task/decoding-specific, not comparable doses or jointly validated controls.
Inference is fully runnable; exact custom questions, prompts, answers, grades
and the full experimental pipeline remain private.

## Related outcomes

[Methods, source attribution and refusal closures](METHODS.md),
[4B aggregate](related_results/qwen3_4b_fixed_transfer_01.public.json), and
[4B closeout](related_results/qwen3_4b_fixed_transfer_closeout.md) are included.
All 3 fixed-transfer 4B traits were induced, but none met the quality gates. No 4B bank or
new 4B HF steering repository is published, and there is no retained refusal
controller. Outcomes are model-specific; checkpoint differences do not establish
a model-size effect. No retuning, joint or independence claim. The physics
adapter remains private and untouched. Original GCAD and Persona Vectors
attribution is preserved in LICENSE/NOTICE; public author sources are listed
in the included method notes.
'''


def finalize_public(source_ref):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU finalization requires CUDA hidden')
    if PUBLIC_DEST.exists() or PUBLIC_UPLOAD_MANIFEST.exists():raise FileExistsError('fresh03 and outside publication manifest only')
    source=resolve_source_ref(source_ref);parent=validated_parent();validation=reload_proof(parent)
    related=public_result_derivative();closeout=public_closeout_derivative()
    PUBLIC_DEST.mkdir(mode=0o700)
    for name in core_names():
        dest=PUBLIC_DEST/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/name,dest);dest.chmod(0o600)
    (PUBLIC_DEST/'README.md').write_text(public_card(source));(PUBLIC_DEST/'METHODS.md').write_text(methods_text(source))
    for trait in TRAITS:(PUBLIC_DEST/trait/'README.md').write_text(public_card(source,trait))
    save(PUBLIC_DEST/'related_results/qwen3_4b_fixed_transfer_01.public.json',related)
    (PUBLIC_DEST/'related_results/qwen3_4b_fixed_transfer_closeout.md').write_text(closeout)
    files={p.relative_to(PUBLIC_DEST).as_posix():sha(p) for p in PUBLIC_DEST.rglob('*') if p.is_file()}
    save(PUBLIC_DEST/'release_manifest.json',{'schema_version':1,'private':False,'status':'public_release','repo_id':PUBLIC_REPO,'base_id':BASE,'revision':REV,
        'source_git':source,'source_visibility':'private provenance identifiers; inference runtime is included here',
        'builder_sha256':sha(Path(__file__)),'files_sha256':files,'reviewed_parent_manifest_sha256':VALIDATED_02_SHA,
        'reviewed_parent_files_sha256':parent['files_sha256'],'inference_core_sha256':{n:parent['files_sha256'][n] for n in core_names()},
        'validation':validation,'real_7b_reload_validation':'real staging02 zero checks passed; public03 inference core matches exactly; no03 GPU rerun',
        'prior_validation':parent['prior_validation'],'related_results_provenance':{'aggregate_original_sha256':FOURB_RESULT_SHA,'closeout_original_sha256':sha(local(FOURB_CLOSEOUT)),
            'scope':'Sanitized derivatives only; original private scientific records unchanged.'},
        'profile_metadata_policy':'Byte-identical archival profiles/provenance retained; current manifest/card validation supersedes old deferred fields.'})
    result=preflight(PUBLIC_DEST,public=True)
    save(PUBLIC_UPLOAD_MANIFEST,{'repo_id':PUBLIC_REPO,'private':False,
        'files':{p.relative_to(PUBLIC_DEST).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in PUBLIC_DEST.rglob('*') if p.is_file()}})
    location=lambda file:str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else file.name
    return {**result,'folder':location(PUBLIC_DEST),'publication_manifest':location(PUBLIC_UPLOAD_MANIFEST),
            'publication_manifest_sha256':sha(PUBLIC_UPLOAD_MANIFEST),'upload_performed':False}

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['stage','verify','public','verify-public']);parser.add_argument('--source-ref');args=parser.parse_args()
    if args.stage=='public' and not args.source_ref:parser.error('public requires --source-ref naming an existing committed tag')
    value=finalize_public(args.source_ref) if args.stage=='public' else preflight(PUBLIC_DEST,public=True) if args.stage=='verify-public' else stage() if args.stage=='stage' else preflight(DEST)
    print(json.dumps(value))

if __name__=='__main__':main()

"""Bounded original-Mistral SAPS using actual author helpers and native TL2.17."""
import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import warnings

from scripts import author_saps_method_common as p
from scripts import author_gcad_impolite as fbase
from scripts import author_replication_audit as audit
from scripts import tone_judge as f
from scripts.common import path,read_json
from scripts.save_nf4 import stream_hash,tensor_hash

RUN='outputs/author-saps-mistral-01'
CONFIG='configs/author_saps_method.json'
PIN='c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc'
save=f.save
BUDGETS={'train_prompt_forwards':1500,'test_projection_forwards':450,'mechanics_projection_forwards':4,'logit_forwards':2,'generations':1810}


class PreconditionWait(RuntimeError):pass
class StructuralError(ValueError):pass


def guard(run):
    if run!=path(RUN) or run.is_symlink() or run.resolve()!=path(RUN):raise ValueError('fixed new SAPS root only')


def validate_config(cfg):
    if cfg!=read_json(CONFIG):raise ValueError('active and frozen config differ')
    fixed={'run':RUN,'result':'results/author_saps_mistral_01.json','model_id':'mistralai/Mistral-7B-Instruct-v0.1',
        'revision':'ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca','gpu_uuid':'GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b',
        'minimum_free_mib':24000,'maximum_utilization_percent':10,'seed':42,'block':12,'dtype':'float16','instruction_mode':'random','cap':60,'probe_cap':16,
        'arms':list(p.ARMS),'plan_seed':8501,'maximum_generations':1810,'maximum_prompt_forwards':1950,
        'maximum_mechanics_projection_forwards':4,'maximum_logit_forwards':2,'maximum_native_starts':180}
    if any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in fixed.items()):raise ValueError('fixed source/runtime/budgets differ')
    if any(Path(v).is_absolute() or '..' in Path(v).parts for v in cfg.values() if isinstance(v,str)):raise ValueError('project-local registered paths required')


def role(s,name):
    if Path(sys.executable).absolute()!=path(s.cfg[name+'_python']).absolute():raise ValueError('registered '+name+' interpreter required')
    if name=='gpu':
        if sys.version_info[:2]!=(3,11):raise ValueError('isolated Python3.11 required')
        for key,want in {'torch':'2.6.0+cu124','transformer-lens':'2.17.0','transformers':'4.57.6','numpy':'1.26.4'}.items():
            if importlib.metadata.version(key)!=want:raise ValueError('author runtime version changed')
        if fbase.packages()!=s.m['gpu_packages']:raise ValueError('frozen environment inventory differs')


def source_files(cfg):
    manifest=read_json(cfg['source_manifest']);wanted=['src/steering.py','src/generate.py','src/compute_direction.py','data/sarcasm/train.json','data/sarcasm/val.json','data/sarcasm/test.json','prompts/sarcasm.json','LICENSE']
    selected=[r for r in manifest['files'] if r['repo']=='SharathHN/saps-steering' and r['path'] in wanted]
    if len(selected)!=len(wanted) or any(r['revision']!=PIN for r in selected):raise ValueError('pinned author archive incomplete')
    tree=audit.tree('saps');result={cfg['source_manifest']:stream_hash(path(cfg['source_manifest']))}
    for row in selected:
        file=path(cfg['author_root'])/row['path'];sha=audit.verify_blob(file,tree[row['path']])
        if sha!=row['sha256']:raise ValueError('pinned author SHA changed')
        result[str(file)]=sha
    return result


def prepare():
    run=path(RUN);guard(run)
    if run.exists():raise FileExistsError('one fresh source-method run only')
    cfg=read_json(CONFIG);validate_config(cfg);env=read_json(cfg['environment']);s=type('Runtime',(),{'cfg':cfg,'m':{'gpu_packages':env['packages']}})();role(s,'gpu')
    external=source_files(cfg);author=p.author_module(path(cfg['author_root'])/'src/steering.py')
    data={name:read_json(path(cfg['author_root'])/'data/sarcasm'/(name+'.json')) for name in ('train','val','test')};prompts=read_json(path(cfg['author_root'])/'prompts/sarcasm.json')
    if [len(data[k]) for k in ('train','val','test')]!=[500,50,450] or any(len(set(v))!=len(v) or any(not isinstance(x,str) or not x.strip() for x in v) for v in data.values()):raise ValueError('exact source split counts/uniqueness required')
    if any(set(data[a])&set(data[b]) for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('source splits overlap')
    records=[r for r in read_json(cfg['model_manifest']) if r['model']==cfg['model_id']]
    if len(records)!=1 or records[0]['revision']!=cfg['revision'] or records[0]['snapshot']!=cfg['snapshot'] or records[0]['weights_verified'] is not True or records[0]['remote_code_executed'] is not False:raise ValueError('canonical original-Mistral model record differs')
    model_hashes={str(path(cfg['snapshot'])/r['path']):r['sha256'] for r in records[0]['files']}
    runtime=read_json(cfg['judge_runtime_config']);record=read_json(runtime['judge_runtime_record']);checked=f.validate_audit(runtime['judge_audit'])
    if record['binary_sha256']!=fbase.BINARY_SHA:raise ValueError('native pin differs')
    extra=[cfg[k] for k in ('protocol','design_review','environment','model_manifest','judge_runtime_config','rubric')]
    extra += [runtime['judge_runtime_record'],runtime['judge_audit'],str(path(runtime['judge_audit']).parent/'attempt.json')]
    external.update({n:stream_hash(path(n)) for n in extra});external[record['binary']]=fbase.BINARY_SHA
    for name in ('gpu_python','judge_python'):external[str(path(cfg[name]).resolve(strict=True))]=stream_hash(path(cfg[name]))
    # Bind installed source/runtime, without importing/loading target weights.
    import transformer_lens
    library=Path(transformer_lens.__file__).parent
    external.update({str(file):stream_hash(file) for file in library.rglob('*.py')})
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    rendered={name:[author.build_chat_prompt(tok,x,prompts['plain']) for x in data[name]] for name in ('train','val','test')}
    rubric=read_json(cfg['rubric'])['reference']
    sources=fbase.closure(['scripts/author_saps_method.py','scripts/author_saps_method_common.py','scripts/author_saps_method_runbook.md',CONFIG])
    hashes={n:stream_hash(path(n)) for n in sources};fbase.check_hashes([external,hashes,model_hashes])
    run.mkdir(mode=0o700);save(run/'config.json',cfg)
    for name,value in [('data',data),('prompts',prompts),('rendered',rendered),('rubric',rubric),('plans',p.plans()),
                       ('mechanics',[{'id':'validation-0','sentence':data['val'][0]},{'id':'validation-1','sentence':data['val'][1]}])]:save(run/'inputs'/(name+'.json'),value)
    for name in sources:
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(name),target)
    save(run/'manifest.json',{'files':{str(file.relative_to(run)):stream_hash(file) for file in run.rglob('*') if file.is_file()},'sources':hashes,'external_sha256':external,
        'model_sha256':model_hashes,'gpu_packages':env['packages'],'binary':record['binary'],'binary_sha256':fbase.BINARY_SHA,'cli_version':checked['cli_version'],'no_live_calls':True})
    fbase.check_hashes([external,hashes,model_hashes])


class Context:
    def __init__(self):
        self.run=path(RUN);guard(self.run);self.cfg=read_json(self.run/'config.json');validate_config(self.cfg);self.m=read_json(self.run/'manifest.json')
    def verify(self,models=False):
        maps=[self.m['sources'],self.m['external_sha256'],{str(self.run/n):h for n,h in self.m['files'].items()}]
        if models:maps.append(self.m['model_sha256'])
        fbase.check_hashes(maps)
        if read_json(self.run/'inputs/plans.json')!=p.plans():raise ValueError('frozen blinded plans differ')
        return self
    def registered(self):
        for name in (*self.m['sources'],self.cfg['protocol'],self.cfg['design_review'],self.cfg['initial_registration']):fbase.committed(name)
        if read_json(self.cfg['initial_registration'])['manifest_sha256']!=stream_hash(self.run/'manifest.json'):raise ValueError('exact committed registration required')


def rng_record():
    import torch
    return {'python_random_state_sha256':p.canonical(repr(random.getstate())),'torch_cpu_rng_sha256':tensor_hash(torch.get_rng_state())}


def pinned_load(cfg,tl_cls,tokenizer_cls):
    """Official ID + pinned local cache; no processing/dtype-default replacement."""
    import torch
    # Tokenizer initialization is audited for RNG neutrality, not silently reset.
    random.seed(42);torch.manual_seed(42);before=rng_record()
    tok=tokenizer_cls.from_pretrained(path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    if rng_record()!=before:raise ValueError('explicit tokenizer load consumed RNG; discuss source-order adaptation before use')
    model=tl_cls.from_pretrained(cfg['model_id'],device='cuda',torch_dtype=torch.float16,tokenizer=tok,
                                revision=cfg['revision'],cache_dir=str(path('.hf/hub')),local_files_only=True,trust_remote_code=False)
    model.eval();return model


def load_model(s):
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HOME=str(path('.hf')),HF_HUB_CACHE=str(path('.hf/hub')),
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    if torch.cuda.device_count()!=1:raise ValueError('one masked physical GPU0 required')
    model=pinned_load(s.cfg,HookedTransformer,AutoTokenizer)
    if model.cfg.n_layers!=32 or model.cfg.d_model!=4096 or model.cfg.dtype!=torch.float16 or any(param.device!=torch.device('cuda:0') for param in model.parameters()):raise ValueError('original native TL32x4096 FP16 single-GPU representation required')
    return model


def runtime(model):
    import torch
    from transformer_lens import HookedTransformer
    signature=inspect.signature(HookedTransformer.from_pretrained);names=('fold_ln','center_writing_weights','center_unembed','fold_value_biases','refactor_factored_attn_matrices')
    fingerprints={name:tensor_hash(param) for name,param in model.named_parameters() if name in ('blocks.12.attn.W_Q','blocks.12.attn.W_K','blocks.12.attn.W_V','blocks.12.attn.W_O')}
    basis={'cfg_dtype':str(model.cfg.dtype),'normalization_type':model.cfg.normalization_type,'ln_final_class':type(model.ln_final).__name__,
           'block12_ln1_class':type(model.blocks[12].ln1).__name__,'n_layers':model.cfg.n_layers,'d_model':model.cfg.d_model,
           'parameter_dtype_counts':dict(Counter(str(v.dtype) for v in model.parameters())),'block12_parameter_sha256':fingerprints,
           'loader_processing_defaults':{name:signature.parameters[name].default for name in names}}
    return {'basis':basis,'cfg':{k:str(v) if isinstance(v,(torch.dtype,torch.device)) else v for k,v in model.cfg.to_dict().items()},
            'default_prepend_bos':model.cfg.default_prepend_bos,'tokenizer_prepends_bos':model.cfg.tokenizer_prepends_bos,
            'tokenizer_padding_side':model.tokenizer.padding_side,'bos_id':model.tokenizer.bos_token_id,'eos_id':model.tokenizer.eos_token_id,
            'generation_defaults':{'stop_at_eos':True,'use_past_kv_cache':True,'eos_token_id':None,'return_type':'input','prepend_bos':'source USE_DEFAULT_VALUE'},
            'rng_after_load':rng_record()}


def charge(s,out,kind,payload):
    used=sum(1 for file in s.run.glob('*/call-*-start.json') if read_json(file)['kind']==kind)
    if kind not in BUDGETS or used>=BUDGETS[kind]:raise ValueError('stage budget exhausted BEFORE marker')
    i=len(list(out.glob('call-*-start.json')));save(out/f'call-{i:04d}-start.json',{'manifest_sha256':stream_hash(s.run/'manifest.json'),'kind':kind,'input_sha256':p.canonical(payload)})


def stage(s,name):return fbase.stage(s,name)


def directions(s):
    import torch
    stage(s,'direction');value=torch.load(s.run/'direction/directions.pt',map_location='cpu',weights_only=True)
    if value['model']!=s.cfg['model_id'] or value['task']!='sarcasm' or value['seed']!=42 or value['instruction_mode']!='random' or value['n_train']!=500:raise ValueError('source direction metadata differs')
    unit=value['steering_directions'];target=value['target_projection']
    if unit.shape!=(32,4096) or target.shape!=(32,) or unit.dtype!=torch.float32 or target.dtype!=torch.float32 or not torch.isfinite(unit).all() or not torch.isfinite(target).all():raise ValueError('source finite FP32 direction/target bank required')
    metadata=read_json(s.run/'direction/tensors.json')
    if metadata!={'steering_directions':tensor_hash(unit),'target_projection':tensor_hash(target)}:raise ValueError('exact stored tensor identity differs')
    return value


def mechanics_lock(s):
    stage(s,'mechanics');directions(s)
    if not read_json(s.run/'mechanics/result.json')['all_pass']:raise ValueError('mechanics failure closes run')
    expected={'manifest_sha256':stream_hash(s.run/'manifest.json'),'direction_run_sha256':stream_hash(s.run/'direction/run.json'),
        'directions_sha256':stream_hash(s.run/'direction/directions.pt'),'direction_integrity_sha256':stream_hash(s.run/'direction/integrity.json'),
        'mechanics_run_sha256':stream_hash(s.run/'mechanics/run.json'),'mechanics_result_sha256':stream_hash(s.run/'mechanics/result.json'),
        'mechanics_integrity_sha256':stream_hash(s.run/'mechanics/integrity.json')}
    fbase.committed(s.cfg['mechanics_registration'])
    if read_json(s.cfg['mechanics_registration'])!=expected:raise ValueError('committed mechanics/direction bindings differ')


def capture_projection(s,out,author,model,prompt,kind,progress):
    charge(s,out,kind,{'prompt':prompt});value=author.capture_last_token_resid_post(model,prompt);progress[kind]+=1
    if value.shape!=(32,4096) or value.dtype.__str__()!='torch.float32' or not value.isfinite().all():raise ValueError('native last-token projection capture invalid')
    return value


def one_generation(s,out,author,model,prompt,hook,cap,progress):
    charge(s,out,'generations',{'prompt':prompt,'cap':cap});value=p.generate(model,prompt,author,hook,cap)
    progress['generations']+=1;progress['generation_wall_seconds']+=value['generation_wall_seconds'];return value


def mechanics(s,out,model,author,progress):
    import torch
    bundle=directions(s);u_cpu=bundle['steering_directions'][12];t_cpu=bundle['target_projection'][12];u=u_cpu.to('cuda:0');target=t_cpu.to('cuda:0')
    prompts=read_json(s.run/'inputs/prompts.json');rows=read_json(s.run/'inputs/mechanics.json');cases={};logits={}
    for arm in ('ordinary','saps_zero','fixed_zero','saps_nonzero','fixed_nonzero'):
        values=[]
        for row in rows:
            prompt=author.build_chat_prompt(model.tokenizer,row['sentence'],prompts['plain']);hook=None;c=None
            if arm.startswith('fixed'):
                h0=capture_projection(s,out,author,model,prompt,'mechanics_projection_forwards',progress)
                c=p.fixed_scalar(h0[12],u_cpu,t_cpu,0. if arm=='fixed_zero' else 1.);hook=author.make_fixed_projection_hook(12,u,c)
            elif arm.startswith('saps'):hook=author.make_saps_hook(12,u,target,0. if arm=='saps_zero' else 1.)
            value=one_generation(s,out,author,model,prompt,hook,16,progress);value.update(id=row['id'],fixed_adjustment=c);values.append(value)
        cases[arm]=values;save(out/(arm+'.json'),values)
    # Two bounded independent source-TL forward calls, baseline vs adaptive1.
    prompt=author.build_chat_prompt(model.tokenizer,rows[0]['sentence'],prompts['plain']);tokens=model.to_tokens(prompt)
    for alpha in (0.,1.):
        charge(s,out,'logit_forwards',{'prompt':prompt,'alpha':alpha})
        with model.hooks(fwd_hooks=[author.make_saps_hook(12,u,target,alpha)]),torch.no_grad():value=model(tokens,return_type='logits')[:,-1,:].float().cpu()
        logits[alpha]=value;progress['logit_forwards']+=1
    tok=lambda name:[x['full_token_ids'] for x in cases[name]]
    steered=[x for name,values in cases.items() if name!='ordinary' for x in values]
    gates={'saps0_hookfree_tokens':tok('ordinary')==tok('saps_zero'),'fixed0_hookfree_tokens':tok('ordinary')==tok('fixed_zero'),
        'finite_hook_samples':all(x['hook_audit']['finite'] for x in steered),
        'prefill_decode_coverage':all(x['hook_audit']['prefill_calls']>0 and x['hook_audit']['decode_calls']>0 and x['hooks_removed'] for x in steered),
        'nonzero_hidden':all(max(x['hook_audit'].get(phase,{}).get('max_abs_hidden_delta',0.) for x in cases[name] for phase in ('prefill','decode'))>0 for name in ('saps_nonzero','fixed_nonzero')),
        'nonzero_logits':(logits[1.]-logits[0.]).abs().max().item()>0}
    save(out/'result.json',{'gates':gates,'all_pass':all(gates.values()),'last_token_logit_max_delta':(logits[1.]-logits[0.]).abs().max().item(),
        'generation_budget':10,'separate_projection_budget':4,'source_hook_scope':'B1 last-token control broadcast to entire forwarded prefix; cached decode1position'})
    if not all(gates.values()):raise ValueError('mechanics failed; no rescue')


def run_arm(s,out,model,author,arm,progress):
    import torch
    bundle=directions(s);unit_cpu=bundle['steering_directions'];target_cpu=bundle['target_projection'];units=unit_cpu.to('cuda:0');targets=target_cpu.to('cuda:0')
    method,alpha=p.ARMS[arm];sentences=read_json(s.run/'inputs/data.json')['test'];plain=read_json(s.run/'inputs/prompts.json')['plain'];h0=[];traces=[]
    source_name=f'mistralai_Mistral-7B-Instruct-v0.1_sarcasm_{method}_alpha{str(alpha).replace(".","p")}_L12-12.jsonl'
    with (out/source_name).open('x',encoding='utf-8') as source_file,(out/'trace.jsonl').open('x',encoding='utf-8') as trace_file:
        for i,sentence in enumerate(sentences):
            prompt=author.build_chat_prompt(model.tokenizer,sentence,plain);c=None
            if method=='fixed_projection':
                captured=capture_projection(s,out,author,model,prompt,'test_projection_forwards',progress);h0.append(captured)
                c=p.fixed_scalar(captured[12],unit_cpu[12],target_cpu[12],alpha);hook=author.make_fixed_projection_hook(12,units[12],c)
            else:hook=author.make_saps_hook(12,units[12],targets[12],alpha)
            value=one_generation(s,out,author,model,prompt,hook,60,progress);value.update(sentence_id=i,sentence=sentence,method=method,alpha=alpha,layer=12,fixed_adjustment=c)
            traces.append(value);source_file.write(json.dumps({k:value[k] for k in ('sentence_id','sentence','method','alpha','layer','output')},ensure_ascii=False)+'\n')
            trace_file.write(json.dumps(value,ensure_ascii=False)+'\n');source_file.flush();trace_file.flush()
    if h0:
        with (out/'h0.pt').open('xb') as file:torch.save(torch.stack(h0),file)
    save(out/'source_invocation.json',{'source_script':'src/generate.py','model':s.cfg['model_id'],'method':method,'alpha':alpha,'task':'sarcasm','instruction_mode':'random',
        'dtype':'float16','layer_start':12,'layer_end':12,'max_new_tokens':60,'batch_size':1,'source_style_file':source_name,'source_return_type_default_preserved':True})


def preconditions(s,phase,arm=None,resume=False):
    required=[path(s.cfg['initial_registration'])]
    if phase in ('direction','mechanics','generate'):
        if phase=='generate' and arm not in p.ARMS:raise PreconditionWait('one fixed arm required')
        name=arm if phase=='generate' else phase
        if (s.run/name).exists():raise PreconditionWait('existing generation phase cannot be repeated')
        if phase=='mechanics':required += [s.run/'direction/integrity.json']
        if phase=='generate':
            if arm not in p.ARMS:raise PreconditionWait('one fixed arm required')
            required += [path(s.cfg['mechanics_registration']),s.run/'mechanics/integrity.json']
            previous=list(p.ARMS)[:list(p.ARMS).index(arm)]
            required += [s.run/name/'integrity.json' for name in previous]
    if phase=='judge':
        required += [s.run/name/'integrity.json' for name in p.ARMS]+[s.run/'packet.json']
        if resume and not (s.run/'grades').is_dir():raise PreconditionWait('no native stream to resume')
        if not resume and (s.run/'grades').exists():raise PreconditionWait('explicit native replay required')
    if any(not x.is_file() for x in required):raise PreconditionWait('stage/release not ready; no execution armed')


def gpu(s,phase,arm=None):
    role(s,'gpu');s.registered()
    if phase=='generate':mechanics_lock(s)
    if phase=='mechanics':directions(s)
    if (s.run/'stopped.json').exists():raise ValueError('permanent stop')
    available=fbase.resource(s.cfg);name=arm if phase=='generate' else phase;out=s.run/name;out.mkdir(mode=0o700,exist_ok=False)
    import torch
    progress={k:0 for k in BUDGETS};progress['generation_wall_seconds']=0.;began=time.perf_counter();complete=False;model=None;messages=[]
    class Capture(logging.Handler):
        def emit(self,record):messages.append(record.getMessage())
    handler=Capture();logger=logging.getLogger();logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');model=load_model(s)
            info=runtime(model);save(out/'runtime.json',info)
            if phase!='direction' and info['basis']!=read_json(s.run/'direction/runtime.json')['basis']:raise ValueError('fresh TL load processing/basis fingerprint differs')
            torch.cuda.reset_peak_memory_stats();author=p.author_module(path(s.cfg['author_root'])/'src/steering.py')
            if phase=='direction':
                data=read_json(s.run/'inputs/data.json');prompts=read_json(s.run/'inputs/prompts.json')
                def before(index,payload):charge(s,out,'train_prompt_forwards',payload)
                proxy=p.CaptureProxy(model,author,data['train'],prompts,before)
                try:units,targets=author.learn_style_direction(proxy,model.tokenizer,data['train'],prompts['plain'],prompts['style'],'random')
                finally:
                    progress['train_prompt_forwards']=len(proxy.ledger);save(out/'forward_ledger.json',proxy.ledger)
                if len(proxy.ledger)!=1500:raise ValueError('actual author extraction forward count differs')
                bundle={'steering_directions':units,'target_projection':targets,'model':s.cfg['model_id'],'task':'sarcasm','instruction_mode':'random','seed':42,'n_train':500,'n_layers':32,'d_model':4096}
                with (out/'directions.pt').open('xb') as file:torch.save(bundle,file)
                reloaded=torch.load(out/'directions.pt',map_location='cpu',weights_only=True)
                if not torch.equal(units,reloaded['steering_directions']) or not torch.equal(targets,reloaded['target_projection']):raise ValueError('source tensor roundtrip differs')
                save(out/'tensors.json',{'steering_directions':tensor_hash(units),'target_projection':tensor_hash(targets)})
                save(out/'rng_after_learning.json',rng_record())
            elif phase=='mechanics':mechanics(s,out,model,author,progress)
            else:run_arm(s,out,model,author,arm,progress)
            save(out/'warnings.json',{'python_warnings':[str(x.message) for x in seen],'logging_messages':messages})
        torch.cuda.synchronize();complete=True
    finally:
        logger.removeHandler(handler)
        save(out/'run.json',{'completed':complete,**available,**progress,'elapsed_wall_seconds':time.perf_counter()-began,'seed_before_each_fresh_load':42,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if model is not None else None,'peak_reserved_bytes':torch.cuda.max_memory_reserved() if model is not None else None,
            'memory_scope':'PyTorch allocator only','cost_scope':'Saved/completed work and irreversible start markers retained separately; interrupted work may add cost',
            'torch_cuda':torch.version.cuda,'packages':fbase.packages(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'dtype':str(model.cfg.dtype) if model is not None else None,
            'backend':'native TransformerLens2.17, not HF SDPA port','files_sha256':{str(x.relative_to(out)):stream_hash(x) for x in out.rglob('*') if x.is_file()}})


def predictions(s):
    values={};data=read_json(s.run/'inputs/data.json')['test'];rendered=read_json(s.run/'inputs/rendered.json')['test'];author=p.author_module(path(s.cfg['author_root'])/'src/steering.py')
    for arm in p.ARMS:
        stage(s,arm)
        with (s.run/arm/'trace.jsonl').open() as file:rows=[json.loads(line) for line in file if line.strip()]
        if len(rows)!=450 or [x['sentence_id'] for x in rows]!=list(range(450)):raise ValueError('complete source test order required')
        source=read_json(s.run/arm/'source_invocation.json')['source_style_file']
        with (s.run/arm/source).open() as file:source_rows=[json.loads(line) for line in file if line.strip()]
        for i,(row,original) in enumerate(zip(rows,source_rows)):
            if row['sentence']!=data[i] or row['rendered_prompt']!=rendered[i] or (row['method'],row['alpha'])!=p.ARMS[arm] or row['layer']!=12:raise ValueError('fixed source input/setting lineage differs')
            if original!={k:row[k] for k in ('sentence_id','sentence','method','alpha','layer','output')} or row['output']!=author.strip_chat_template(row['source_full_decoded_output']):raise ValueError('unchanged author stripping/source record mismatch')
            if row['full_token_ids']!=row['input_ids']+row['generated_ids'] or len(row['generated_ids'])>60 or not row['hooks_removed']:raise ValueError('actual full token trace differs')
            if row['source_generate_kwargs']!={'max_new_tokens':60,'do_sample':False,'verbose':False}:raise ValueError('source generation arguments changed')
        values[arm]=rows
    return values


def bodies(s):
    values=predictions(s);rubric=read_json(s.run/'inputs/rubric.json')
    return [{'rubric':rubric,'expected_count':20,'expected_ids':[r['id'] for r in batch],
        'return_schema':{'ratings':'Exact id plus integer0..5 sarcasm, entailment, grammaticality, clarity. No reasons/extra fields.'},
        'items':[{'id':r['id'],'original_complaint':values[r['arm']][r['sentence_id']]['sentence'],
                  'generated_sentence':values[r['arm']][r['sentence_id']]['output']} for r in batch]} for batch in read_json(s.run/'inputs/plans.json')]


def packet(s):save(s.run/'packet.json',{'manifest_sha256':stream_hash(s.run/'manifest.json'),'source_runs_sha256':{arm:stream_hash(s.run/arm/'run.json') for arm in p.ARMS},'batches':bodies(s)})


def bound_bodies(s):
    saved=read_json(s.run/'packet.json');expected={'manifest_sha256':stream_hash(s.run/'manifest.json'),'source_runs_sha256':{arm:stream_hash(s.run/arm/'run.json') for arm in p.ARMS},'batches':bodies(s)}
    if saved!=expected:raise ValueError('exact blind packet reconstruction differs')
    return saved['batches']


def binding(s):return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'packet_sha256':stream_hash(s.run/'packet.json'),
                       'binary_sha256':fbase.BINARY_SHA,'cli_version':s.m['cli_version'],'argv':f.argv('claude'),'maximum_starts':180}


def parse(attempt,payload):
    try:outer,models=f.outer_result(attempt)
    except f.InvalidSubmission as exc:raise StructuralError('malformed native envelope') from exc
    try:return p.scores(f.result_json(outer.get('result')),payload['expected_ids']),models
    except (ValueError,TypeError,KeyError) as exc:raise StructuralError('structural malformed/incomplete schema only') from exc


def read_grades(s,partial=False):
    directory=s.run/'grades';payloads=bound_bodies(s);bound=stream_hash(directory/'run_binding.json')
    if read_json(directory/'run_binding.json')!=binding(s):raise ValueError('grader authority differs')
    if len(list(directory.glob('*-start.json')))>180:raise ValueError('180 native budget exceeded')
    for file in directory.glob('*attempt*'):
        m=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
        if not m or int(m[1])>=90:raise ValueError('stray attempt artifact')
    found={};provenance=[]
    for bi,payload in enumerate(payloads):
        valid=False
        for ai in range(2):
            stem=f'batch-{bi:03d}-attempt-{ai}';file=directory/(stem+'.json')
            if not file.exists():
                if list(directory.glob(stem+'*')):raise f.UncertainAttempt('uncertain call stops permanently')
                break
            attempt=fbase.checked_attempt(directory,bi,ai,payload,bound,allow_pending_receipt=partial)
            try:scores,models=parse(attempt,payload)
            except StructuralError:
                if ai==1:raise ValueError('structural attempts exhausted')
                continue
            if ai==0 and list(directory.glob(f'batch-{bi:03d}-attempt-1*')):raise ValueError('attempt after first valid batch')
            found.update(scores);provenance.append({'batch':bi,'first_valid_attempt':ai,'actual_model_usage_ids':models,'payload_sha256':f.canonical_hash(payload),'raw_sha256':stream_hash(file)});valid=True;break
        if not valid and not partial:raise ValueError('incomplete grading')
    return {'ratings':found,'batches':provenance,'binding_sha256':bound,'first_valid_only':True}


def judge(s,resume=False):
    role(s,'judge');s.registered();mechanics_lock(s)
    if (s.run/'stopped.json').exists():raise ValueError('permanent stop')
    binary=shutil.which('claude')
    if not binary or stream_hash(Path(binary))!=fbase.BINARY_SHA or f.version('claude')!=s.m['cli_version']:raise ValueError('pinned native identity required')
    f.validate_audit(read_json(s.cfg['judge_runtime_config'])['judge_audit']);payloads=bound_bodies(s);directory=s.run/'grades';directory.mkdir(mode=0o700,exist_ok=True)
    with f.exclusive_run(directory):
        f.save_or_match(directory/'run_binding.json',binding(s));read_grades(s,True)
        for bi,payload in enumerate(payloads):
            for ai in range(2):
                file=directory/f'batch-{bi:03d}-attempt-{ai}.json'
                native_budget(directory,file)
                attempt,_=f.replay_or_call(directory,bi,ai,payload,stream_hash(directory/'run_binding.json'),'claude')
                try:parse(attempt,payload);break
                except StructuralError:
                    if ai==1:raise
        return read_grades(s)


def native_budget(directory,completed_attempt):
    if not completed_attempt.exists() and len(list(directory.glob('*-start.json')))>=180:raise ValueError('native budget BEFORE marker')


def report(s):
    costs={file.parent.name:{k:v for k,v in read_json(file).items() if k!='files_sha256'} for file in s.run.glob('*/run.json')}
    stop=read_json(s.run/'stopped.json') if (s.run/'stopped.json').exists() else None
    result={'status':'incomplete','scope':'Original-Mistral released SAPS fixed-layer method, not exact numerical reproduction',
        'manifest_sha256':stream_hash(s.run/'manifest.json'),'source_sha256':s.m['sources'],'model':{'id':s.cfg['model_id'],'revision':s.cfg['revision']},
        'costs':costs,'totals':{kind:sum(c.get(kind,0) for c in costs.values()) for kind in BUDGETS},'native_starts':len(list((s.run/'grades').glob('*-start.json'))),'stop':stop,
        'limits':['Source actual TransformerLens2.17 loader processing defaults, FP16 compatibility path, BOS quirks and cached B1 all-prefix broadcast are retained; no HF-coordinate transfer.',
                  'Fixed block12 is prospective, not a per-test-sentence oracle or validation sweep. Released log sentences/prompt metadata mismatch released data, so public artifact arithmetic is not a matched input/runtime comparator.',
                  'One pinned Claude co-rates20 items with four original definitions instead of a three-judge average; server snapshot unknown and batch contrast effects limit ordinal comparisons.',
                  'Original complaint provides entailment reference only; sarcasm judged from generated sentence alone. Level3 permits partial irony; self-contained clear sarcasm starts4.',
                  'Caps/empty/degeneration and truncated-high-score counts are reported alongside style; no unregistered relabeling, post-hoc selection, random arm, capability or joint trait release.']}
    if (s.run/'grades/accepted.json').exists():
        accepted=read_grades(s)
        if accepted!=read_json(s.run/'grades/accepted.json'):raise ValueError('frozen accepted grades differ')
        rows=predictions(s);ratings={(x['arm'],x['sentence_id']):accepted['ratings'][x['id']] for b in read_json(s.run/'inputs/plans.json') for x in b}
        result['metrics']=p.aggregate(rows,ratings);result['firstvalid_batches']=len(accepted['batches'])
        result['actual_model_usage_ids']=sorted({name for b in accepted['batches'] for name in b['actual_model_usage_ids']})
        result['accepted_sha256']=stream_hash(s.run/'grades/accepted.json')
        if stop is None:result['status']='complete'
    from scripts.author_gcad_impolite_common import public_safe
    public_safe(result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','verify','direction','mechanics','generate','packet','judge','report']);parser.add_argument('--arm',choices=list(p.ARMS));parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--execute-judgments',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare();print('CPU snapshot prepared');return
    if args.stage in ('direction','mechanics','generate') and not args.execute_gpu:print('No execution: --execute-gpu required');return
    if args.stage=='judge' and not args.execute_judgments:print('No execution: --execute-judgments required');return
    live=args.stage in ('direction','mechanics','generate','judge');s=None;armed=False;accepted=None;result=None
    try:
        s=Context()
        if live:preconditions(s,args.stage,args.arm,args.resume);armed=True
        s.verify(models=live)
        try:
            if (s.run/'stopped.json').exists() and args.stage not in ('verify','report'):raise ValueError('closed study')
            if args.stage in ('direction','mechanics','generate'):gpu(s,args.stage,args.arm)
            elif args.stage=='packet':packet(s)
            elif args.stage=='judge':accepted=judge(s,args.resume)
            elif args.stage=='report':result=report(s)
        finally:s.verify(models=live)
        if args.stage in ('direction','mechanics','generate'):
            out=s.run/(args.arm if args.stage=='generate' else args.stage);save(out/'integrity.json',{'manifest_sha256':stream_hash(s.run/'manifest.json'),'run_sha256':stream_hash(out/'run.json'),'entry_exit_integrity_passed':True})
        if accepted is not None:f.save_or_match(s.run/'grades/accepted.json',accepted)
        if result is not None:save(path(s.cfg['result']),result)
    except (PreconditionWait,fbase.ResourceWait):print('Precondition/resource wait; no new execution');return
    except BaseException as exc:
        if live and armed and path(RUN).is_dir():fbase.stop(s,'fatal_execution_or_integrity',exc)
        raise
    print('SAPS stage verified: '+args.stage)


if __name__=='__main__':main()

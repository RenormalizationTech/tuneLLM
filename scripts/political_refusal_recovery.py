"""One registered network recovery; frozen parent science is imported unchanged.

The complete scientific data, settings and all future branch plans are frozen
upfront. This CLI deliberately exposes no fit/intervention/model-tuning stage.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

from scripts import political_refusal as parent
from scripts.political_refusal import PRIMARY_RUBRIC, all_plans, render, compile_fixture
from scripts import political_refusal_common as c
from scripts import author_gcad_impolite as infra
from scripts import author_gcad_impolite_common as native
from scripts import sarcasm_method_transfer_common as shared
from scripts import tone_judge as f
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash

RUN = "outputs/political-refusal-recovery-01"
CONFIG = "configs/political_refusal_recovery.json"
save = f.save


Wait = parent.Wait
stage = parent.stage
predictions = parent.predictions
bodies = parent.bodies
parse = parent.parse
generate_headroom = parent.generate_headroom
stop = parent.stop


def guard(run):
    if run != path(RUN) or run.is_symlink() or run.resolve() != path(RUN):
        raise ValueError("only the new fixed calibration/headroom root is writable")


OPERATIONAL_FIELDS = {'run','result','initial_registration','calibration_registration','headroom_registration'}
PARENT = 'outputs/political-refusal-01'
PARENT_PROVENANCE = 'results/protocols/political-refusal-parent-closure-01.json'
RECOVERY_REVIEW = 'results/reviews/political_refusal_network_recovery_review.md'
SCIENCE_FILES = ('config.json','inputs/data.json','inputs/plans.json','inputs/rendered.json',
                 'inputs/primary_rubric.json','inputs/coherence_rubric.json','inputs/model_defaults.json')


def science_equal(cfg, original):
    """Only explicit execution locations differ; all other config fields exact."""
    return ({k:v for k,v in cfg.items() if k not in OPERATIONAL_FIELDS|{'recovery'}} ==
            {k:v for k,v in original.items() if k not in OPERATIONAL_FIELDS})


def validate_config(cfg):
    original=read_json(parent.CONFIG);parent.validate_config(original)
    if cfg!=read_json(CONFIG) or not science_equal(cfg,original):raise ValueError('recovery scientific config differs')
    expected={'run':RUN,'result':'results/political_refusal_recovery_headroom_01.json',
        'initial_registration':'results/protocols/political-refusal-recovery-01.json',
        'calibration_registration':'results/protocols/political-refusal-recovery-calibration-01.json',
        'headroom_registration':'results/protocols/political-refusal-recovery-headroom-01.json'}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('fixed recovery paths differ')
    if cfg.get('recovery')!={'parent_run':PARENT,'parent_config':parent.CONFIG,'parent_provenance':PARENT_PROVENANCE,
            'review':RECOVERY_REVIEW,'parent_charged_starts':1,'network_execution_policy':'require_escalated','maximum_recoveries':1}:
        raise ValueError('one-recovery provenance/policy differs')


def file_inventory(root):
    files={}
    for file in sorted(root.rglob('*')):
        if file.is_symlink():raise ValueError('parent symlinks forbidden')
        if file.is_file():files[str(file.relative_to(root))]=stream_hash(file)
    return files


def content_free_failure(attempt):
    """Recognize only this recorded DNS failure, never a grading submission."""
    outer=c.strict_json(attempt['stdout'])
    if (attempt.get('returncode')!=1 or outer.get('is_error') is not True or outer.get('modelUsage')!={}
            or outer.get('result')!="API Error: Can't reach the API server — check your internet or DNS (EAI_AGAIN)"):
        raise ValueError('parent is not the reviewed content-free DNS failure')
    if any(k in outer for k in ('ratings','grade','grades','answer','completion')):
        raise ValueError('parent has possible grading content')
    if 'ratings' in outer['result'].lower():raise ValueError('parent contains ratings')
    return outer


def parent_facts(root):
    """Read-only checks; no exclusive_run, resume, receipt repair or old writes."""
    if root.is_symlink():raise ValueError('parent symlink forbidden')
    starts=list((root/'native').rglob('batch-*-attempt-*-start.json'))
    wanted=root/'native/calibration/primary/batch-000-attempt-0-start.json'
    if starts!=[wanted] or list(root.rglob('accepted.json')) or (root/'calibration/check.json').exists() or (root/'headroom').exists():
        raise ValueError('parent must have exactly one start, no labels and no Qwen outputs')
    directory=wanted.parent
    if {p.name for p in directory.glob('*attempt*')}!={'batch-000-attempt-0-start.json','batch-000-attempt-0.json','batch-000-attempt-0-receipt.json'}:
        raise ValueError('unexpected parent attempt artifacts')
    if read_json(root/'stopped.json')!={'stop_cause':'fatal_execution_or_integrity','stage':'calibration','error_type':'RuntimeError'}:
        raise ValueError('parent fatal marker differs')
    ps=type('ParentReadOnly',(),{'run':root,'cfg':read_json(root/'config.json')})()
    payloads=parent.bodies(ps,'calibration','primary');bound=read_json(directory/'run_binding.json')
    expected={'manifest_sha256':stream_hash(root/'manifest.json'),'phase':'calibration','metric':'primary',
              'payloads_sha256':f.canonical_hash(payloads),'initial_maximum_starts':32,'combined_maximum_starts':424}
    if bound!=expected:raise ValueError('parent payload/manifest binding differs')
    attempt=infra.checked_attempt(directory,0,0,payloads[0],stream_hash(directory/'run_binding.json'))
    content_free_failure(attempt)
    return {'native_starts':1,'first_valid_batches':0,'qwen_generations':0,
            'calibration_primary_payloads_sha256':bound['payloads_sha256']}


def verify_parent(cfg, record=None):
    root=path(cfg['recovery']['parent_run'])
    record=read_json(cfg['recovery']['parent_provenance']) if record is None else record
    if record.get('status')!='incomplete_infrastructure_no_labels' or record.get('parent_run')!=PARENT:
        raise ValueError('parent provenance status/root differs')
    if file_inventory(root)!=record['files_sha256']:raise ValueError('parent file bytes/inventory changed')
    infra.check_hashes([record['registrations_sha256']])
    if parent_facts(root)!=record['verified_counts_and_payload']:
        raise ValueError('parent closure evidence differs')
    return record


def verify_scientific_snapshots(s):
    root=path(PARENT)
    for name in SCIENCE_FILES:
        if stream_hash(s.run/name)!=stream_hash(root/name):raise ValueError('parent scientific snapshot differs: '+name)
    if stream_hash(s.run/'inputs/parent_config.json')!=stream_hash(root/'config.json'):
        raise ValueError('exact parent config reference differs')
    if not science_equal(s.cfg,read_json(s.run/'config.json')):raise ValueError('execution/science config differs')
    actual=f.canonical_hash(bodies(s,'calibration','primary'))
    record=read_json(s.cfg['recovery']['parent_provenance'])
    if actual!=record['verified_counts_and_payload']['calibration_primary_payloads_sha256']:
        raise ValueError('recovery payload differs from failed parent')


def verify_recompiled_inputs(cfg, data, rendered, tok):
    """CPU parity check only; copied scientific bytes remain authoritative."""
    compiled=compile_fixture(cfg)
    for row in compiled['calibration']:
        row['authored_token_ids']=tok(row['answer'],add_special_tokens=False)['input_ids']
    if compiled!=data:raise ValueError('recompiled source/calibration data differs')
    expected={}
    for phase in ('fit','development','heldout'):
        for row in data[phase]:
            conditions=[(k,cfg['systems'][k]) for k in ('positive','negative')] if phase=='fit' else [('baseline',None),('prompted',cfg['systems']['positive'])]
            for condition,system in conditions:
                prompt=render(tok,row,system)
                expected[row['id']+'/'+condition]={'prompt':prompt,'input_ids':tok(prompt,add_special_tokens=False)['input_ids']}
    if expected!=rendered:raise ValueError('re-rendered prompt/token IDs differ')


def prepare():
    run=path(RUN);guard(run)
    if run.exists():raise FileExistsError('fresh recovery root only')
    cfg=read_json(CONFIG);validate_config(cfg);record=verify_parent(cfg)
    ps=parent.Context();ps.verify(models=True)
    if read_json(ps.run/'config.json')!=read_json(parent.CONFIG):raise ValueError('parent configuration changed')
    infra.verify_role(type('Runtime',(),{'cfg':cfg,'m':{'gpu_packages':ps.m['gpu_packages']}})(),'gpu')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    verify_recompiled_inputs(cfg,read_json(ps.run/'inputs/data.json'),read_json(ps.run/'inputs/rendered.json'),tok)
    # Copy scientific artifacts rather than re-tokenizing or regenerating plans.
    sources=infra.closure(['scripts/political_refusal_recovery.py','scripts/political_refusal_recovery_runbook.md',CONFIG])
    hashes={n:stream_hash(path(n)) for n in sources}
    external={**ps.m['sources'],**ps.m['external_sha256']}
    external.update({n:stream_hash(path(n)) for n in (PARENT_PROVENANCE,RECOVERY_REVIEW)})
    infra.check_hashes([hashes,external,ps.m['model_sha256']])
    run.mkdir(mode=0o700);save(run/'execution_config.json',cfg)
    for name in SCIENCE_FILES:
        dest=run/name;dest.parent.mkdir(parents=True,mode=0o700,exist_ok=True);shutil.copyfile(ps.run/name,dest);dest.chmod(0o600)
    shutil.copyfile(ps.run/'config.json',run/'inputs/parent_config.json');(run/'inputs/parent_config.json').chmod(0o600)
    for name in sources:
        dest=run/'source'/name;dest.parent.mkdir(parents=True,mode=0o700,exist_ok=True);shutil.copyfile(path(name),dest);dest.chmod(0o600)
    save(run/'manifest.json',{'sources':hashes,'external_sha256':external,'model_sha256':ps.m['model_sha256'],
        **{k:ps.m[k] for k in ('gpu_packages','cli_version','binary','binary_sha256','combined_budget','initial_budget')},
        'files':file_inventory(run),'parent_charged_starts':1,'parent_provenance_sha256':stream_hash(path(PARENT_PROVENANCE)),
        'network_execution_policy':'require_escalated','no_model_or_judge_calls':True})
    s=Context();s.verify(models=True)


class Context:
    def __init__(self):
        self.run=path(RUN);guard(self.run);self.cfg=read_json(self.run/'execution_config.json');validate_config(self.cfg)
        self.m=read_json(self.run/'manifest.json')
    def verify(self,models=False):
        maps=[self.m['sources'],self.m['external_sha256'],{str(self.run/n):h for n,h in self.m['files'].items()}]
        if models:maps.append(self.m['model_sha256'])
        infra.check_hashes(maps)
        if self.m['parent_charged_starts']!=1 or self.m['parent_provenance_sha256']!=stream_hash(path(PARENT_PROVENANCE)):
            raise ValueError('parent charge/provenance differs')
        verify_parent(self.cfg);verify_scientific_snapshots(self)
        if read_json(self.run/'inputs/plans.json')!=all_plans(read_json(self.run/'inputs/data.json'),self.cfg):raise ValueError('frozen plans changed')
    def registered(self):
        for name in (*self.m['sources'],self.cfg['protocol'],self.cfg['design_review'],self.cfg['data_acceptance'],
                     PARENT_PROVENANCE,RECOVERY_REVIEW,self.cfg['initial_registration']):infra.committed(name)
        if read_json(self.cfg['initial_registration'])!=registration_expected(self):raise ValueError('initial recovery registration differs')


def registration_expected(s):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),
            'parent_provenance_sha256':s.m['parent_provenance_sha256'],'parent_charged_starts':1,
            'initial_cumulative_maximum_starts':32,'global_cumulative_maximum_starts':424,
            'recovery_batch_maximum_structural_attempts':2,'network_execution_policy':'require_escalated'}


def recovery_disclosure(s):
    return {'parent_status':'incomplete_infrastructure_no_labels','parent_charged_starts':1,
            'recovery_starts':native_count(s)-1,'parent_provenance_sha256':s.m['parent_provenance_sha256'],
            'registration_sha256':stream_hash(path(s.cfg['initial_registration'])),
            'science_and_payloads_unchanged':True,'no_parent_labels_or_qwen_outputs':True,
            'network_permission_change':{'parent':'default sandbox DNS unavailable','recovery_required':'require_escalated network-enabled host execution',
                'outer_os_sandbox_relaxed':True,'native_argv_unchanged':True,'native_environment_unchanged':True,
                'llm_tool_permissions_unchanged':True,'same_user_auth_and_runtime':True,'new_grader_tool_grants':False,
                'verification_scope':'Operator launch policy; the process does not pretend to detect sandbox permission mode.'}}


def binding(s,phase,metric):
    return {"manifest_sha256":stream_hash(s.run/"manifest.json"),"phase":phase,"metric":metric,
        "payloads_sha256":f.canonical_hash(bodies(s,phase,metric)),"initial_maximum_starts":32,"combined_maximum_starts":424,
        **({"headroom_run_sha256":stream_hash(s.run/"headroom/run.json"),"headroom_integrity_sha256":stream_hash(s.run/"headroom/integrity.json")} if phase=="headroom" else {})}


def native_count(s):
    return s.m["parent_charged_starts"] + len(list((s.run/"native").glob("**/batch-*-attempt-*-start.json")))


def read_grades(s,phase,metric,partial=False):
    """Read-only receipt replay; unlike historical helpers, local ceiling is32."""
    out=s.run/"native"/phase/metric;payloads=bodies(s,phase,metric);wanted=binding(s,phase,metric)
    if read_json(out/"run_binding.json")!=wanted or native_count(s)>32:raise ValueError("initial grading binding/budget differs")
    for file in out.glob('*attempt*'):
        match=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
        if not match or int(match[1])>=len(payloads):raise ValueError("stray/out-of-budget attempt artifact")
    values={};receipts=[];bound=stream_hash(out/"run_binding.json")
    for bi,body in enumerate(payloads):
        valid=False
        for ai in range(2):
            stem=f'batch-{bi:03d}-attempt-{ai}';file=out/(stem+'.json')
            if not file.exists():
                if list(out.glob(stem+'*')):raise f.UncertainAttempt("uncertain native start; no new call")
                break
            attempt=infra.checked_attempt(out,bi,ai,body,bound,allow_pending_receipt=partial)
            try:parsed,models=parse(attempt,body,metric,s.cfg)
            except f.InvalidSubmission:
                if ai==1:raise
                continue
            if ai==0 and list(out.glob(f'batch-{bi:03d}-attempt-1*')):raise ValueError("attempt after first valid")
            values[str(bi)]=parsed;receipts.append({"batch":bi,"first_valid_attempt":ai,"models":models,"raw_sha256":stream_hash(file)});valid=True;break
        if not valid and not partial:raise ValueError("incomplete stream; no partial scoring")
    return {"values":values,"batches":receipts,"binding_sha256":bound}


def judge_stream(s,phase,metric,resume):
    infra.verify_role(s,"judge")
    binary=shutil.which('claude')
    if not binary or stream_hash(Path(binary))!=infra.BINARY_SHA or f.version(binary)!=s.m["cli_version"]:raise ValueError("pinned native executable/version differs")
    f.validate_audit(read_json(s.cfg["judge_runtime_config"])["judge_audit"])
    return run_journal(s,phase,metric,binary,resume)


def run_journal(s,phase,metric,binary,resume):
    """Small budget-aware disk loop; invocation/receipt/isolation source unchanged."""
    out=s.run/"native"/phase/metric
    if resume and not out.is_dir():raise Wait("no stream to resume")
    if out.exists() and not resume:raise Wait("existing stream requires explicit first-valid replay")
    out.mkdir(parents=True,mode=0o700,exist_ok=True)
    with f.exclusive_run(out):
        f.save_or_match(out/"run_binding.json",binding(s,phase,metric));read_grades(s,phase,metric,True)
        for bi,body in enumerate(bodies(s,phase,metric)):
            for ai in range(2):
                file=out/f'batch-{bi:03d}-attempt-{ai}.json'
                if not file.exists() and (native_count(s)>=32 or native_count(s)>=424):raise ValueError("initial32-start ceiling BEFORE new invocation")
                attempt,_=f.replay_or_call(out,bi,ai,body,stream_hash(out/"run_binding.json"),binary)
                try:parse(attempt,body,metric,s.cfg);break
                except f.InvalidSubmission:
                    if ai==1:raise
        accepted=read_grades(s,phase,metric);f.save_or_match(out/"accepted.json",accepted);return accepted


def calibration_result(s):
    primary=read_grades(s,"calibration","primary");data=read_json(s.run/"inputs/data.json");plan=read_json(s.run/"inputs/plans.json")["calibration"]["primary"]
    if primary!=read_json(s.run/'native/calibration/primary/accepted.json'):raise ValueError('accepted calibration changed')
    items=[];ratings=[];expected={};byid={r["id"]:r for r in data["calibration"]}
    for bi,batch in enumerate(plan):
        for item in batch:
            row=byid[item["physical_id"]];alias=item["id"]
            items.append({"id":alias,"fact_ids":row["fact_ids"]});actual=primary["values"][str(bi)][alias]
            ratings.append({k:v for k,v in actual.items() if k!="descriptive_framing"});expected[alias]=data["calibration_expected"][row["id"]]
    result=c.calibration_check(items,{"ratings":ratings},expected,framing_labels=s.cfg["framing_labels"])
    result["primary_accepted_sha256"]=stream_hash(s.run/"native/calibration/primary/accepted.json")
    result['mechanical_invalid_authored_cases']=sum(c.mechanical_validity(r['answer'],r['authored_token_ids'],eos_token_ids=[])['invalid_mechanical'] for r in data['calibration'])
    return result


def lock(s,kind):
    if kind=="calibration":
        result=read_json(s.run/"calibration/check.json")
        if result!=calibration_result(s) or not result["all_pass"]:raise ValueError("exact passing calibration required")
        if read_grades(s,"calibration","coherence")!=read_json(s.run/'native/calibration/coherence/accepted.json'):raise ValueError('accepted coherence changed')
        files={"calibration_check_sha256":"calibration/check.json","primary_accepted_sha256":"native/calibration/primary/accepted.json","coherence_accepted_sha256":"native/calibration/coherence/accepted.json"}
    else:
        result=read_json(s.run/"headroom/choice.json")
        if result!=headroom_result(s) or result["mode"]=="close":raise ValueError("exact non-null frozen headroom required")
        files={"headroom_choice_sha256":"headroom/choice.json","headroom_run_sha256":"headroom/run.json","headroom_integrity_sha256":"headroom/integrity.json","primary_accepted_sha256":"native/headroom/primary/accepted.json"}
    wanted={"manifest_sha256":stream_hash(s.run/"manifest.json"),**{k:stream_hash(s.run/v) for k,v in files.items()}}
    infra.committed(s.cfg[kind+"_registration"])
    if read_json(s.cfg[kind+"_registration"])!=wanted:raise ValueError("exact committed stage registration differs")


def headroom_scores(s):
    values=predictions(s);data=read_json(s.run/"inputs/data.json");questions={(r["canonical_id"],r["language"]):r for r in data["development"]};plan=read_json(s.run/"inputs/plans.json")["headroom"]["primary"]
    grades=read_grades(s,"headroom","primary");decoded={}
    if grades!=read_json(s.run/"native/headroom/primary/accepted.json"):raise ValueError("accepted grading changed")
    for bi,batch in enumerate(plan):
        for entry in batch:
            q=questions[entry["canonical_id"],entry["language"]];decoded[q["id"],entry["condition"]]=grades["values"][str(bi)][entry["id"]]
    eos=read_json(s.run/"inputs/model_defaults.json")["eos_token_id"];eos=[eos] if type(eos) is int else eos
    byid={r["id"]:r for r in data["development"]}
    return {arm:[c.score_response(byid[r["id"]],r,decoded[r["id"],arm],eos_token_ids=eos) for r in rows] for arm,rows in values.items()}


def headroom_result(s):
    rows=headroom_scores(s)
    return c.headroom_choice(rows["baseline"],rows["prompted"],thresholds=s.cfg["thresholds"])


def report(s):
    result={"status":"incomplete","scope":"Initial calibration/headroom only; no steering or fit/intervention execution exists in this runner.",
        "manifest_sha256":stream_hash(s.run/"manifest.json"),"source_sha256":s.m["sources"],"native_starts":native_count(s),"first_valid_batches":0,"fresh_generations":0,
        "recovery":recovery_disclosure(s),
        "combined_scientific_ceiling":s.m["combined_budget"],"limits":["Reference-consistency on this source pool, not certified truth or ideological agreement.","Chinese evaluation questions are project translations; paired languages and related events are not independent.","Headroom is not a steering effect; a positive mode only supports a later separately registered implementation with unchanged scientific rules."]}
    result['model']={'id':s.cfg['model_id'],'revision':s.cfg['revision'],'dtype':'bfloat16','attention':'sdpa'}
    result['generation_settings']={'seed':42,'batch_size':1,'do_sample':False,'max_new_tokens':512}
    result['generation_start_records']=len(list((s.run/'headroom').glob('call-*-start.json')))
    for phase,metrics in [("calibration",("primary","coherence")),("headroom",("primary",))]:
        for metric in metrics:
            file=s.run/"native"/phase/metric/"accepted.json"
            if file.exists():
                accepted=read_grades(s,phase,metric)
                if accepted!=read_json(file):raise ValueError("accepted stream changed")
                result["first_valid_batches"]+=len(accepted["batches"])
    if (s.run/"calibration/check.json").exists():result["calibration"]=calibration_result(s)
    if (s.run/"headroom/integrity.json").exists():
        run=stage(s);result["fresh_generations"]=run["generations"];result["cost"]={k:v for k,v in run.items() if k not in ("files_sha256","generation_config","packages")}
    if (s.run/"headroom/choice.json").exists():
        choice=headroom_result(s)
        if choice!=read_json(s.run/"headroom/choice.json"):raise ValueError("early choice drift")
        result["headroom"]=choice;result["status"]="headroom_null" if choice["mode"]=="close" else "headroom_available_requires_separate_registered_extension"
        pred=predictions(s);questions={r["id"]:r for r in read_json(s.run/"inputs/data.json")["development"]}
        result["mechanical_by_language"]={arm:{lang:{"n":len(rows),"caps":sum(r["hit_token_limit"] for r in rows),"eos":sum(r["ended_eos"] for r in rows),"mean_tokens":sum(len(r["generated_ids"]) for r in rows)/len(rows)} for lang in ("EN","ZH") if (rows:=[r for r in records if questions[r["id"]]["language"]==lang])} for arm,records in pred.items()}
    if (s.run/"stopped.json").exists():
        result["stop"]=read_json(s.run/"stopped.json");cause=result["stop"]["stop_cause"]
        result["status"]="calibration_null" if cause=="calibration_failed" else "headroom_null" if cause=="no_headroom" else "incomplete"
    native.public_safe(result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','verify','calibration','headroom','judge','decide','report'])
    parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--execute-native',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage in ('calibration','judge') and not args.execute_native or args.stage=='headroom' and not args.execute_gpu:
        print('No execution flag; no calls or writes.');return
    s=None
    try:
        s=Context();s.verify(models=args.stage=='headroom')
        if args.stage=='verify':print('source/input/model-metadata bindings verified; no calls');return
        s.registered()
        if args.stage in ('calibration','headroom','judge') and (s.run/'stopped.json').exists():raise Wait('terminal stop; no new calls')
        if args.stage=='calibration':
            if (s.run/'calibration/check.json').exists():raise Wait('calibration already closed; no rerating')
            if args.resume and not (s.run/'native/calibration').is_dir():raise Wait('no calibration phase to resume')
            # Replay a completed first stream when resuming the two-stream phase.
            for metric in ('primary','coherence'):
                directory=s.run/'native/calibration'/metric
                judge_stream(s,'calibration',metric,args.resume and directory.is_dir())
                if metric=='primary' and not calibration_result(s)['all_pass']:break
            s.verify();result=calibration_result(s);save(s.run/'calibration/check.json',result)
            if not result['all_pass']:stop(s,'calibration_failed')
            print(json.dumps({'calibration_pass':result['all_pass'],'native_starts':native_count(s)}));return
        if args.stage=='headroom':
            lock(s,'calibration')
            if (s.run/'headroom').exists():raise Wait('headroom generation already started; no repeat')
            resource=infra.resource(s.cfg)  # Busy before creation: no start/no fatal marker.
            generate_headroom(s,resource);s.verify(models=True)
            save(s.run/'headroom/integrity.json',{'manifest_sha256':stream_hash(s.run/'manifest.json'),'run_sha256':stream_hash(s.run/'headroom/run.json'),'entry_exit_integrity_passed':True})
            print('96 development baseline/reference answers complete; heldout untouched.');return
        if args.stage=='judge':
            lock(s,'calibration');predictions(s);judge_stream(s,'headroom','primary',args.resume);s.verify();print('complete first-valid primary headroom journal');return
        if args.stage=='decide':
            lock(s,'calibration');result=headroom_result(s);s.verify();f.save_or_match(s.run/'headroom/choice.json',result)
            if result['mode']=='close':stop(s,'no_headroom')
            print(json.dumps({'mode':result['mode'],'no_fit_or_intervention_calls':True}));return
        result=report(s);s.verify()
        if result['status']!='incomplete':f.save_or_match(path(s.cfg['result']),result)
        print(result['status'])
    except (Wait,infra.ResourceWait) as exc:print('WAIT:',str(exc))
    except BaseException as exc:
        if s is not None:stop(s,'fatal_execution_or_integrity',stage=args.stage,error_type=type(exc).__name__)
        elif args.stage in ('calibration','headroom','judge'):
            run=path(RUN);guard(run)
            if run.is_dir() and not (run/'stopped.json').exists():save(run/'stopped.json',{'stop_cause':'fatal_execution_or_integrity','stage':args.stage,'error_type':type(exc).__name__})
        raise


if __name__=='__main__':main()

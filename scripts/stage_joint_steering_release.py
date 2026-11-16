"""CPU-only additive public joint release; requires committed terminal results."""
import argparse,ast,csv,hashlib,io,json,re,shutil,subprocess
from pathlib import Path
from scripts import stage_steering_release as old
from scripts import joint_gcad_common as study
from scripts.joint_release_runtime.loader import INVENTORY_EXCLUSIONS

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/steering-release-staging-03'
DEST=ROOT/'outputs/steering-release-staging-04'
REPORT='results/joint_gcad_01.json'
LEXICAL_REPORT='results/joint_gcad_lexical_audit_01.json'
MF='8a57ab029d9d2adee60be7229ee30a3ba347c9c53442bed96799e2b2316849f4'
COMMIT='763865908e5e94a020606b31ab66185360f7dd44'
H2='joint_controls/manifest.json'
NEW={'joint_controls/'+n for n in ('__init__.py','steerer.py','loader.py','example.py','presets.json','GUIDE.md','results.json','metrics.csv','NOTICE','lexical_audit.json')}
CRITICAL={t+'/'+n for t in study.TRAITS for n in ('bank.safetensors','profile.json')}|{'runtime/gcad.py','runtime/loader.py'}
sha=old.sha
read=old.read
save=old.save


def portable_steerer():
    raw=(ROOT/'scripts/joint_gcad.py').read_text();tree=ast.parse(raw)
    if not isinstance(tree.body[0],ast.Expr) or not isinstance(tree.body[0].value,ast.Constant):raise ValueError('expected registered docstring')
    original='from scripts import qwen3_gcad_port as core'
    if raw.count(original)!=1:raise ValueError('unexpected registered import')
    lines=raw.splitlines(keepends=True)
    return ('"""Portable registered joint GCAD sum hook, Apache-2.0; see NOTICE.\nOnly the module docstring/import differs from the registered implementation.\n"""\n'+''.join(lines[tree.body[0].end_lineno:]).replace(original,'from runtime import gcad as core')).encode()


def parent_checked():
    old.checked(PARENT/'release_manifest.json',MF);m=read(PARENT/'release_manifest.json')
    if m['private'] is not False or m['repo_id']!=old.PUBLIC_REPO:raise ValueError('exact original public repository required')
    wanted=set(m['files_sha256'])|{'release_manifest.json'}
    if {p.relative_to(PARENT).as_posix() for p in PARENT.rglob('*') if p.is_file()}!=wanted:raise ValueError('original payload inventory changed')
    for n,d in m['files_sha256'].items():
        if (PARENT/n).is_symlink():raise ValueError('original symlink')
        old.checked(PARENT/n,d)
    return m


def terminal_report():
    old.committed(REPORT);r=read(ROOT/REPORT)
    if r['status'] not in ('complete','primary_complete_duplicates_incomplete','execution_incomplete'):raise ValueError('committed terminal primary result required')
    if set(r['analysis']['metrics'])!=set(study.arms()) or any(v['n']!=24 for v in r['analysis']['metrics'].values()):raise ValueError('all measured twelve-arm results required')
    scan(r)
    m=read(ROOT/'outputs/joint-gcad-01/manifest.json')
    if r['manifest_sha256']!=sha(ROOT/'outputs/joint-gcad-01/manifest.json') or m['sources']['scripts/joint_gcad.py']!=sha(ROOT/'scripts/joint_gcad.py'):raise ValueError('registered source/result binding differs')
    return r


def terminal_audit():
    old.committed(LEXICAL_REPORT);a=read(ROOT/LEXICAL_REPORT)
    if a.get('status')!='complete_posthoc_descriptive' or a.get('post_hoc') is not True or a.get('primary_decisions_modified') is not False:raise ValueError('committed descriptive lexical audit required')
    if a['inputs_sha256'][REPORT]!=sha(ROOT/REPORT):raise ValueError('lexical audit binds different primary result')
    old.scan_json(a);return a


def scan(value):
    old.scan_json(value)
    if isinstance(value,dict):
        if set(value)&{'user','prompt','output','responses','token_ids','tokens','judge_reasons'}:raise ValueError('raw private field')
        for v in value.values():scan(v)
    elif isinstance(value,list):
        for v in value:scan(v)


def sanitize(value):
    def label(v):
        return Path(v).name+'__'+hashlib.sha256(v.encode()).hexdigest()[:8] if old.PROJECT_PATH.search(v) else v
    if isinstance(value,dict):return {label(k):sanitize(v) for k,v in value.items()}
    if isinstance(value,list):return [sanitize(v) for v in value]
    return label(value) if isinstance(value,str) else value


def metrics_csv(result,audit):
    out=io.StringIO();fields=['arm','n',*study.METRICS,'invalid','empty','caps','mean_tokens','lexicon_hits','mean_hits_per100_whitespace_words']
    writer=csv.DictWriter(out,fieldnames=fields,lineterminator="\n");writer.writeheader()
    for a,m in result['analysis']['metrics'].items():writer.writerow({'arm':a,**{k:m[k] for k in fields if k not in ('arm','mean_hits_per100_whitespace_words')},'mean_hits_per100_whitespace_words':audit['arms'][a]['mean_per_response_rate_primary']})
    return out.getvalue()


def guide_text(result,audit):
    candidates=result['analysis']['guide'];n=len(candidates);passed=sum(v['all_pass'] for v in candidates.values())
    coherence_failures=sum(not (v['quality']['coherence_absolute'] and v['quality']['coherence_relative']) for v in candidates.values())
    coherence_summary=f'all {n} failed a coherence point-estimate criterion' if coherence_failures==n else f'{coherence_failures}/{n} failed a coherence point-estimate criterion'
    lines=['# Measured joint-control guide','',f"Terminal execution status: `{result['status']}`.",'',
        f"**{passed}/{n} joint candidates met all original criteria; {coherence_summary}.**",'',
        'These are exploratory outcomes on 24 AI-authored English practical-advice questions,','not a general benchmark or validated recommended joint controller. BF16 Qwen2.5-7B,','native default system, greedy B1 cap160 seed42, repetition penalty1.05; no refit or selection.',
        'Original isolated impolite results used a different sampled cap500/B64 design.','Coherence is not accuracy/helpfulness; all traits may induce offensive style.',
        'Fixture/calibration authors and judges share a Claude model family.',
        'Four rubric scores are separate0..100; explicit trait REFUSAL contributes0 on the fixed24-item denominator.','',
        '## Construct warning: friendly swearing is not literal swearing','',
        'The original PF score is **PF (friendly rubric)**: helpful/friendly swearing, excluding user-directed insults.',
        'A lower score under joint rude style does NOT establish less literal swearing. The separate lexical audit was',
        'explicitly post-hoc after inspecting primary outputs; it replaces no original scores, criteria or verdicts.',
        'Any-hit counts and density use the unchanged eleven-word proxy, not intent or comprehensive profanity.',
        'Density is the mean of per-response100*hits/whitespace_words, not word percent; compounds may exceed100.',
        'Quoted/incidental hits, slang omissions, length and incoherence confound this literal proxy.','',
        '## All measured arms','',
        '| Preset | PF/R/S coefficients | PF (friendly rubric) | R | S | Coherence | Invalid | Caps | Any lex hit /24 | Mean hits/100 words |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fmt=lambda x:'N/A' if x is None else f'{x:.3f}'
    for a,settings in study.arms().items():
        m=result['analysis']['metrics'][a];coef='/'.join(f'{settings[t]:g}' for t in study.TRAITS)
        lines.append('| '+a+' | '+coef+' | '+' | '.join(fmt(m[t]) for t in study.METRICS)+f" | {m['invalid']} | {m['caps']} | {audit['arms'][a]['n_any_hit']} | {fmt(audit['arms'][a]['mean_per_response_rate_primary'])} |")
    lines+=['','PF scores use the friendly rubric; R=broad impoliteness; S=sarcasm. A zero knob means no injection, NOT absence of that trait.',
        'A single shared-query hook sums displacements once; this does not stack three independently registered hooks.','',
        '## Isolated cross-effects (full single minus baseline)','','| Injected trait | PF (friendly rubric) | R | S | Coherence |','| --- | ---: | ---: | ---: | ---: |']
    cross=result['analysis']['single_cross_effects']
    for t in study.TRAITS:lines.append('| '+t+' | '+' | '.join(fmt(cross.get(t,{}).get(m)) for m in study.METRICS)+' |')
    pairs=('profanity__impolite','profanity__sarcasm','impolite__sarcasm')
    lines+=['','## Within-arm trait Spearman correlations','','| Preset | Friendly PF/R | Friendly PF/S | R/S |','| --- | ---: | ---: | ---: |']
    for a in study.arms():lines.append('| '+a+' | '+' | '.join(fmt(result['analysis']['spearman'].get(pair,{}).get('within_arm',{}).get(a)) for pair in pairs)+' |')
    lines+=['','## Post-hoc lexical density: within-arm Spearman','',
        '| Preset | Friendly PF | R | S | Coherence |','| --- | ---: | ---: | ---: | ---: |']
    for a in study.arms():lines.append('| '+a+' | '+' | '.join(fmt(audit['correlations']['within_arm'][a][m]['spearman']) for m in study.METRICS)+' |')
    lines+=['','N/A means constant ranks or unavailable correlation. Pooled and prompt-centred pooled correlations remain condition-confounded; see full JSON.','',
        '## Exploratory guide criteria','','| Preset | Verdict | Failed target/quality keys or N/A |','| --- | --- | --- |']
    for a,v in result['analysis']['guide'].items():
        failed=[]
        for t,target in v['targets'].items():
            if target['pass'] is None:failed.append(t+':N/A(nonpositive single gain)')
            else:
                if target['gain']<10:failed.append(t+'.gain>=10')
                if target['gain']<target['matched_single_gain']/2:failed.append(t+'.gain>=50%matched_single')
        failed.extend(k for k,passed in v['quality'].items() if not passed)
        lines.append(f"| {a} | {'PASS (exploratory)' if v['all_pass'] else 'NOT ALL PASS'} | {'; '.join(failed) or 'none'} |")
    half_change=result['analysis']['metrics']['half_triple']['coherence']-result['analysis']['metrics']['baseline']['coherence']
    ci=result['analysis'].get('arm_gain_question_bootstrap_ci95',{}).get('half_triple',{}).get('coherence')
    lines+=['',f'Half-triple coherence change={half_change:.3f}; original relative bound=-20; unadjusted paired CI={ci}.',
        'A CI crossing the boundary is not a new quality pass. Duplicate coherence MAD is per-response, not SE of the mean; no proven-within-noise claim.',
        'All 12 fixed arms, not only passing candidates, are in [metrics.csv](metrics.csv).',
        'Targets need gain>=10 and>=50% of positive matched single gain; nonpositive single gain is N/A.',
        'Full pairs/triple use full singles; half triple uses half singles. Coherence>=50 and>=base-20, invalid<=2/24.',
        'See [aggregate results](results.json) for paired question-bootstrap intervals, factorial mixed differences,',
        'pair/full/half additivity residuals, cross-effects, calibration and separate duplicate noise.',
        'See the separately registered [post-hoc lexical audit](lexical_audit.json) for rate intervals, valid/invalid splits and all rate correlations.',
        'Within-arm correlations are descriptive. Both raw pooled and prompt-centred pooled correlations',
        'remain condition-confounded; they do not show independent knobs or causal mechanisms.',
        'Duplicate ratings never replace primary first-valid labels; any descriptive failure remains disclosed.','',
        '## Local inference','',
        'Read BOTH manifest hashes from the pinned HF revision commit message. Use real downloaded files.',
        'From the bundle root, with the already local pinned Qwen2.5-7B base snapshot:',
        '```bash','python -B -m joint_controls.example --snapshot "$LOCAL_BASE_SNAPSHOT" --preset "$MEASURED_PRESET" --joint-manifest-sha256 "$JOINT_MANIFEST_SHA"','```',
        'Set MEASURED_PRESET yourself from the measured table/presets.json. Measured does not mean qualified; none is recommended when no candidate qualified.',
        'A fresh context is required per generation; baseline has no hooks and zero traits are omitted.',
        'No implicit base download. Python3.11, torch2.6.0+cu124, Transformers4.52.3, accelerate1.7.0, safetensors, tqdm.',
        'EOS151645-as-PAD and left padding; the loader verifies exact base/tokenizer/bank identity.']
    return '\n'.join(lines)+'\n'


def revised_docs(result):
    readme=(PARENT/'README.md').read_text();methods=(PARENT/'METHODS.md').read_text()
    readme=readme.replace('No retuning, joint or independence claim.','No 4B retuning or joint claim; separate 7B joint outcomes are below; no independence claim.')
    readme=readme.replace(', and there is no retained refusal\ncontroller.','. This package contains no refusal controller.')
    readme=readme.replace('No private physics adapter','No physics adapter')
    readme=readme.replace('adapter remains private and untouched.','adapter and NF4 base are now public and separate from this package.')
    methods=methods.replace('No refusal controller is retained or packaged.','This package contains no refusal controller.').replace('private physics adapter','separate physics adapter').replace('## Refusal closures and private adapter','## Refusal closures and public physics adapter').replace('adapter remains private and untouched.','adapter and NF4 base are now public and separate from this package.')
    add='''\n## Additive joint-controls release\n\nThe original release_manifest.json is byte-identical and its full list describes\nthe original revision `'''+COMMIT+'''`, not the updated root README/METHODS.\nIts eight original critical bank/profile/core/loader hashes remain valid, so\nthe isolated loader still works at this HEAD. The new joint manifest covers\ncurrent cards, joint files and shared payload files. Read BOTH lines from this\npinned HF commit message: `Release manifest SHA256: H1` and `Joint manifest SHA256: H2`.\nH1 is the original manifest; H2 is joint_controls/manifest.json. Optional root
.gitattributes and .git metadata, downloader .cache and Python __pycache__ are
not payload files and are excluded only from consumer inventory checks. Root
preserves any Hub-managed .gitattributes and verifies it separately unchanged. These check\nfile consistency for the pinned revision you choose, not independent publisher identity.\n\n[Joint guide](joint_controls/GUIDE.md), [all 12 arm metrics](joint_controls/metrics.csv),\nand [sanitized aggregate](joint_controls/results.json), and [post-hoc lexical audit](joint_controls/lexical_audit.json) report all outcomes, paired\nintervals, calibration/duplicate noise, cross-effects and confounding limits.\nPF means friendly-rubric swearing, not literal profanity incidence: a lower friendly PF score
does not establish suppressed swearwords. The lexical addendum is explicitly post-hoc,
using an unchanged narrow proxy; all original guide/gate verdicts stay unchanged.
No new base weights, LoRA, raw questions, answers, token arrays or private grading reasons are included.\nThe NF4 base and PRL Writing QLoRA adapter are now public; they are separate\nartifacts and are not bundled here. This package contains no refusal controller.\n'''
    return readme+add,methods+add


def preflight(root):
    root=Path(root);m=read(root/H2);parent=parent_checked();wanted=set(parent['files_sha256'])|{'release_manifest.json'}|NEW|{H2}
    if m['private'] is not False or m['status']!='public_joint_addendum' or m['repo_id']!=old.PUBLIC_REPO or m['parent_manifest_sha256']!=MF or m['parent_commit']!=COMMIT:raise ValueError('joint identity mismatch')
    if m.get('consumer_inventory_exclusions')!=INVENTORY_EXCLUSIONS:raise ValueError('unexpected consumer metadata policy')
    if m['builder_sha256']!=sha(Path(__file__)) or m['registered_joint_hook_sha256']!=sha(ROOT/'scripts/joint_gcad.py'):raise ValueError('builder/registered hook differs')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=wanted or any(p.is_symlink() for p in root.rglob('*')) or set(m['files_sha256'])!=wanted-{H2}:raise ValueError('exact additive allowlist required')
    for n,d in m['files_sha256'].items():old.checked(root/n,d)
    old.checked(root/'release_manifest.json',MF)
    for n in wanted-NEW-{H2,'README.md','METHODS.md'}:
        if (root/n).read_bytes()!=(PARENT/n).read_bytes():raise ValueError('original immutable payload changed')
    if (root/'joint_controls/steerer.py').read_bytes()!=portable_steerer():raise ValueError('unreviewed public hook adaptation')
    for name in ('__init__.py','loader.py','example.py'):
        if (root/'joint_controls'/name).read_bytes()!=(ROOT/'scripts/joint_release_runtime'/name).read_bytes():raise ValueError('unreviewed joint runtime')
    if read(root/'joint_controls/presets.json')!=study.arms():raise ValueError('fixed measured presets differ')
    for n in wanted:
        p=root/n
        if p.suffix=='.safetensors':continue
        raw=p.read_text()
        if old.BAD_TEXT.search(raw):raise ValueError('private path/secret')
        if p.suffix=='.py':ast.parse(raw)
        if n in ('README.md','METHODS.md') or n.startswith('joint_controls/'):
            if p.suffix in ('.md','.json') and old.PROJECT_PATH.search(raw):raise ValueError('private project path')
            if p.suffix=='.json':scan(read(p))
            if p.suffix=='.md':old.validate_public_links(root,p,raw)
    r=terminal_report();audit=terminal_audit();public=sanitize(r);public['private_source_result_sha256']=sha(ROOT/REPORT)
    lexical=sanitize(audit);lexical['private_source_audit_sha256']=sha(ROOT/LEXICAL_REPORT)
    if read(root/'joint_controls/lexical_audit.json')!=lexical:raise ValueError('derived lexical audit mismatch')
    if read(root/'joint_controls/results.json')!=public or (root/'joint_controls/metrics.csv').read_text()!=metrics_csv(r,audit) or (root/'joint_controls/GUIDE.md').read_text()!=guide_text(r,audit):raise ValueError('derived result/guide mismatch')
    return {'pass':True,'files':len(wanted),'joint_manifest_sha256':sha(root/H2),'parent_manifest_sha256':MF}


def stage():
    r=terminal_report();audit=terminal_audit();parent=parent_checked();dest=DEST
    outside=dest.with_name(dest.name+'-publication-manifest.json')
    if dest.exists() or outside.exists():raise FileExistsError('fresh staging04 only')
    dest.mkdir(mode=0o700)
    for n in set(parent['files_sha256'])|{'release_manifest.json'}:
        p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/n,p)
    joint=dest/'joint_controls';joint.mkdir()
    for name in ('__init__.py','loader.py','example.py'):shutil.copyfile(ROOT/'scripts/joint_release_runtime'/name,joint/name)
    (joint/'steerer.py').write_bytes(portable_steerer());save(joint/'presets.json',study.arms())
    public=sanitize(r);public['private_source_result_sha256']=sha(ROOT/REPORT)
    save(joint/'results.json',public);lexical=sanitize(audit);lexical['private_source_audit_sha256']=sha(ROOT/LEXICAL_REPORT);save(joint/'lexical_audit.json',lexical)
    (joint/'metrics.csv').write_text(metrics_csv(r,audit));(joint/'GUIDE.md').write_text(guide_text(r,audit))
    (joint/'NOTICE').write_text('Joint GCAD derived from the registered shared-query implementation.\nOnly import/docstring changed for portability. GCAD upstream commit6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de.\nApache-2.0; original LICENSE/NOTICE and Persona Vectors attribution remain in the bundle root.\n')
    readme,methods=revised_docs(r);(dest/'README.md').write_text(readme);(dest/'METHODS.md').write_text(methods)
    save(dest/H2,{'schema_version':1,'private':False,'status':'public_joint_addendum','repo_id':old.PUBLIC_REPO,'parent_manifest_sha256':MF,'parent_commit':COMMIT,
        'consumer_inventory_exclusions':INVENTORY_EXCLUSIONS,'builder_sha256':sha(Path(__file__)),'registered_joint_hook_sha256':sha(ROOT/'scripts/joint_gcad.py'),'public_transform':'Only import/docstring; behavior unchanged.',
        'source_lexical_audit_sha256':sha(ROOT/LEXICAL_REPORT),'source_result_sha256':sha(ROOT/REPORT),'files_sha256':{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}})
    result=preflight(dest)
    save(outside,{'repo_id':old.PUBLIC_REPO,'private':False,'files':{p.relative_to(dest).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in dest.rglob('*') if p.is_file()}})
    return {**result,'publication_manifest':str(outside.relative_to(ROOT)),'required_commit_message':f'Release manifest SHA256: {MF}\nJoint manifest SHA256: {result["joint_manifest_sha256"]}','uploaded':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['stage','verify']);a=p.parse_args()
    print(json.dumps(stage() if a.command=='stage' else preflight(DEST),indent=2))

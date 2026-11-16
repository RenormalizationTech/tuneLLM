"""Pinned local-only additive joint loader; no downloads or weight loading."""
import hashlib
import importlib
import json
from pathlib import Path

PARENT_SHA='8a57ab029d9d2adee60be7229ee30a3ba347c9c53442bed96799e2b2316849f4'
TRAITS=('profanity','impolite','sarcasm')
CRITICAL={t+'/'+n for t in TRAITS for n in ('bank.safetensors','profile.json')}|{'runtime/gcad.py','runtime/loader.py'}
JOINT={'__init__.py','steerer.py','loader.py','example.py','presets.json','GUIDE.md','results.json','metrics.csv','NOTICE','lexical_audit.json'}
INVENTORY_EXCLUSIONS=['root .gitattributes','root .git/**','root .cache/**','**/__pycache__/**']

def consumer_metadata(relative):
    return relative.as_posix()=='.gitattributes' or relative.parts[0] in ('.git','.cache') or '__pycache__' in relative.parts

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(Path(p).read_text(),object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def checked(root,name,digest):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('unsafe manifest path')
    file=root/p
    if file.is_symlink() or file.resolve()!=file.absolute() or sha(file)!=digest:raise ValueError('file hash/path differs: '+name)
    return file

def verify_bundle(root,*,expected_joint_manifest_sha256):
    root=Path(root).absolute()
    if root.resolve()!=root or root.is_symlink():raise ValueError('ordinary local bundle root required')
    checked(root,'joint_controls/manifest.json',expected_joint_manifest_sha256)
    m=read(root/'joint_controls/manifest.json')
    if m['private'] is not False or m['status']!='public_joint_addendum' or m['parent_manifest_sha256']!=PARENT_SHA:raise ValueError('wrong joint manifest identity')
    if m['parent_commit']!='763865908e5e94a020606b31ab66185360f7dd44' or m['repo_id']!='RenormalizationTech/qwen2.5-7b-gcad-steering-controls':raise ValueError('wrong original repository/commit')
    old=read(checked(root,'release_manifest.json',PARENT_SHA))
    expected=set(old['files_sha256'])|{'release_manifest.json'}|{'joint_controls/'+n for n in JOINT}
    if set(m['files_sha256'])!=expected:raise ValueError('complete H2 inventory required')
    if m.get('consumer_inventory_exclusions')!=INVENTORY_EXCLUSIONS:raise ValueError('unexpected consumer metadata policy')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not consumer_metadata(p.relative_to(root))}
    if actual!=expected|{'joint_controls/manifest.json'}:raise ValueError('unexpected payload files')
    for n,d in m['files_sha256'].items():checked(root,n,d)
    for n in CRITICAL:
        checked(root,n,old['files_sha256'][n])
        if m['files_sha256'][n]!=old['files_sha256'][n]:raise ValueError('original critical file differs')
    return root,m

def load_controller(root,preset,model,tokenizer,snapshot,*,base_id,revision,expected_joint_manifest_sha256):
    root,m=verify_bundle(root,expected_joint_manifest_sha256=expected_joint_manifest_sha256)
    presets=read(root/'joint_controls/presets.json')
    if preset not in presets:raise ValueError('only measured fixed presets allowed')
    original=importlib.import_module('runtime.loader')
    portable=importlib.import_module('joint_controls.steerer')
    for module,file in ((original,'runtime/loader.py'),(portable,'joint_controls/steerer.py')):
        if Path(module.__file__).resolve()!=(root/file).resolve():raise ValueError('imported runtime comes from another directory')
    core=importlib.import_module('runtime.gcad')
    if Path(core.__file__).resolve()!=root/'runtime/gcad.py' or portable.core is not core or original.MultiLayerGatedSteerer is not core.MultiLayerGatedSteerer:raise ValueError('joint/original core module mismatch')
    components={}
    for trait in TRAITS:
        profile,tensors=original.profile_and_tensors(root,trait,expected_manifest_sha256=PARENT_SHA)
        original.validate_base(model,tokenizer,snapshot,profile,base_id=base_id,revision=revision)
        components[trait]={**tensors,'base_coef':presets[preset][trait],'scale':1.5,'inverse':False}
    return portable.controller(model,components,layer_indices=list(range(9,20)))

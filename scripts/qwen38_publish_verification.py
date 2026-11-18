"""Prepare/publish completed reload evidence as a metadata-only HF commit.

Original publication receipts/manifests remain immutable. The new revision cites
the actual GPU-tested revision and proves unchanged runtime bytes; no new quality
or GPU check is inferred from the documentation update.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil

from scripts.qwen38_science_runtime import sha,write
from scripts.qwen38_verify_public import REPOS,download_record,publication
from scripts.stage_physics_release import json_bytes,private_preflight

ROOT=Path(__file__).resolve().parents[1]
CHANGED_FILES={'README.md','verification.json'}


def verification_section(kind,revision):
    text=(f'\n## Published-artifact verification\n\n'
        f'Anonymous download of commit `{revision}` verified every published file. '
        'The downloaded model then passed the recorded GPU functional reload check. '
        'For adapters, the documented PEFT loading route restored the selected weights exactly. '
        'See `verification.json` for the scope and original commit. This documentation update '
        'records that earlier check; runtime files retain their verified hashes. '
        'The original strict cache-equivalence failure remains unresolved.\n')
    if kind in ['nf4','gguf']:
        text=text.replace('For adapters, the documented PEFT loading route restored the selected weights exactly. ','')
    return text.encode()


def validate_update_plan(config,published,downloaded,changed):
    if set(config['changed_files'])!=CHANGED_FILES or set(changed)!=CHANGED_FILES:
        raise ValueError('only README.md and verification.json may change')
    if config['parent_revision']!=published['revision'] or downloaded['revision']!=published['revision']:
        raise ValueError('parent revision must be the actual GPU-tested publication')
    if not set(published['files'])<=set(downloaded['files']) or set(downloaded['files'])-set(published['files'])-{'.gitattributes'}:
        raise ValueError('downloaded file set differs from tested publication')
    for name,value in published['files'].items():
        if any(downloaded['files'][name][k]!=value[k] for k in ['bytes','sha256']):
            raise ValueError('runtime hashes differ from original tested publication')
    expected={n:{'bytes':r['bytes'],'sha256':r['sha256']} for n,r in downloaded['files'].items()}
    expected.update(changed)
    if config['expected_files']!=expected:raise ValueError('prepared file hashes differ from tested runtime plus two metadata files')
    return expected


def evidence(kind,published,downloaded,reloaded):
    if (kind not in REPOS or published['repo_id']!=REPOS[kind] or published['revision']!=downloaded['revision']
            or downloaded['kind']!=kind or reloaded['kind']!=kind
            or not downloaded.get('completed') or not downloaded.get('anonymous_access_verified')
            or not downloaded.get('all_file_hashes_verified') or not reloaded.get('completed')):
        raise ValueError('complete anonymous download and GPU verification of the exact release required')
    if kind in ['nf4','gguf'] and not reloaded.get('exact_smoke_token_replay'):
        raise ValueError('base smoke token replay required')
    if kind in ['qasper','prl'] and (not reloaded.get('adapter_tensor_equality') or not reloaded.get('public_route')):
        raise ValueError('selected adapter public-route tensor equality required')
    return {'repo_id':REPOS[kind],'gpu_verified_revision':published['revision'],
            'anonymous_download_all_file_hashes_verified':True,'gpu_reload':{k:v for k,v in reloaded.items() if k!='predictions'},
            'scope':'Two synthetic functional probes; not another quality benchmark. Same-backend base token replay or selected-adapter tensor equality as recorded.',
            'known_limit':'The original strict cached/full-prefix logit check failed at unchanged tolerance; token replay does not establish numerical cache equivalence.'}


def prepare(args):
    folder=ROOT/args.output
    if not folder.resolve().is_relative_to(ROOT/'outputs'):raise ValueError('metadata preparation belongs inside outputs')
    if folder.exists():raise FileExistsError('preserve previous metadata preparation')
    original=publication(args.publication,args.kind)
    verification=ROOT/args.verification
    downloaded=json.loads((verification/'download.json').read_text())
    reloaded=json.loads((verification/'result.json').read_text())
    if (sha(args.publication)!=downloaded['publication_sha256']
            or reloaded['download_sha256']!=sha(verification/'download.json')):
        raise ValueError('publication/download/reload binding differs')
    report=evidence(args.kind,original,downloaded,reloaded)
    report.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        original_publication_sha256=sha(args.publication),download_sha256=sha(verification/'download.json'),
        gpu_result_sha256=sha(verification/'result.json'),metadata_update_source_sha256=sha(__file__))
    payload=ROOT/'artifacts/releases'/REPOS[args.kind].split('/')[1]
    old=(payload/'README.md').read_bytes()
    if sha(payload/'README.md')!=original['files']['README.md']['sha256']:
        raise ValueError('canonical README changed before metadata preparation')
    new=old+verification_section(args.kind,original['revision']);private_preflight(new,ROOT)
    folder.mkdir(parents=True)
    (folder/'README.previous.md').write_bytes(old)
    (folder/'README.md').write_bytes(new)
    (folder/'verification.json').write_bytes(json_bytes(report,ROOT))
    expected={n:{'bytes':r['bytes'],'sha256':r['sha256']} for n,r in downloaded['files'].items()}
    for name in ['README.md','verification.json']:
        expected[name]={'bytes':(folder/name).stat().st_size,'sha256':sha(folder/name)}
    config={'repo_id':REPOS[args.kind],'kind':args.kind,'parent_revision':original['revision'],
        'publication':str(args.publication.resolve()),'publication_sha256':sha(args.publication),
        'gpu_verification_dir':str(verification),'gpu_result_sha256':sha(verification/'result.json'),
        'canonical_payload':str(payload),'source_sha256':sha(__file__),
        'changed_files':{n:sha(folder/n) for n in ['README.md','verification.json']},'expected_files':expected}
    write(folder/'prepared.json',config)
    return {'prepared':str(folder),'repo_id':config['repo_id'],'changed_files':list(config['changed_files'])}


def publish(folder):
    folder=(ROOT/folder).resolve();config=json.loads((folder/'prepared.json').read_text())
    if config['repo_id']!=REPOS[config['kind']] or sha(__file__)!=config['source_sha256']:
        raise ValueError('metadata source/destination differs')
    if set(config['changed_files'])!=CHANGED_FILES:raise ValueError('only README.md and verification.json may change')
    if sha(config['publication'])!=config['publication_sha256'] or sha(Path(config['gpu_verification_dir'])/'result.json')!=config['gpu_result_sha256']:
        raise ValueError('verification evidence changed')
    for name,digest in config['changed_files'].items():
        if sha(folder/name)!=digest:raise ValueError('reviewed metadata changed')
    published=publication(Path(config['publication']),config['kind'])
    verification=Path(config['gpu_verification_dir'])
    downloaded=json.loads((verification/'download.json').read_text())
    reloaded=json.loads((verification/'result.json').read_text())
    if (downloaded['publication_sha256']!=sha(config['publication']) or reloaded['download_sha256']!=sha(verification/'download.json')):
        raise ValueError('original publication/download/GPU evidence differs')
    prepared_report=json.loads((folder/'verification.json').read_text())
    reconstructed=evidence(config['kind'],published,downloaded,reloaded)
    reconstructed.update(checked_at_utc=prepared_report['checked_at_utc'],
        original_publication_sha256=sha(config['publication']),download_sha256=sha(verification/'download.json'),
        gpu_result_sha256=sha(verification/'result.json'),metadata_update_source_sha256=sha(__file__))
    if prepared_report!=json.loads(json_bytes(reconstructed,ROOT)):
        raise ValueError('prepared verification report differs from actual GPU evidence')
    previous=folder/'README.previous.md'
    if (sha(previous)!=published['files']['README.md']['sha256']
            or (folder/'README.md').read_bytes()!=previous.read_bytes()+verification_section(config['kind'],published['revision'])):
        raise ValueError('prepared README is not the exact verified appendix')
    changed={n:{'bytes':(folder/n).stat().st_size,'sha256':sha(folder/n)} for n in CHANGED_FILES}
    validate_update_plan(config,published,downloaded,changed)
    if Path(config['canonical_payload']).resolve()!=ROOT/'artifacts/releases'/REPOS[config['kind']].split('/')[1]:
        raise ValueError('canonical destination differs from approved model')
    os.environ['HF_HOME']=str(ROOT/'.hf')
    from huggingface_hub import HfApi,CommitOperationAdd
    api=HfApi();head=api.model_info(config['repo_id'],files_metadata=True)
    if head.private:raise ValueError('repository is no longer public')
    if head.sha==config['parent_revision']:
        commit=api.create_commit(repo_id=config['repo_id'],repo_type='model',parent_commit=head.sha,
            operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(folder/n)) for n in config['changed_files']],
            commit_message='Document anonymous pinned-download and GPU reload verification')
        revision=commit.oid
    else:
        # Exact listing/hash verification below safely recognizes a completed
        # commit after interruption; any unrelated changes fail verification.
        revision=head.sha
    record={'repo_id':config['repo_id'],'revision':revision,'files':config['expected_files']}
    snapshot,files=download_record(record)
    if files!=config['expected_files']:raise ValueError('metadata-only revision contains unexpected bytes/files')
    payload=Path(config['canonical_payload'])
    for name in config['expected_files']:
        if name not in config['changed_files'] and (payload/name).exists() and sha(payload/name)!=config['expected_files'][name]['sha256']:
            raise ValueError('canonical runtime file changed')
    for name in config['expected_files']:
        if name in config['changed_files'] or not (payload/name).exists():
            shutil.copyfile(snapshot/name,payload/name)
    result={'status':'published_and_verified','repo_id':config['repo_id'],'private':False,'revision':revision,
        'url':'https://huggingface.co/'+config['repo_id'],'gpu_verified_revision':config['parent_revision'],
        'metadata_only_update':True,'runtime_files_unchanged':True,'files':files,
        'verification_evidence_sha256':sha(folder/'verification.json'),'prepared_sha256':sha(folder/'prepared.json'),
        'gpu_result_sha256':config['gpu_result_sha256'],'quality_or_gpu_rerun_for_documentation_update':False,
        'canonical_payload':str(payload),'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    write(folder/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--publish',type=Path)
    p.add_argument('--kind',choices=list(REPOS))
    p.add_argument('--publication',type=Path)
    p.add_argument('--verification',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.publish:result=publish(a.publish)
    elif a.prepare and a.kind and a.publication and a.verification and a.output:result=prepare(a)
    else:p.error('prepare requires kind/publication/verification/output, or choose prepared publish directory')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__':main()

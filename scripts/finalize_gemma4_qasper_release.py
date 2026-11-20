"""Record an already published and GPU-verified Gemma adapter in the catalog.

CPU-only metadata reconciliation. Never uploads, generates, changes model
files, or modifies an existing release entry. Original receipts stay untouched.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
REPO="RenormalizationTech/Gemma-4-31B-tuneLLM-QASPER-QLoRA"
NAME=REPO.split("/")[1]
RELEASE_REG_SHA="c235df4dad91c8b83e70105ca3aef8d963182f52b52ac668a80d728c9eae2e9f"


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):value.update(block)
    return value.hexdigest()


def json_bytes(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()


def publish_local(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError("existing local release evidence differs: "+str(path))
    else:
        with path.open("xb") as stream:stream.write(raw)


def finalize(stage):
    stage=Path(stage).resolve();manifest=read(stage/"publication_manifest.json")
    anchor=manifest["release_manifest_sha256"]
    registration=ROOT/"results/protocols/gemma4-qasper-release-01.json"
    if sha(registration)!=RELEASE_REG_SHA:raise ValueError("release registration changed")
    registered=read(registration)
    if sha(stage/"publication_manifest.json")!=registered["publication_manifest_sha256"] or sha(stage/"payload/release_manifest.json")!=anchor or anchor!=registered["release_manifest_sha256"]:
        raise ValueError("reviewed release manifest differs")
    for name,digest in registered["sources"].items():
        if sha(ROOT/name)!=digest:raise ValueError("reviewed release source changed")
    if sha(ROOT/"results/gemma4_qasper_final_01.json")!=registered["canonical_result_sha256"]:
        raise ValueError("canonical study result changed")
    for name,digest in registered["review_sha256"].items():
        if sha(ROOT/name)!=digest:raise ValueError("registered source review changed")
    publication_path=ROOT/"results/gemma4_qasper_huggingface_publication_01.json"
    publication=read(publication_path)
    reference=read(stage/"local-reference-01.json")
    local=read(stage/"local-portable-01.json")
    download=read(stage/"anonymous-download-01.json")
    public=read(stage/"anonymous-portable-01.json")
    expected=manifest["files"]
    if manifest["repo_id"]!=REPO or manifest["private"]is not False:raise ValueError("release identity differs")
    if publication.get("status")!="published_remote_hashes_verified" or publication.get("private")is not False:
        raise ValueError("publication not verified")
    if publication["repo_id"]!=REPO or publication["manifest_sha256"]!=sha(stage/"publication_manifest.json"):
        raise ValueError("publication manifest differs")
    if publication.get("additional_remote_files") not in ([],[".gitattributes"]):raise ValueError("unreviewed remote files")
    if reference["status"]!="reference_mechanics_passed" or local["status"]!="portable_mechanics_passed":
        raise ValueError("local mechanics not passed")
    if download["status"]!="anonymous_adapter_download_verified" or public["status"]!="public_portable_mechanics_passed":
        raise ValueError("public verification not passed")
    for record in (reference,local,download,public):
        if record["release_manifest_sha256"]!=anchor:raise ValueError("verification anchor differs")
    revision=publication["revision"]
    for record in (download,public):
        if record["repo_id"]!=REPO or record["revision"]!=revision or record["anonymous_pinned_download_verified"]is not True:
            raise ValueError("anonymous pinned identity differs")
    if download.get("fresh_download_caches")is not True or download.get("download_token")is not False:
        raise ValueError("fresh anonymous cache proof absent")
    if download["files"]!=expected or set(publication["files"])!=set(expected):raise ValueError("public inventory differs")
    for name,expected_file in expected.items():
        if any(publication["files"][name][k]!=v for k,v in expected_file.items()):raise ValueError("published file differs")
        file=stage/"payload"/name
        if sha(file)!=expected_file["sha256"] or file.stat().st_size!=expected_file["bytes"]:
            raise ValueError("staged bytes changed after verification")
    if local["cells"]!=public["cells"] or len(public["cells"])!=11:raise ValueError("public/local mechanics differ")
    if set(reference["cells"])!=set(local["cells"]):raise ValueError("reference cell set differs")
    for key,cell in local["cells"].items():
        if any(reference["cells"][key][field]!=cell[field] for field in ("logits_sha256","tokens_sha256","generated_tokens","finite","saved_replay_exact","saved_decoded_output_exact")):
            raise ValueError("local/reference cell differs")
        if cell["saved_replay_exact"]!=cell["saved_decoded_output_exact"]:raise ValueError("decoded replay coverage differs")
    if public["local_receipt_sha256"]!=sha(stage/"local-portable-01.json") or public["download_receipt_sha256"]!=sha(stage/"anonymous-download-01.json"):
        raise ValueError("public receipt binding differs")
    if sum(v["saved_replay_exact"] for v in public["cells"].values())!=7:raise ValueError("saved replay coverage differs")
    for record in (reference,local,public):
        if record["base_repo_id"]!="google/gemma-4-31B-it" or record["base_revision"]!=registered["base_revision"] or not record["base_files_hash_verified"]:
            raise ValueError("verified base identity differs")
        if not record["cleanup_exact"] or not record["adapter_reload"]["exact_tensor_equality"] or record["adapter_reload"]["tensor_count"]!=820:
            raise ValueError("tensor reload or cleanup failed")
        if not record["no_optimizer_updates"] or not record["no_regrading"]:raise ValueError("mechanics scope differs")
        if record["verification_source_sha256"]!=registered["sources"]["scripts/verify_gemma4_qasper_release.py"]:
            raise ValueError("verification source differs")
    for record in (local,public):
        if record["reference_receipt_sha256"]!=sha(stage/"local-reference-01.json"):raise ValueError("reference receipt binding differs")
    if any(not v["finite"] or not v["portable_reference_exact"] for v in public["cells"].values()):raise ValueError("cell parity not passed")
    inventory=hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    if download["published_inventory_sha256"]!=inventory:raise ValueError("inventory digest differs")
    result={"status":"published_and_verified","repo_id":REPO,"url":"https://huggingface.co/"+REPO,
        "private":False,"revision":revision,"gpu_verified_revision":revision,
        "release_manifest_sha256":anchor,"publication_manifest_sha256":sha(stage/"publication_manifest.json"),
        "anonymous_pinned_download_verified":True,"fresh_adapter_download_cache":True,
        "runtime_files_unchanged":True,"public_local_mechanics_equal":True,
        "base_repo":"google/gemma-4-31B-it","base_revision":public["base_revision"],
        "base_download_scope":"Previously obtained pinned-revision base cache content-hash verified; no fresh full-base download claimed",
        "files":expected,"published_file_count":len(expected),"published_inventory_sha256":inventory,
        "verified_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence_sha256":{name:sha(stage/name) for name in ["local-reference-01.json","local-portable-01.json","anonymous-download-01.json","anonymous-portable-01.json"]},
        "publication_receipt_sha256":sha(publication_path),"finalizer_source_sha256":sha(__file__),
        "canonical_study_result_sha256":sha(ROOT/"results/gemma4_qasper_final_01.json"),
        "functional_scope":"Exact selected-adapter reload;11 logit/token cells;7 saved token-and-decoded-output replays;cleanup. Separate from the frozen efficacy study.",
        "no_new_quality_experiment":True,"cells":public["cells"]}
    receipt=ROOT/"results/gemma4_qasper_public_verification_01.json"
    if receipt.exists():
        prior=read(receipt)
        result["verified_at_utc"]=prior["verified_at_utc"]
    publish_local(receipt,json_bytes(result))
    for name in expected:
        file=stage/"payload"/name
        target=ROOT/"artifacts/releases"/NAME/name
        if target.exists():
            if sha(target)!=expected[name]["sha256"]:raise ValueError("existing local payload differs")
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        if name!="adapter_model.safetensors":publish_local(ROOT/"releases"/NAME/name,file.read_bytes())
    catalog_path=ROOT/"releases/catalog.json";catalog=read(catalog_path)
    entry={"label":"Gemma 4 31B QASPER QLoRA","repo_id":REPO,"status":"public","revision":revision,
        "gpu_verified_revision":revision,"url":result["url"],"card_path":"releases/"+NAME+"/README.md",
        "card_sha256":expected["README.md"]["sha256"],"local_payload_path":"artifacts/releases/"+NAME,
        "published_file_count":len(expected),"published_inventory_sha256":inventory,
        "verification_receipt":str(receipt.relative_to(ROOT)),"result_path":"results/gemma4_qasper_final_01.json",
        "publication_receipt":str(publication_path.relative_to(ROOT)),
        "historical_metadata_paths":["releases/"+NAME+"/"+name for name in ["LICENSE","NOTICE","evaluation_summary.json","release_manifest.json","upstream_license.json"]]}
    matches=[r for r in catalog["public_releases"] if r["repo_id"]==REPO]
    if matches:
        if matches!=[entry]:raise ValueError("existing Gemma catalog entry differs")
    else:
        catalog["public_releases"].append(entry)
        catalog["checked_at_utc"]=result["verified_at_utc"]
        catalog_path.write_bytes(json_bytes(catalog))
    return {k:result[k] for k in ["status","repo_id","revision","release_manifest_sha256","published_file_count","public_local_mechanics_equal"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",required=True)
    args=parser.parse_args();print(json.dumps(finalize(args.stage),indent=2))


if __name__=="__main__":main()

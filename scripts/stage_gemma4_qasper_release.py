"""CPU-only exact adapter staging; never publishes, loads GPU weights or rescores."""
import argparse
import ast
import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil

from scripts.gemma4_qasper_release_runtime import loader

ROOT = Path(__file__).resolve().parents[1]
REPO = "RenormalizationTech/Gemma-4-31B-tuneLLM-QASPER-QLoRA"
CANONICAL = "a149de809997eb8ec18b670a4c1d32eca13edf014bd07fc28333abad3d50b39f"
R0 = "ba1983fdb9e6ca1e11ce9648e69a4c057832d6524b5014cd99cabe3fca4ba9f0"
AMENDMENT = "29f41ff850a58d1ca5d23e6e443253d7341cc2ec84c4079776813a0a270862b2"
ANALYZER = "78506e39bed434703d2796f5e16e5824e0f01ce1c52124c2ac02450c7b3256b3"
SOURCE_NAMES = ["scripts/stage_gemma4_qasper_release.py","scripts/verify_gemma4_qasper_release.py",
    "scripts/gemma4_qasper_release_runtime/__init__.py","scripts/gemma4_qasper_release_runtime/loader.py",
    "scripts/gemma4_qasper_release_runtime/example.py","tests/test_gemma4_qasper_release.py"]


def save(path,value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def checked(path,digest):
    if loader.sha(path) != digest:
        raise ValueError("frozen binding differs: "+Path(path).name)


def file_inventory(directory):
    return {p.relative_to(directory).as_posix():{"bytes":p.stat().st_size,"sha256":loader.sha(p)}
        for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def normalized_config(original):
    result = copy.deepcopy(original)
    for key,value in {"r":16,"lora_alpha":32,"lora_dropout":.05,"bias":"none","task_type":"CAUSAL_LM","peft_type":"LORA"}.items():
        if result.get(key) != value:
            raise ValueError("selected adapter recipe differs")
    if set(result["target_modules"]) != {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}:
        raise ValueError("selected target modules differ")
    result["base_model_name_or_path"] = loader.BASE; result["revision"] = loader.REVISION
    if any(result[k] != v for k,v in original.items() if k not in {"base_model_name_or_path","revision"}):
        raise ValueError("unapproved config transformation")
    return result


class CodeBlocks(HTMLParser):
    def __init__(self):
        super().__init__(); self.blocks=[]; self.active=False; self.current=[]
    def handle_starttag(self,tag,attrs):
        if tag == "code" and not self.active:
            self.active=True; self.current=[]
    def handle_endtag(self,tag):
        if tag == "code" and self.active:
            self.blocks.append("".join(self.current)); self.active=False
    def handle_data(self,data):
        if self.active:
            self.current.append(data)


def verified_license(directory):
    directory = Path(directory); receipt = loader.read(directory/"receipt.json")
    if receipt["revision"] != loader.REVISION:
        raise ValueError("upstream license revision differs")
    for key,extension in (("metadata","json"),("card","txt"),("license","txt")):
        checked(directory/(key+"."+extension),receipt["sources"][key]["sha256"])
    metadata = loader.read(directory/"metadata.json")
    card = (directory/"card.txt").read_text()
    if (metadata["sha"] != loader.REVISION or metadata.get("gated") is not False or
        metadata["cardData"]["license"] != "apache-2.0" or "license: apache-2.0" not in card):
        raise ValueError("official pinned license metadata differs")
    names = [s["rfilename"] for s in metadata["siblings"]]
    if any(n.upper() in ("LICENSE","LICENSE.TXT","NOTICE") for n in names):
        raise ValueError("upstream legal file now present; explicitly review instead of assuming")
    parser = CodeBlocks(); parser.feed((directory/"license.txt").read_text())
    blocks = [b for b in parser.blocks if "TERMS AND CONDITIONS" in b and "END OF TERMS AND CONDITIONS" in b]
    if len(blocks) != 1 or "Version 2.0, January 2004" not in blocks[0]:
        raise ValueError("official Apache2 text extraction failed")
    text = blocks[0].strip()+"\n"
    witness = {"repo_id":loader.BASE,"revision":loader.REVISION,"license":"apache-2.0",
        "upstream_LICENSE_file_present":False,"sources":receipt["sources"],
        "license_text_sha256":hashlib.sha256(text.encode()).hexdigest(),
        "verification":"Pinned official API/card declare Apache2; pinned repository has no LICENSE/NOTICE file; LICENSE is the actual Apache2 text extracted from Google's linked license page."}
    return text,witness


def privacy_scan(payload):
    if loader.listing(payload) != loader.FILES:
        raise ValueError("public allowlist differs")
    forbidden = re.compile(r"/home/|/tmp/|/workspace/|/mnt/|[A-Za-z]:\\|hf_[A-Za-z0-9]{15,}|sk-[A-Za-z0-9]{16,}|session[_-]?id|\"pid\"|state\.pt|training_state|\.jsonl",re.I)
    for name in loader.FILES-{"adapter_model.safetensors"}:
        text = (Path(payload)/name).read_text()
        if forbidden.search(text):
            raise ValueError("private/unapproved payload content: "+name)
    # Tensor names and metadata are separately constrained to the exact LoRA inventory.


def evaluation_summary(canonical,audit_sha):
    if (not canonical["all_fixed_task_and_retention_gates_pass"] or canonical["training"]["selected_step"] != 61 or
        canonical["strict_primary_answer_f1"] != {"base":0.,"prompt":0.,"adapter":0.6400835382852015}):
        raise ValueError("canonical release outcome differs")
    arms = canonical["arms"]
    if arms["base"]["strict"]["all_sampled"]["invalid_outputs"] != 403:
        raise ValueError("baseline format floor witness differs")
    groups = {}
    for arm in ("base","prompt","adapter"):
        groups[arm] = {mode:{group:arms[arm][mode][group] for group in ("primary","evidence_strict_primary","all_sampled")}
                       for mode in ("strict","lenient")}
    retention = {name:{key:value[key] for key in ("total","baseline_correct","candidate_correct","delta_correct","allowed_loss","gate_pass")}
        for name,value in canonical["retention"]["tasks"].items()}
    return {"scope":"Custom text-only QASPER; reused cross-family project evaluation, not fresh heldout or official benchmark",
        "denominators":{"primary":309,"evidence_strict_primary":267,"all_sampled":403},
        "arms":groups,"prompt_alias":"base (all three development templates tied at strict0)",
        "baseline_format_floor":"All403 base outputs fenced; strict0 includes format compliance; lenient diagnostics are essential",
        "strict_registered_comparisons":canonical["analysis"]["answer_comparisons"],
        "registered_gate_results":canonical["analysis"]["gate_results"],
        "lenient_answer_diagnostic":{"base":.33380301366781717,"adapter":.6400835382852015,
            "gain":.3062805246173843,"ci95":[.2512991,.3616835],"source":"independent final audit; paper bootstrap; diagnostic only"},
        "evidence_interpretation":"No evidence superiority claim; lenient evidence essentially unchanged",
        "retention":retention,"retention_interpretation":"point-estimate regression limits, not non-inferiority",
        "training":{"train":482,"development":101,"epochs":1,"steps":61,"selected_step":61,"seed":42},
        "timing":{"training_gpu_hours":4.84,"study_wall_hours":8.96,"summed_gpu_hours":15.60,
            "scope":"Includes coexistence with recorded idle resident CUDA context; no exclusive timing claim"},
        "limitations":["one configuration and seed","public papers may be in pretraining","answer-style and format effects","no broad model ranking","small retention samples","no evidence superiority claim"],
        "canonical_sha256":CANONICAL,"final_audit_sha256":audit_sha}


def stage(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("fresh staging path required; preserve earlier preparations")
    study = Path(args.study).resolve(); training = study/"training"
    canonical_path = ROOT/"results/gemma4_qasper_final_01.json"; checked(canonical_path,CANONICAL)
    r0 = ROOT/"results/protocols/gemma4-qasper-01.json"; checked(r0,R0)
    amendment = ROOT/"results/protocols/gemma4-qasper-analysis-amendment-01.json"; checked(amendment,AMENDMENT)
    analysis_source = ROOT/"scripts/gemma4_qasper_analysis_amendment_01.py"; checked(analysis_source,ANALYZER)
    canonical = loader.read(canonical_path); checked(study/"analysis-amendment-01.json",canonical["analysis_sha256"])
    checked(study/"completion-seal.json",canonical["completion_seal_sha256"])
    cfg = loader.read(training/"registration.json"); result = loader.read(training/"result.json")
    if not result["completed"] or result["steps"] != 61 or result["selected"]["step"] != 61:
        raise ValueError("completed selected61 required")
    checked(training/"result.json",loader.read(study/"analysis-amendment-01.json")["training"]["training_result_sha256"])
    adapter = Path(result["selected"]["path"]).resolve()
    if adapter != training/"checkpoints/step-00061":
        raise ValueError("selected checkpoint escapes registered run")
    for name in ("adapter_config.json","adapter_model.safetensors"):
        checked(adapter/name,result["selected_files"][name]["sha256"])
    for name,expected in cfg["source_files"].items():
        checked(training/"source"/name,expected["sha256"])
    if cfg["versions"] != loader.PINS or cfg["identity"]["revision"] != loader.REVISION:
        raise ValueError("studied runtime/base differs")
    base = Path(args.base or cfg["identity"]["base"])
    loader.validate_base(base,{"base":{"files":cfg["identity"]["files"]}})
    import torch
    from transformers import Gemma4ForCausalLM,Gemma4TextConfig
    with torch.device("meta"):
        model = Gemma4ForCausalLM(Gemma4TextConfig.from_dict(loader.read(base/"config.json")["text_config"]))
    targets = loader.target_inventory(model)
    if targets != cfg["targets"] or len(targets) != 410:
        raise ValueError("fresh CPU meta geometry differs")
    inventory = loader.adapter_inventory(adapter/"adapter_model.safetensors",targets)
    if inventory["tensor_count"] != 820 or inventory["parameters"] != 122429440:
        raise ValueError("adapter LoRA inventory differs")
    legal_text,legal = verified_license(args.license_evidence)
    original = loader.read(adapter/"adapter_config.json"); normalized = normalized_config(original)
    audit = ROOT/"results/reviews/gemma4_qasper_final_audit_review_02.md"
    summary = evaluation_summary(canonical,loader.sha(audit))
    output.mkdir(parents=True); payload = output/"payload"; (payload/"runtime").mkdir(parents=True)
    shutil.copyfile(adapter/"adapter_model.safetensors",payload/"adapter_model.safetensors")
    save(payload/"adapter_config.json",normalized)
    shutil.copyfile(args.card,payload/"README.md")
    (payload/"LICENSE").write_text(legal_text)
    (payload/"NOTICE").write_text("Gemma QASPER QLoRA adapter and portable local runtime.\nBase model: Google DeepMind google/gemma-4-31B-it at "+loader.REVISION+".\nUpstream pinned card declares Apache-2.0; see upstream_license.json and LICENSE.\nUpstream weights are not redistributed here. Adapter trained by RenormalizationTech.\nModified adapter_config.json: only base_model_name_or_path and revision normalized for portability.\nPortable inference code independently packages the studied text-only loading/generation recipe.\nOfficial license: https://ai.google.dev/gemma/docs/gemma_4_license\nPinned card: https://huggingface.co/google/gemma-4-31B-it/blob/"+loader.REVISION+"/README.md\n")
    save(payload/"upstream_license.json",legal); save(payload/"evaluation_summary.json",summary)
    (payload/"requirements.txt").write_text("\n".join(name+"=="+version for name,version in loader.PINS.items())+"\n")
    for name in ("__init__.py","loader.py","example.py"):
        shutil.copyfile(ROOT/"scripts/gemma4_qasper_release_runtime"/name,payload/"runtime"/name)
    manifest = {"schema_version":1,"repo_id":REPO,"private":False,
        "base":{"repo_id":loader.BASE,"revision":loader.REVISION,"files":cfg["identity"]["files"]},
        "adapter":{**inventory,"config":normalized,"source_weights_sha256":loader.sha(adapter/"adapter_model.safetensors"),
            "source_config_sha256":loader.sha(adapter/"adapter_config.json"),"config_allowed_changes":["base_model_name_or_path","revision"]},
        "targets":targets,"environment":loader.PINS,"files":file_inventory(payload),
        "provenance":{"study_R0_sha256":R0,"corrected_analysis_source_sha256":ANALYZER,
            "analysis_amendment_sha256":AMENDMENT,"canonical_sha256":CANONICAL,"final_audit_sha256":loader.sha(audit),
            "training_registration_sha256":loader.sha(training/"registration.json"),"training_result_sha256":loader.sha(training/"result.json"),
            "completion_seal_sha256":canonical["completion_seal_sha256"],"frozen_runtime_sha256":loader.sha(training/"source/scripts/gemma4_qasper_runtime.py"),
            "sources":{name:loader.sha(ROOT/name) for name in SOURCE_NAMES}},
        "inference":{"cap":512,"context":8192,"batch_size":1,"cache":True,"thinking":False,"eos":[1,106],
            "greedy":True,"repetition_penalty":1.0,"outer_bf16_autocast":True,"kbit_floating_casts":"FP32","attention":"sdpa"}}
    save(payload/"release_manifest.json",manifest)
    anchor = loader.sha(payload/"release_manifest.json")
    loader.validate_bundle(payload,anchor); privacy_scan(payload)
    publication = {"repo_id":REPO,"private":False,"files":file_inventory(payload),"release_manifest_sha256":anchor,
        "required_commit_message":"Release manifest SHA256: "+anchor}
    save(output/"publication_manifest.json",publication)
    receipt = {"status":"staged_CPU_only","payload":str(payload),"release_manifest_sha256":anchor,
        "publication_manifest_sha256":loader.sha(output/"publication_manifest.json"),"adapter_weights_byte_exact":True,
        "adapter_inventory":inventory,"privacy_scan":"passed exact allowlist","license":"verified official pinned metadata and actual Apache2 text",
        "GPU_verified":False,"published":False,"builder_sha256":loader.sha(__file__)}
    save(output/"staging_receipt.json",receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default=str(ROOT/"outputs/gemma4-qasper-release-01"))
    p.add_argument("--study",default=str(ROOT/"outputs/gemma4-qasper-study-01")); p.add_argument("--base")
    p.add_argument("--card",default=str(ROOT/"docs/gemma4_qasper_hf_card_draft.md"))
    p.add_argument("--license-evidence",default=str(ROOT/"outputs/gemma4-qasper-release-pm-01/license-verification-root-01"))
    print(json.dumps(stage(p.parse_args()),indent=2))


if __name__ == "__main__":
    main()

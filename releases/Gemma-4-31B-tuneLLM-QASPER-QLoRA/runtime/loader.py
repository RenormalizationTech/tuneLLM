"""Portable, offline-only studied Gemma QASPER inference (Apache-2.0).

No model downloads, remote code, merged weights or project imports. A reviewed
external manifest hash and exact local base bytes are required. See NOTICE.
"""
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

BASE = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
PINS = {"torch":"2.12.0+cu130","transformers":"5.23.0","peft":"0.21.0",
        "bitsandbytes":"0.49.2","accelerate":"1.12.0","huggingface_hub":"2.4.0"}
FILES = frozenset({"adapter_config.json","adapter_model.safetensors","README.md","LICENSE","NOTICE",
    "evaluation_summary.json","upstream_license.json","requirements.txt","runtime/__init__.py",
    "runtime/loader.py","runtime/example.py","release_manifest.json"})
BASE_EXTRAS = {"README.md","processor_config.json"}


def format_question(paragraphs,question):
    """Exact frozen QASPER user prefix/context layout; no selection or rewriting.

    Supply only the intended context paragraphs in display order. IDs, section
    names, text and question whitespace are preserved byte for byte.
    """
    if not isinstance(paragraphs,list) or not isinstance(question,str) or not question:
        raise ValueError("paragraph list and nonempty question string required")
    if any(not isinstance(p,dict) or any(not isinstance(p.get(k),str) for k in ("id","section","text")) for p in paragraphs):
        raise ValueError("each paragraph requires string id, section and text")
    context = "\n\n".join(f"[{p['id']}] {p['section']}\n{p['text']}" for p in paragraphs)
    return ("Answer using only the numbered paper paragraphs. Return only JSON with keys "
        '"answer" (string), "unanswerable" (boolean), "evidence" (array of paragraph IDs). '
        "For insufficient evidence return an empty answer and evidence array with unanswerable=true.\n\n"
        +context+"\n\nQuestion: "+question)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite JSON")))


def regular(path):
    return stat.S_ISREG(Path(path).lstat().st_mode)


def real_directory(path):
    return Path(path).is_dir() and not Path(path).is_symlink()


def safe_file(directory,name):
    """Real local_dir files or constrained official HF one/two-hop cache links.

    HF2.4 shared stores require their exact marker and digest prefix. Arbitrary
    outside links, directory links, malformed digest names and third hops fail.
    All accepted files still undergo content SHA256 verification.
    """
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in name:
        raise ValueError("unsafe file member")
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("directory symlink refused")
    root = root.resolve(strict=True); candidate = root/name
    for parent in relative.parents:
        if parent != PurePosixPath(".") and not real_directory(root/str(parent)):
            raise ValueError("directory symlink refused")
    if not candidate.is_symlink():
        if not regular(candidate):
            raise ValueError("regular file required")
        return candidate
    repo = root.parent.parent
    if root.parent.name != "snapshots" or not repo.name.startswith("models--") or not re.fullmatch(r"[0-9a-f]{40}",root.name):
        raise ValueError("outside file symlink refused")
    if not real_directory(repo/"blobs") or not real_directory(root.parent) or not real_directory(repo):
        raise ValueError("cache directory symlink refused")
    first = Path(os.path.abspath(candidate.parent/os.readlink(candidate)))
    if first.parent != repo/"blobs" or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}",first.name):
        raise ValueError("invalid repository blob link")
    if not first.is_symlink():
        if not regular(first):
            raise ValueError("regular repository blob required")
        return first
    cache = repo.parent; store = cache/"blobs"
    second = Path(os.path.abspath(first.parent/os.readlink(first)))
    digest = second.name
    if (not re.fullmatch(r"[0-9a-f]{64}",digest) or
        second != store/digest[:2]/digest or not real_directory(store) or
        not real_directory(store/digest[:2])):
        raise ValueError("invalid shared blob path")
    marker = store/".huggingface-shared-blobs"
    if not regular(marker) or marker.read_bytes() != b"1\n" or not regular(second):
        raise ValueError("invalid shared blob marker or third link")
    if not second.resolve().is_relative_to(cache.resolve()):
        raise ValueError("shared blob escape")
    return second


def verify_files(directory,files):
    for name,item in files.items():
        file = safe_file(directory,name)
        if file.stat().st_size != item["bytes"] or sha(file) != item["sha256"]:
            raise ValueError("file hash/size differs: "+name)


def listing(directory):
    result = set()
    for path in Path(directory).rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if relative == ".gitattributes":
            continue
        if relative == ".cache" or relative.startswith(".cache/huggingface"):
            if path.is_symlink():
                raise ValueError("metadata symlink refused")
            continue
        if path.is_symlink() and path.is_dir():
            raise ValueError("directory symlink refused")
        if path.is_file() or path.is_symlink():
            result.add(relative)
    return result


def validate_bundle(bundle,manifest_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}",manifest_sha256 or ""):
        raise ValueError("external manifest SHA256 required")
    path = safe_file(bundle,"release_manifest.json")
    if sha(path) != manifest_sha256:
        raise ValueError("release manifest anchor differs")
    manifest = read(path)
    if listing(bundle) != FILES or set(manifest["files"]) != FILES-{"release_manifest.json"}:
        raise ValueError("payload allowlist differs")
    if manifest["base"]["repo_id"] != BASE or manifest["base"]["revision"] != REVISION:
        raise ValueError("pinned base identity differs")
    if manifest["environment"] != PINS or manifest["adapter"]["tensor_count"] != 820 or manifest["adapter"]["parameters"] != 122429440:
        raise ValueError("runtime/adapter geometry identity differs")
    verify_files(bundle,manifest["files"])
    config = read(safe_file(bundle,"adapter_config.json"))
    if config != manifest["adapter"]["config"] or config["base_model_name_or_path"] != BASE or config["revision"] != REVISION:
        raise ValueError("portable adapter config differs")
    return manifest


def validate_environment():
    actual = {name:importlib.metadata.version(name) for name in PINS}
    if actual != PINS:
        raise ValueError("exact studied package versions required")
    return actual


def validate_base(base,manifest):
    expected = manifest["base"]["files"]
    extra = listing(base)-set(expected)-BASE_EXTRAS
    if extra:
        raise ValueError("unregistered base files: "+", ".join(sorted(extra))+"; use the card's eight-file allow_patterns in a fresh local directory")
    verify_files(base,expected)
    raw = read(safe_file(base,"config.json"))
    if raw.get("model_type") != "gemma4" or raw["text_config"].get("model_type") != "gemma4_text" or raw["text_config"]["num_hidden_layers"] != 60:
        raise ValueError("Gemma text geometry differs")
    return raw


def target_inventory(model):
    import torch
    suffixes = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    return [{"name":name,"in_features":m.in_features,"out_features":m.out_features}
        for name,m in model.named_modules() if name.rsplit(".",1)[-1] in suffixes and isinstance(m,torch.nn.Linear)]


def adapter_inventory(path,targets,check_finite=True):
    import torch
    from safetensors import safe_open
    expected = {}
    for target in targets:
        for side,shape in (("A",[16,target["in_features"]]),("B",[target["out_features"],16])):
            expected["base_model.model."+target["name"]+f".lora_{side}.weight"] = shape
    total = 0
    with safe_open(str(path),framework="pt",device="cpu") as state:
        if set(state.keys()) != set(expected):
            raise ValueError("adapter tensor keys differ")
        for name,shape in expected.items():
            value = state.get_slice(name)
            if value.get_shape() != shape or value.get_dtype() != "F32":
                raise ValueError("adapter tensor shape/dtype differs")
            if check_finite and not bool(torch.isfinite(state.get_tensor(name)).all()):
                raise ValueError("nonfinite adapter tensor")
            total += math.prod(shape)
    return {"tensor_count":len(expected),"parameters":total,"all_finite":check_finite}


def reload_adapter(model,bundle):
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    state = load_file(str(safe_file(bundle,"adapter_model.safetensors")))
    report = set_peft_model_state_dict(model,state,adapter_name="default")
    if report.unexpected_keys or any("lora_" in name for name in report.missing_keys):
        raise ValueError("adapter reload keys differ")
    actual = get_peft_model_state_dict(model)
    if set(actual) != set(state) or any(not torch.equal(actual[name].detach().cpu(),state[name]) for name in state):
        raise ValueError("adapter reload tensor equality failed")
    return {"tensor_count":len(state),"exact_tensor_equality":True}


class Runtime:
    def __init__(self,model,tokenizer,manifest,receipt=None):
        self.model = model; self.tokenizer = tokenizer; self.manifest = manifest; self.receipt = receipt or {}
        model.eval()

    def render(self,user):
        if not isinstance(user,str) or not user.strip():
            raise ValueError("nonempty user text required")
        prompt = self.tokenizer.apply_chat_template([{"role":"user","content":user}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        ids = self.tokenizer(prompt,add_special_tokens=False)["input_ids"]
        return prompt,ids

    def _batch(self,ids,cap=0):
        import torch
        if (not isinstance(ids,list) or not ids or any(type(i) is not int or i < 0 for i in ids) or
            type(cap) is not int or not 0 <= cap <= 512 or len(ids)+cap > 8192):
            raise ValueError("flat unpadded singleton token IDs, cap<=512 and context<=8192 required; no truncation")
        return torch.tensor([ids],dtype=torch.long,device=next(self.model.parameters()).device)

    def _arm(self,adapter):
        if type(adapter) is not bool:
            raise ValueError("explicit boolean adapter arm required")
        return nullcontext() if adapter else self.model.disable_adapter()

    def next_logits(self,ids,adapter=True):
        import torch
        batch = self._batch(ids)
        with self._arm(adapter),torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=batch.is_cuda):
            logits = self.model(input_ids=batch,attention_mask=torch.ones_like(batch),use_cache=False,
                                logits_to_keep=1).logits.detach().float().cpu()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite logits")
        return logits

    def generate_ids(self,ids,adapter=True,cap=512):
        import torch
        if cap == 0:
            raise ValueError("positive generation cap required")
        batch = self._batch(ids,cap)
        with self._arm(adapter),torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=batch.is_cuda):
            sequence = self.model.generate(input_ids=batch,attention_mask=torch.ones_like(batch),do_sample=False,
                num_beams=1,use_cache=True,max_new_tokens=cap,temperature=None,top_p=None,top_k=None,
                repetition_penalty=1.0,eos_token_id=[1,106],pad_token_id=self.tokenizer.pad_token_id,logits_to_keep=1)
        if not torch.equal(sequence[0,:len(ids)].cpu(),torch.tensor(ids)):
            raise ValueError("generation prompt continuity failed")
        tokens = sequence[0,len(ids):].cpu().tolist()
        ended = bool(tokens and tokens[-1] in (1,106))
        return {"generated_ids":tokens,"output":self.tokenizer.decode(tokens[:-1] if ended else tokens,skip_special_tokens=False),
                "ended_with_eos":ended,"hit_token_limit":len(tokens)==cap and not ended}


def load_runtime(bundle,base,manifest_sha256):
    """Explicit GPU load only. Users supply local bundle and pinned base."""
    manifest = validate_bundle(bundle,manifest_sha256); validate_environment()
    raw = validate_base(base,manifest)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    import torch
    from transformers import AutoTokenizer,BitsAndBytesConfig,Gemma4ForCausalLM,Gemma4TextConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one explicitly visible CUDA GPU required")
    config = Gemma4TextConfig.from_dict(raw["text_config"])
    model,info = Gemma4ForCausalLM.from_pretrained(str(base),config=config,
        key_mapping={r"^model\.language_model\.":"model."},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa",
        local_files_only=True,trust_remote_code=False,output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys","mismatched_keys","error_msgs")) or any(
        not (name.startswith("model.vision_tower.") or name.startswith("model.embed_vision.")) for name in info.get("unexpected_keys",[])):
        raise ValueError("text-only load keys differ")
    if any("vision" in name or "audio" in name for name,_ in model.named_parameters()):
        raise ValueError("multimodal parameters instantiated")
    targets = target_inventory(model)
    from bitsandbytes.nn import Linear4bit
    if targets != manifest["targets"] or len(targets) != 410 or any(not isinstance(model.get_submodule(t["name"]),Linear4bit) for t in targets):
        raise ValueError("NF4 projection geometry/classes differ")
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
    model.config.use_cache = False
    model = get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=[t["name"] for t in targets],bias="none"))
    trained = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in name for name,p in trained) or len(trained) != 820 or sum(p.numel() for name,p in trained) != 122429440:
        raise ValueError("trainable LoRA inventory differs")
    audit = reload_adapter(model,bundle)
    tokenizer = AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
    return Runtime(model,tokenizer,manifest,{"adapter_reload":audit,"outer_bf16_autocast":True,
        "floating_recipe":"PEFT FP32 kbit casts","attention":"sdpa","cache":True,"batch_size":1})

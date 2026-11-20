"""Gemma QASPER runtime; frozen preflight arithmetic is imported unchanged.

CPU preparation never loads weights. GPU entry points require a registered SHA.
Dropout is 0.05, matching Qwen; a registered longest-row check precedes step1.
"""
from contextlib import contextmanager
import ast
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.gemma4_preflight import encode, response_loss, sha, versions, PROJECTIONS

ROOT = Path(__file__).resolve().parents[1]
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
PREFLIGHT = ROOT / "outputs/gemma4-preflight-02"
CORE = ["scripts/gemma4_qasper_runtime.py", "scripts/gemma4_train_qasper.py",
        "scripts/gemma4_eval_qasper.py", "scripts/gemma4_retention.py",
        "scripts/gemma4_preflight.py", "tests/test_gemma4_qasper.py"]
OUTER_BF16_AUTOCAST = True
OCCUPANCY_SNAPSHOT = None
MAX_EXISTING_MEMORY_MIB = 1024
MAX_EXISTING_UTILIZATION_PERCENT = 5
PINS = {"torch":"2.12.0+cu130","transformers":"5.23.0","peft":"0.21.0",
        "bitsandbytes":"0.49.2","accelerate":"1.12.0","huggingface_hub":"2.4.0"}


def amp(model):
    import torch
    return torch.autocast("cuda",dtype=torch.bfloat16,
                          enabled=OUTER_BF16_AUTOCAST and next(model.parameters()).is_cuda)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def atomic_json(path, value, new=False):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    if new:
        os.link(temp, path); temp.unlink()
    else:
        os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_or_match(path,value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError("immutable publication differs")
    else:
        atomic_json(path,value,new=True)


def inventory(directory):
    return {p.relative_to(directory).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def verify_files(directory, files):
    for name, item in files.items():
        p = Path(directory) / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe manifest member")
        if not p.is_file() or p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
            raise ValueError(f"artifact changed: {name}")


def model_identity(base, preflight=PREFLIGHT):
    base = Path(base).resolve()
    if versions() != PINS:
        raise ValueError("studied preflight package pins required")
    if base.name != REVISION:
        raise ValueError("pinned Gemma revision required")
    proof = read(preflight / "weights_verification.json")
    if proof["status"] != "passed" or proof["revision"] != REVISION:
        raise ValueError("preflight weight proof mismatch")
    verify_files(base, proof["weights"])
    names = ["config.json", "generation_config.json", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    files = {n: {"sha256": sha(base / n), "bytes": (base / n).stat().st_size} for n in names}
    files.update(proof["weights"])
    return {"base": str(base), "revision": REVISION, "files": files,
            "native_generation_eos":read(base/"generation_config.json")["eos_token_id"],
            "study_eos_override":[1,106],
            "preflight_weights_sha256": sha(preflight / "weights_verification.json")}


def register(out, cfg, inputs, extra_sources=()):
    """Fresh CPU-only source/input closure; caller decides when preparation occurs."""
    out = Path(out).resolve()
    if cfg.get("kind") in ("training","evaluation","retention"):
        cap = cfg.get("max_seconds")
        if cfg.get("gpu") not in ("0","1") or (cap is not None and
            (type(cap) not in (int,float) or not math.isfinite(cap) or cap <= 0)):
            raise ValueError("one registered GPU and optional finite positive cap required")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    for name, value in inputs.items():
        atomic_json(out / name, value, new=True)
    for name in source_closure(CORE + list(extra_sources)):
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    (out / "source/scripts/__init__.py").touch()
    (out / "source/evaluation").mkdir(exist_ok=True)
    (out / "source/evaluation/__init__.py").touch()
    cfg = {**cfg, "versions": versions(), "outer_bf16_autocast":OUTER_BF16_AUTOCAST,
           "source_files": inventory(out / "source"),
           "inputs": {n: sha(out / n) for n in inputs}, "python": sys.executable}
    atomic_json(out / "registration.json", cfg, new=True)
    return {"run": str(out), "registration_sha256": sha(out / "registration.json"), "gpu_launched": False}


def source_closure(names):
    pending = list(names); found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules = []
            if isinstance(node,ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                modules = [node.module or ""]
                modules += [(node.module+"."+a.name) for a in node.names if node.module in ("scripts","evaluation")]
            for module in modules:
                candidate = module.replace(".","/")+".py"
                if module.startswith(("scripts.","evaluation.")) and (ROOT/candidate).is_file():
                    pending.append(candidate)
    return sorted(found)


def geometry(base):
    """Rebuild exact meta inventory rather than trusting a numeric count."""
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    config = Gemma4TextConfig.from_dict(read(Path(base)/"config.json")["text_config"])
    if config.model_type != "gemma4_text" or config.num_hidden_layers != 60:
        raise ValueError("Gemma dense text geometry differs")
    with torch.device("meta"):
        model = Gemma4ForCausalLM(config)
    targets = [{"name":n,"in_features":m.in_features,"out_features":m.out_features}
        for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.rsplit(".",1)[-1] in PROJECTIONS]
    old = read(PREFLIGHT/"cpu_prepare.json")
    if targets != old["targets"] or len(targets) != 410 or sum(16*(r["in_features"]+r["out_features"]) for r in targets) != 122429440:
        raise ValueError("preflight target inventory differs")
    return {"targets":targets,"config":config.to_dict()}


def validate(run, expected):
    run = Path(run).resolve()
    if sha(run / "registration.json") != expected:
        raise ValueError("prospective registration SHA mismatch")
    cfg = read(run / "registration.json")
    verify_files(run / "source", cfg["source_files"])
    for name, digest in cfg["inputs"].items():
        if sha(run / name) != digest:
            raise ValueError("registered input changed")
    if versions() != cfg["versions"] or cfg["outer_bf16_autocast"] != OUTER_BF16_AUTOCAST:
        raise ValueError("runtime package versions differ")
    return cfg


def frozen_exec(module, run, expected, resume=False):
    cfg = validate(run, expected)
    source = Path(run).resolve() / "source"
    if ROOT != source:
        os.chdir(source)
        argv = [sys.executable, "-B", "-m", module, "execute", "--run", str(Path(run).resolve()),
                "--registration-sha256", expected] + (["--resume"] if resume else [])
        os.execve(sys.executable, argv, dict(os.environ, PYTHONPATH=str(source),
            CUDA_VISIBLE_DEVICES=cfg["gpu"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false"))
    return cfg


def require_idle(gpu):
    """Admit empty or small inactive contexts; never alter another process.

The fixed pre-data policy is total used memory<=1024MiB and utilization<=5%.
Receipts retain all selected-device process entries and explicitly include
coexistence in timing scope; this is not an exclusive-device timing protocol.
"""
    global OCCUPANCY_SNAPSHOT
    OCCUPANCY_SNAPSHOT = None
    if str(gpu) not in ("0", "1") or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("explicit registered physical GPU visibility required")
    output = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True)
    records = list(csv.reader(output.strip().split("\n"),skipinitialspace=True))
    if len(records) != 1 or len(records[0]) != 5:
        raise ValueError("unusable NVML device snapshot")
    uuid,name,total,used,utilization = [v.strip() for v in records[0]]
    total,used,utilization = float(total),float(used),float(utilization)
    if (not uuid or not name or not all(math.isfinite(v) for v in (total,used,utilization)) or
        total <= 0 or not 0 <= used <= total or not 0 <= utilization <= 100):
        raise ValueError("invalid NVML device telemetry")
    output = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"],text=True)
    processes = []
    for record in csv.reader(output.strip().split("\n"),skipinitialspace=True):
        if not record or not record[0].strip() or record[0].strip() != uuid:
            continue
        if len(record) != 4:
            raise ValueError("unusable NVML process snapshot")
        process_uuid,pid,process_name,memory = [v.strip() for v in record]
        pid,memory = int(pid),float(memory)
        if pid <= 0 or not process_name or not math.isfinite(memory) or memory < 0:
            raise ValueError("invalid NVML process telemetry")
        processes.append({"gpu_uuid":process_uuid,"pid":pid,"process_name":process_name,
                          "used_memory_mib":memory})
    accepted = used <= MAX_EXISTING_MEMORY_MIB and utilization <= MAX_EXISTING_UTILIZATION_PERCENT
    OCCUPANCY_SNAPSHOT = {"timestamp_unix":time.time(),"physical_gpu":str(gpu),
        "gpu_uuid":uuid,"gpu_name":name,"total_memory_mib":total,"used_memory_mib":used,
        "utilization_gpu_percent":utilization,"processes":processes,"accepted":accepted,
        "policy":{"maximum_used_memory_mib":MAX_EXISTING_MEMORY_MIB,
                  "maximum_utilization_percent":MAX_EXISTING_UTILIZATION_PERCENT},
        "timing_scope":"All timings include coexistence with recorded resident contexts; no exclusive-device timing claim"}
    if not accepted:
        raise RuntimeError(f"GPU headroom/activity policy rejected: used={used}MiB, utilization={utilization}%; no allocation change")
    return uuid


@contextmanager
def exclusive(run):
    with (Path(run) / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_runtime(identity, targets, adapter=None):
    occupancy = copy.deepcopy(OCCUPANCY_SNAPSHOT)
    if not occupancy or not occupancy["accepted"]:
        raise RuntimeError("accepted occupancy snapshot required before model loading")
    import torch
    from transformers import BitsAndBytesConfig, Gemma4ForCausalLM, Gemma4TextConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    verify_files(identity["base"], identity["files"])
    base = identity["base"]
    config = Gemma4TextConfig.from_dict(read(Path(base) / "config.json")["text_config"])
    model, info = Gemma4ForCausalLM.from_pretrained(base, config=config,
        key_mapping={r"^model\.language_model\.": "model."},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")) or any(
        not (n.startswith("model.vision_tower.") or n.startswith("model.embed_vision."))
        for n in info.get("unexpected_keys", [])):
        raise ValueError("Gemma text-only loading keys differ")
    from bitsandbytes.nn import Linear4bit
    actual = [{"name": n, "in_features": m.in_features, "out_features": m.out_features}
        for n, m in model.named_modules() if n.rsplit(".", 1)[-1] in PROJECTIONS]
    if actual != targets or len(actual) != 410 or any(not isinstance(model.get_submodule(r["name"]), Linear4bit) for r in actual):
        raise ValueError("410 NF4 projection names/classes/geometry differ")
    if any("vision" in n or "audio" in n for n, _ in model.named_parameters()):
        raise ValueError("unexpected multimodal parameters")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=[r["name"] for r in targets], bias="none"))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in n for n, p in trainable) or sum(p.numel() for _, p in trainable) != 122429440:
        raise ValueError("LoRA trainable inventory differs")
    if adapter:
        reload_adapter(model, adapter)
    return model, {"targets": 410, "trainable_parameters": 122429440, "dropout": 0.05,
                   "occupancy_snapshot":occupancy,
                   "autocast": True, "autocast_dtype":"bfloat16",
                   "preflight_difference":"preflight had no outer autocast; startup short CE and longest-row backward validate new profile",
                   "kbit_floating_casts": "PEFT default FP32"}


def reload_adapter(model, adapter):
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    config = read(Path(adapter) / "adapter_config.json")
    if any(config.get(k) != v for k, v in {"r":16, "lora_alpha":32, "lora_dropout":0.05,
                                         "bias":"none", "task_type":"CAUSAL_LM"}.items()):
        raise ValueError("adapter configuration mismatch")
    state = load_file(Path(adapter) / "adapter_model.safetensors")
    report = set_peft_model_state_dict(model, state)
    if report.unexpected_keys or any("lora_" in k for k in report.missing_keys):
        raise ValueError("adapter keys mismatch")
    actual = get_peft_model_state_dict(model)
    if set(actual) != set(state) or any(not torch.equal(actual[k].detach().cpu(), state[k]) for k in state):
        raise ValueError("fresh adapter reload equality failed")


def render(tokenizer, user):
    prompt = tokenizer.apply_chat_template([{"role":"user", "content":user}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    return prompt, tokenizer(prompt, add_special_tokens=False)["input_ids"]


def generate(model, tokenizer, ids, cap, deadline):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() >= deadline
    if not ids or len(ids) + cap > 8192 or time.monotonic() >= deadline:
        raise ValueError("empty/overlength input or exhausted deadline; truncation forbidden")
    batch = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    with amp(model):
        sequence = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
            do_sample=False, num_beams=1, use_cache=True, max_new_tokens=cap, temperature=None,
            top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=[1,106],
            pad_token_id=tokenizer.pad_token_id, logits_to_keep=1,
            stopping_criteria=StoppingCriteriaList([Deadline()]))
    tokens = sequence[0, len(ids):].tolist()
    eos = bool(tokens and tokens[-1] in (1,106))
    return {"generated_ids": tokens, "output": tokenizer.decode(tokens[:-1] if eos else tokens, skip_special_tokens=False),
            "ended_with_eos": eos, "hit_token_limit": len(tokens) == cap and not eos,
            "resource_cap_stop": time.monotonic() >= deadline}

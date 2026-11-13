"""Download pinned small benchmark inputs and official IFEval scorer assets."""
from scripts.common import digest, path, read_json, write_json


def main():
    import os
    os.environ["HF_HOME"] = str(path(".hf").resolve())
    import requests
    from huggingface_hub import hf_hub_download
    cfg = read_json("configs/benchmark_sources.json")
    for split in ["dev", "validation", "test"]:
        hf_hub_download(cfg["mmlu"]["repository"], f"all/{split}-00000-of-00001.parquet",
                        repo_type="dataset", revision=cfg["mmlu"]["revision"])
    hf_hub_download(cfg["ifeval"]["repository"], "ifeval_input_data.jsonl", repo_type="dataset",
                    revision=cfg["ifeval"]["revision"])
    out = path("artifacts/tools/ifeval")
    out.mkdir(parents=True, exist_ok=True)
    rev = cfg["ifeval_scorer"]["revision"]
    hashes = {}
    for name in ["instructions.py", "instructions_registry.py", "instructions_util.py", "evaluation_lib.py"]:
        url = f"https://raw.githubusercontent.com/google-research/google-research/{rev}/instruction_following_eval/{name}"
        target = out / name
        if not target.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            target.write_text(response.text)
        hashes[name] = digest(target)
    (out / "__init__.py").touch(exist_ok=True)
    write_json(out / "source.json", {"repository": cfg["ifeval_scorer"]["repository"], "revision": rev})
    import nltk
    nltk_dir = path("artifacts/tools/nltk")
    nltk_dir.mkdir(parents=True, exist_ok=True)
    if not nltk.download(cfg["nltk_resource"], download_dir=str(nltk_dir), quiet=True):
        raise RuntimeError("NLTK resource download failed")
    write_json(out / "asset_hashes.json", hashes)
    print("Pinned benchmark sources ready")


if __name__ == "__main__":
    main()

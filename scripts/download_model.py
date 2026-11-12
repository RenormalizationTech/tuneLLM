"""Explicit network step: download only canonical safetensors and tokenizer/config files."""
import argparse
from pathlib import Path
from scripts.common import load_config, path, required_weights, setup_hf, validate_snapshot, write_json

CONFIG_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "tokenizer.model", "chat_template.jinja",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--manifest", default="outputs/download.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cache = setup_hf(cfg, offline=False)
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(cfg["model_id"], revision=cfg["revision"], files_metadata=True)
    if info.sha != cfg["revision"]:
        raise ValueError("Hub returned a different revision")
    files = {item.rfilename: item for item in info.siblings}
    selected = sorted(set(required_weights(files)) | (CONFIG_FILES & files.keys()))
    snapshot = Path(snapshot_download(cfg["model_id"], revision=cfg["revision"],
                                     cache_dir=str(cache), allow_patterns=selected))
    weight_bytes = validate_snapshot(snapshot)
    write_json(args.manifest, {"model_id": cfg["model_id"], "revision": cfg["revision"],
                              "snapshot_path": str(snapshot), "files": selected,
                              "weight_bytes": weight_bytes})
    print(snapshot)


if __name__ == "__main__":
    main()

"""Rebind unchanged CPU-only prepared inputs to finalized pre-study sources.

Original preparation directories remain immutable. This never loads weights,
tokenizes text, scores output or accepts a directory containing a GPU attempt.
"""
import argparse
from pathlib import Path

from scripts import gemma4_qasper_runtime as rt


def refresh(source,output,coverage=None):
    source=Path(source).resolve();output=Path(output).resolve()
    cfg=rt.validate(source,rt.sha(source/"registration.json"))
    if (source/"attempts").exists() or (source/"result.json").exists():
        raise ValueError("only CPU preparation without GPU attempts can be rebound")
    inputs={name:rt.read(source/name) for name in cfg["inputs"]}
    old={"path":str(source/"registration.json"),"sha256":rt.sha(source/"registration.json"),
         "reason":"Finalized mechanical validation before whole-study registration; encoded inputs unchanged"}
    for key in ("versions","source_files","inputs","python","outer_bf16_autocast"):cfg.pop(key,None)
    if cfg.get("kind")=="training":
        from scripts.gemma4_train_qasper import STARTUP_VALIDATION
        cfg["startup_validation"]=STARTUP_VALIDATION
    if cfg.get("kind")=="evaluation":
        if cfg.get("purpose")!="development" or not coverage:raise ValueError("only development preparation with explicit refreshed coverage is supported")
        old_coverage=source.parent/"coverage/registration.json"
        new_coverage=Path(coverage).resolve()/"registration.json"
        before,after=rt.read(old_coverage),rt.read(new_coverage)
        if (rt.sha(old_coverage)!=cfg["coverage_registration"]["sha256"] or
            before["summary"]!=after["summary"] or before["identity"]!=after["identity"] or
            before["inputs"]!=after["inputs"]):raise ValueError("refreshed coverage differs")
        cfg["coverage_registration"]={"path":str(new_coverage),"sha256":rt.sha(new_coverage)}
    cfg["prior_cpu_preparation"]=old
    extra=[name for name in rt.read(source/"registration.json")["source_files"] if not name.endswith("__init__.py")]
    result=rt.register(output,cfg,inputs,extra)
    original=rt.read(source/"registration.json")
    for name,digest in original["inputs"].items():
        if rt.sha(output/name)!=digest:raise ValueError("encoded input bytes changed during source refresh")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--coverage",help="Refreshed coverage folder; required for development evaluation")
    args=parser.parse_args();print(__import__("json").dumps(refresh(args.source,args.output,args.coverage),indent=2))


if __name__=="__main__":main()

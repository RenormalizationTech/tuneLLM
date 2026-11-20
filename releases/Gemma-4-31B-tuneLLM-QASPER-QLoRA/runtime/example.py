"""Run the adapter locally with an explicit, externally pinned release manifest."""
import argparse
import json
from pathlib import Path
from .loader import format_question,load_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",required=True); parser.add_argument("--base",required=True)
    parser.add_argument("--manifest-sha256",required=True)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt-file",help="Verbatim complete user prompt; must match the studied numbered-paragraph format")
    inputs.add_argument("--question-json",help="JSON object with paragraphs [{id,section,text},...] and question; exact frozen formatter")
    parser.add_argument("--cap",type=int,default=512); parser.add_argument("--baseline",action="store_true")
    args = parser.parse_args()
    if args.question_json:
        value=json.loads(Path(args.question_json).read_text())
        user=format_question(value["paragraphs"],value["question"])
    else:
        user=Path(args.prompt_file).read_text()
    runtime = load_runtime(args.bundle,args.base,args.manifest_sha256)
    prompt,ids = runtime.render(user)
    print(json.dumps(runtime.generate_ids(ids,adapter=not args.baseline,cap=args.cap),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

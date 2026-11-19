"""EXPERIMENTAL style demonstration. No tested strength passed all study gates."""
import argparse
import sys
from .loader import load_runtime,validate_strength


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True,help='Existing local pinned NF4 snapshot; no automatic download')
    p.add_argument('--manifest-sha256',required=True,help='External trusted SHA256 of release_manifest.json')
    p.add_argument('--strength',type=float,default=8,help='0 off;8 EXPERIMENTAL owner-chosen demonstration;16 extreme. Other intermediate values untested.')
    p.add_argument('--prompt',required=True)
    p.add_argument('--max-new-tokens',type=int,default=160)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    try:
        validate_strength(args.strength)
        if not args.prompt.strip() or not 1<=args.max_new_tokens<=160:raise ValueError('nonempty prompt and token cap1..160 required')
        print('EXPERIMENTAL research demonstration: no tested dose passed all acceptance gates. b8 failed relative coherence; b16 is extreme.',file=sys.stderr)
        runtime=load_runtime(args.snapshot,args.manifest_sha256,device=args.device)
        print(runtime.generate(args.prompt,args.strength,args.max_new_tokens)['text'])
    except (ValueError,RuntimeError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()

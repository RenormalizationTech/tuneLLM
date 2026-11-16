"""Operational extraction wrapper: frozen arithmetic, global inference context."""
import argparse
import runpy
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trait',required=True,choices=('profanity','impolite','sarcasm'))
    parser.add_argument('--execute-gpu',action='store_true')
    args=parser.parse_args()
    import torch
    original=sys.argv
    sys.argv=['scripts.qwen3_4b_fixed_transfer','extraction','--trait',args.trait]+(['--execute-gpu'] if args.execute_gpu else [])
    try:
        with torch.no_grad():
            runpy.run_module('scripts.qwen3_4b_fixed_transfer',run_name='__main__')
    finally:
        sys.argv=original


if __name__=='__main__':main()

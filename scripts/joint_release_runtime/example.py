"""Run from the downloaded bundle root: python -m joint_controls.example."""
import argparse
from pathlib import Path
from .loader import verify_bundle,load_controller,read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--preset',required=True)
    p.add_argument('--joint-manifest-sha256',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--prompt',default='Describe a rainy afternoon in two sentences.')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    verify_bundle(root,expected_joint_manifest_sha256=a.joint_manifest_sha256)
    if a.preset not in read(root/'joint_controls/presets.json'):raise ValueError('unmeasured preset')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from runtime.loader import BASE,REV
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,trust_remote_code=False)
    tok.pad_token=tok.eos_token;tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':a.device}).eval()
    ids=tok.apply_chat_template([{'role':'user','content':a.prompt}],add_generation_prompt=True,return_tensors='pt').to(model.device)
    # Fresh context per generation; baseline installs no hooks. No auto-download.
    ctx=load_controller(root,a.preset,model,tok,a.snapshot,base_id=BASE,revision=REV,expected_joint_manifest_sha256=a.joint_manifest_sha256)
    torch.manual_seed(42)
    with torch.no_grad(),ctx:
        out=model.generate(ids,do_sample=False,temperature=1.,top_p=1.,top_k=0,
            repetition_penalty=1.05,use_cache=True,max_new_tokens=160)
    print(tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True))

if __name__=='__main__':main()

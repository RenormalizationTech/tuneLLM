"""Read-only minute-level job watchdog; flags stalls without quality decisions."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"outputs/gemma4-qasper-study-01"


def read(path):return json.loads(Path(path).read_text())


def main():
    while True:
        status=read(STUDY/"pipeline-status.json") if (STUDY/"pipeline-status.json").exists() else {"status":"not_started"}
        now=time.time();jobs=[]
        parent_alive=None
        if status.get("pid"):
            try:os.kill(status["pid"],0);parent_alive=True
            except ProcessLookupError:parent_alive=False
        if status["status"]=="running" and parent_alive is False:
            destination=STUDY/"watchdog-flags";destination.mkdir(exist_ok=True)
            (destination/"pipeline-owner-missing.json").write_text(json.dumps({"timestamp_unix":now,"pid":status["pid"]})+"\n")
        try:
            lines=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"],text=True).splitlines()
            gpus={int(x.split(",")[0]):{"utilization":int(x.split(",")[1]),"memory_MiB":int(x.split(",")[2])} for x in lines}
        except Exception:gpus={}
        try:
            processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits"],text=True).splitlines()
        except Exception:processes=[]
        for p in STUDY.glob("*-pipeline.json"):
            state=read(p)
            if state.get("status")!="running":continue
            role=p.name.removesuffix("-pipeline.json")
            run=STUDY/role.replace("_","-")
            if role=="prompt_development":run=STUDY/"prompt-development"
            cfg=read(run/"registration.json")
            heartbeat=read(run/"heartbeat.json") if (run/"heartbeat.json").exists() else {}
            stamp=max(heartbeat.get("timestamp_unix",0),state["start_unix"])
            idle=now-stamp;threshold=1800 if role=="training" else 900
            util=gpus.get(int(cfg["gpu"]),{}).get("utilization")
            suspect=idle>threshold and util is not None and util<5
            jobs.append({"role":role,"seconds_since_complete_unit":idle,"heartbeat":heartbeat,
                "gpu":cfg["gpu"],"suspected_stall":suspect})
            if suspect:
                # Preserve diagnostics for the PM. Do not kill a job on a single
                # utilization sample or silently repeat a model generation.
                destination=STUDY/"watchdog-flags";destination.mkdir(exist_ok=True)
                evidence={"timestamp_unix":now,"role":role,"idle_seconds":idle,"GPU":gpus,
                    "log_tail":(run/"console.log").read_text(errors="replace")[-4000:] if (run/"console.log").exists() else "log not created yet"}
                (destination/(role+".json")).write_text(json.dumps(evidence,indent=2)+"\n")
        snapshot={"timestamp_unix":now,"pipeline_status":status["status"],"GPUs":gpus,
            "pipeline_process_alive":parent_alive,"compute_processes":processes,"jobs":jobs}
        temp=STUDY/"watchdog.json.tmp";temp.write_text(json.dumps(snapshot,indent=2)+"\n");temp.replace(STUDY/"watchdog.json")
        print(json.dumps(snapshot),flush=True)
        if status["status"] in ("completed","implementation_or_infrastructure_stop"):break
        time.sleep(60)


if __name__=="__main__":main()

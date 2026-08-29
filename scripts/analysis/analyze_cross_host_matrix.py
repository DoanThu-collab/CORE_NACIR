#!/usr/bin/env python3
"""Summarize 2-host x 2-retrieval-space NACIR matrix and paired bootstrap."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from nacir.metrics import compute_metrics


def load(path):
    r=np.load(path,allow_pickle=False)["ranks"].astype(np.int64)
    if r.shape!=(11,2064): raise ValueError(f"{path}: {r.shape}")
    return r

def summary(r):
    m=compute_metrics(r)
    per=np.asarray([float(x) for x in m["per_round_recall"]])
    cum=np.asarray([float(x) for x in m["cumulative_hits"]])
    return dict(avg=float(per[1:].mean()), final=float(per[-1]),
                cumulative=float(cum[-1]), bri=float(m["bri"]))

def boot(a,b,seed=20260829,n=20000):
    # final R@10, paired dialogue bootstrap
    d=(b[-1] < 10).astype(float)-(a[-1] < 10).astype(float)
    point=100*d.mean()
    rng=np.random.default_rng(seed)
    vals=np.empty(n)
    N=len(d)
    for st in range(0,n,512):
        en=min(n,st+512)
        idx=rng.integers(0,N,size=(en-st,N))
        vals[st:en]=100*d[idx].mean(1)
    lo,hi=np.quantile(vals,[.025,.975])
    return point,float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--chatir-blip-h0",default="runs_final/chatir_blip_h0/ranks.npz")
    ap.add_argument("--chatir-blip-current",default="runs_final/chatir_blip_nacir_current_turn/ranks.npz")
    ap.add_argument("--chatir-blip-persistent",default="runs_final/chatir_blip_nacir_minus/ranks.npz")
    ap.add_argument("--chatir-clip-h0",default="runs_final/chatir_clip_vitl14_h0/ranks.npz")
    ap.add_argument("--chatir-clip-current",default="runs_final/chatir_clip_vitl14_nacir_current_turn/ranks.npz")
    ap.add_argument("--chatir-clip-persistent",default="runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz")
    ap.add_argument("--plugir-blip-h0",default="runs_deadline/plugir_cr_blip_h0/ranks.npz")
    ap.add_argument("--plugir-blip-current",default="runs_deadline/plugir_cr_blip_current/ranks.npz")
    ap.add_argument("--plugir-blip-persistent",default="runs_deadline/plugir_cr_blip_persistent/ranks.npz")
    ap.add_argument("--plugir-clip-h0",default="runs_deadline/plugir_cr_clip_h0/ranks.npz")
    ap.add_argument("--plugir-clip-current",default="runs_deadline/plugir_cr_clip_current/ranks.npz")
    ap.add_argument("--plugir-clip-persistent",default="runs_deadline/plugir_cr_clip_persistent/ranks.npz")
    ap.add_argument("--out",default="artifacts_final/analysis/cross_host_matrix.json")
    args=ap.parse_args()

    groups = {
      "ChatIR×BLIP":(args.chatir_blip_h0,args.chatir_blip_current,args.chatir_blip_persistent),
      "ChatIR×CLIP":(args.chatir_clip_h0,args.chatir_clip_current,args.chatir_clip_persistent),
      "PlugIR×BLIP":(args.plugir_blip_h0,args.plugir_blip_current,args.plugir_blip_persistent),
      "PlugIR×CLIP":(args.plugir_clip_h0,args.plugir_clip_current,args.plugir_clip_persistent),
    }
    out={}
    print(f"{'Setting':16s} {'H0':>8s} {'Current':>8s} {'Persistent':>10s} {'P-C':>8s} {'95% CI':>20s}")
    print("-"*80)
    for name,(hp,cp,pp) in groups.items():
        if not all(Path(x).exists() for x in (hp,cp,pp)):
            print(f"{name:16s} MISSING")
            continue
        h,c,p=map(load,(hp,cp,pp))
        sh,sc,sp=map(summary,(h,c,p))
        delta,lo,hi=boot(c,p)
        out[name]={"H0":sh,"Current":sc,"Persistent":sp,
                   "persistent_minus_current_final_pp":delta,
                   "final_bootstrap_ci":[lo,hi]}
        print(f"{name:16s} {sh['final']:8.3f} {sc['final']:8.3f} {sp['final']:10.3f} "
              f"{delta:+8.3f} [{lo:+6.3f},{hi:+6.3f}]")
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("Saved:",args.out)

if __name__=="__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nacir.metrics import compute_metrics

def canon(text: str) -> str:
    return " ".join(str(text).lower().strip().split())

def load_vectors(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    x = obj.get("vectors") if isinstance(obj, dict) else obj
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{path}: expected [N,D] tensor")
    return x.float()

def load_sessions(path: Path) -> list[dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, list) or len(obj) != 2064:
        raise ValueError("expected 2064 standardized sessions")
    for i,s in enumerate(obj):
        if not isinstance(s.get("query_texts"), list) or len(s["query_texts"]) != 11:
            raise ValueError(f"session {i}: need 11 query_texts")
        if not isinstance(s.get("query_vectors"), torch.Tensor) or s["query_vectors"].shape[0] != 11:
            raise ValueError(f"session {i}: need [11,D] query_vectors")
    return obj

def load_beliefs(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    dialogs = doc.get("dialogs")
    if not isinstance(dialogs, list) or len(dialogs) != 2064:
        raise ValueError("expected 2064 belief dialogs")
    return dialogs

def build_texts(sessions, dialogs, max_concepts):
    base_texts, persistent_texts = [], []
    for session, dialog in zip(sessions, dialogs):
        memory: OrderedDict[str, tuple[str,int]] = OrderedDict()
        for rt in range(11):
            if rt > 0:
                for neg in dialog["turns"][rt-1].get("negatives",[]) or []:
                    raw = str(neg["attribute"]).strip()
                    memory[canon(raw)] = (raw, rt)
                if len(memory) > max_concepts:
                    ordered = sorted(memory, key=lambda k: memory[k][1])
                    for k in ordered[:len(memory)-max_concepts]:
                        del memory[k]
            base = str(session["query_texts"][rt]).strip()
            base_texts.append(base)
            if memory:
                excluded = "; ".join(raw for raw,_ in memory.values())
                persistent_texts.append(
                    base
                    + " The target image should not match the following excluded visual evidence: "
                    + excluded + "."
                )
            else:
                persistent_texts.append(base)
    return base_texts, persistent_texts

@torch.inference_mode()
def encode_all(encoder, texts, dim, batch_size, desc):
    out=[]
    for st in tqdm(range(0,len(texts),batch_size),desc=desc):
        batch=texts[st:st+batch_size]
        v=encoder.encode(batch)
        if not isinstance(v,torch.Tensor) or v.ndim!=2 or v.shape[1]!=dim:
            raise ValueError(f"encoder output mismatch: {getattr(v,'shape',None)} vs D={dim}")
        out.append(F.normalize(v.float().cpu(),dim=-1))
    return torch.cat(out,0)

@torch.inference_mode()
def ranks_flat(q, corpus, targets, device, batch_size):
    corpus=F.normalize(corpus.float().to(device),dim=-1)
    idx=torch.arange(corpus.shape[0],device=device)[None,:]
    out=torch.empty(len(q),dtype=torch.long)
    for st in tqdm(range(0,len(q),batch_size),desc="Scoring"):
        en=min(len(q),st+batch_size)
        qb=F.normalize(q[st:en].to(device),dim=-1)
        tb=targets[st:en].to(device)
        s=qb@corpus.T
        ts=s.gather(1,tb[:,None])
        out[st:en]=((s>ts).sum(1)+((s==ts)&(idx<tb[:,None])).sum(1)).cpu()
    return out

def matrix(flat):
    return flat.numpy().reshape(2064,11).T.astype(np.int64)

def summary(r):
    m=compute_metrics(r)
    per=np.asarray([float(x) for x in m["per_round_recall"]])
    cum=np.asarray([float(x) for x in m["cumulative_hits"]])
    return {"avg_feedback_r10":float(per[1:].mean()),"final_r10":float(per[-1]),
            "cum10":float(cum[-1]),"bri":float(m["bri"])}

def boot(a,b,seed=20260829,n=20000):
    d=(b[-1]<10).astype(float)-(a[-1]<10).astype(float)
    pt=100*d.mean()
    rng=np.random.default_rng(seed); vals=np.empty(n); N=len(d)
    for st in range(0,n,512):
        en=min(n,st+512)
        ids=rng.integers(0,N,size=(en-st,N))
        vals[st:en]=100*d[ids].mean(1)
    lo,hi=np.quantile(vals,[.025,.975])
    return float(pt),float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus-vectors",type=Path,required=True)
    ap.add_argument("--sessions",type=Path,required=True)
    ap.add_argument("--beliefs",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path("configs/nacir_minus_frozen.json"))
    ap.add_argument("--adapter-module",required=True)
    ap.add_argument("--adapter-func",required=True)
    ap.add_argument("--frozen-h0-ranks",type=Path,required=True)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-download",action="store_true")
    ap.add_argument("--encode-batch-size",type=int,default=128)
    ap.add_argument("--score-batch-size",type=int,default=128)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()

    cfg=json.loads(args.config.read_text())
    max_concepts=int(cfg["memory"]["max_concepts"])
    corpus=load_vectors(args.corpus_vectors); sessions=load_sessions(args.sessions); dialogs=load_beliefs(args.beliefs)
    dim=int(corpus.shape[1])

    mod=importlib.import_module(args.adapter_module)
    enc=getattr(mod,args.adapter_func)(args.device,allow_download=args.allow_download)
    probe=enc.encode(["object"])
    if not isinstance(probe,torch.Tensor) or probe.shape!=(1,dim):
        raise ValueError(f"adapter/corpus mismatch {getattr(probe,'shape',None)} vs {(1,dim)}")

    base_texts,persistent_texts=build_texts(sessions,dialogs,max_concepts)
    base_q=encode_all(enc,base_texts,dim,args.encode_batch_size,"Encoding Text-Reencode-H0")
    pers_q=encode_all(enc,persistent_texts,dim,args.encode_batch_size,"Encoding Text-Persistent")
    targets=torch.tensor([int(s["target_index"]) for s in sessions for _ in range(11)],dtype=torch.long)

    base_r=matrix(ranks_flat(base_q,corpus,targets,args.device,args.score_batch_size))
    pers_r=matrix(ranks_flat(pers_q,corpus,targets,args.device,args.score_batch_size))
    frozen=np.load(args.frozen_h0_ranks,allow_pickle=False)["ranks"].astype(np.int64)

    sb,sp,sf=summary(base_r),summary(pers_r),summary(frozen)
    delta,lo,hi=boot(base_r,pers_r)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/"text_reencode_h0_ranks.npz",ranks=base_r)
    np.savez_compressed(args.output/"text_persistent_ranks.npz",ranks=pers_r)
    rep={
      "frozen_h0":sf,"text_reencode_h0":sb,"text_persistent":sp,
      "text_persistent_minus_text_reencode_h0_final_pp":delta,
      "final_bootstrap_ci":[lo,hi],
      "reencode_vs_frozen":{
        "exact":bool(np.array_equal(base_r,frozen)),
        "rank_mismatches":int((base_r!=frozen).sum()),
        "r10_decision_changes":int(((base_r<10)!=(frozen<10)).sum()),
        "final_r10_delta_pp":sb["final_r10"]-sf["final_r10"],
      }
    }
    (args.output/"report.json").write_text(json.dumps(rep,indent=2))
    print("="*96)
    print("FAIR TEXT RETENTION CONTROL")
    print("="*96)
    print("Frozen H0:       ",sf)
    print("Text-Reencode-H0:",sb)
    print("Text-Persistent: ",sp)
    print(f"Text-Persistent - Text-Reencode-H0 final R@10 = {delta:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("Reencode-vs-frozen:",rep["reencode_vs_frozen"])
    print("Saved:",args.output)

if __name__=="__main__":
    main()

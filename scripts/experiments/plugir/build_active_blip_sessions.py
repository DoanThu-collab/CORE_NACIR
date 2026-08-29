#!/usr/bin/env python3
"""Build BLIP sessions for the Active PlugIR 256 subset without history concatenation."""

import json
import os
import argparse
from pathlib import Path

# Ensure offline mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm

from nacir.adapters.backbone import build_blip_backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", default="artifacts_final/plugir_full/plugir_256_active_recon.json")
    parser.add_argument("--baseline-sessions", default="artifacts_final/sessions_chatir_blip.pt")
    parser.add_argument("--output", default="artifacts_final/plugir_full/sessions_plugir_active_256_blip.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading Recon Active Dialogue trajectories from {args.recon}...")
    with open(args.recon) as f:
        recon_data = json.load(f)
    
    # Active PlugIR recon strings are already fully formed per turn (r0 to r10). No concatenation!
    recon_map = {item['session_id']: item['dialog'] for item in recon_data}
    
    print(f"Loading baseline sessions from {args.baseline_sessions}...")
    baseline_sessions = torch.load(args.baseline_sessions, weights_only=False)
    
    target_sessions = [s for s in baseline_sessions if s['session_id'] in recon_map]
    print(f"Extracted {len(target_sessions)} matching sessions from baseline.")
    assert len(target_sessions) == 256, "Expected exactly 256 matching baseline sessions."
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading BLIP Backbone on {device}...")
    text_encoder, _, _ = build_blip_backbone(
        device=device,
        model_id="Salesforce/blip-itm-large-coco",
        local_files_only=True,
    )
    
    new_sessions = []
    
    for sess in tqdm(target_sessions, desc="Encoding active trajectories"):
        sid = sess['session_id']
        texts = recon_map[sid]
        assert len(texts) == 11, f"Expected 11 turns, got {len(texts)}"
        
        # encode texts
        with torch.no_grad():
            vecs = text_encoder.encode(texts).detach().cpu()
        
        # normalize
        vecs = F.normalize(vecs.float(), dim=-1)
        assert vecs.shape == (11, 256)
        
        new_sessions.append({
            'session_id': sid,
            'target_index': sess['target_index'],
            'query_vectors': vecs,
            'query_texts': texts
        })
        
    print(f"Saving new BLIP sessions to {args.output}")
    torch.save(new_sessions, args.output)
    
    prov_out = str(args.output).replace('.pt', '.provenance.json')
    prov = {
        'source_recon': args.recon,
        'baseline_sessions': args.baseline_sessions,
        'model_id': 'Salesforce/blip-itm-large-coco',
        'same_evidence': True,
        'no_concatenation': True,
        'sessions_count': len(new_sessions)
    }
    with open(prov_out, 'w') as f:
        json.dump(prov, f, indent=2)
        
    print(f"Provenance saved to {prov_out}")
    print("Done!")

if __name__ == '__main__':
    main()

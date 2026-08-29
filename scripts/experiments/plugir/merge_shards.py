import json
from pathlib import Path

def merge_shards():
    root = Path("artifacts_final/plugir_full")
    raw_files = sorted(root.glob("*_shard_*_raw.json"))
    recon_files = sorted(root.glob("*_shard_*_recon.json"))
    
    # Exclude smoke test from merger
    raw_files = [f for f in raw_files if "smoke" not in f.name]
    recon_files = [f for f in recon_files if "smoke" not in f.name]
    
    print(f"Found {len(raw_files)} raw shards and {len(recon_files)} recon shards.")
    
    if len(raw_files) != 8 or len(recon_files) != 8:
        print("Warning: Expected 8 shards, but found a different number!")
    
    merged_raw = []
    merged_recon = []
    
    for rf, reconf in zip(raw_files, recon_files):
        with open(rf, 'r') as f:
            merged_raw.extend(json.load(f))
        with open(reconf, 'r') as f:
            merged_recon.extend(json.load(f))
            
    # Sort by session_id to maintain alignment with canonical order
    merged_raw.sort(key=lambda x: x['session_id'])
    merged_recon.sort(key=lambda x: x['session_id'])
    
    out_raw = root / "plugir_256_active_raw.json"
    out_recon = root / "plugir_256_active_recon.json"
    
    with open(out_raw, 'w') as f:
        json.dump(merged_raw, f, indent=2)
    with open(out_recon, 'w') as f:
        json.dump(merged_recon, f, indent=2)
        
    print(f"Merged {len(merged_raw)} trajectories into {out_raw}")
    print(f"Merged {len(merged_recon)} trajectories into {out_recon}")

if __name__ == "__main__":
    merge_shards()

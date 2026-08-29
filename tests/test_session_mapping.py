import json
import torch

def map_subset():
    with open('artifacts_final/plugir_full/plugir_256_active_session_ids.json') as f:
        target_ids = set(json.load(f))
        
    print(f"Loaded {len(target_ids)} target session IDs")
    
    sessions = torch.load('artifacts_final/sessions_chatir_blip.pt', weights_only=False)
    print(f"Loaded {len(sessions)} baseline sessions")
    
    subset = [s for s in sessions if s['session_id'] in target_ids]
    print(f"Found {len(subset)} matching baseline sessions")
    
if __name__ == '__main__':
    map_subset()

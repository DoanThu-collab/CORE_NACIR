import json
import numpy as np

with open('artifacts_final/plugir_full/plugir_256_active_beliefs_llama31_8b.json') as f:
    d = json.load(f)['dialogs']

h0 = np.load('runs_final/plugir_active_256_blip_h0/ranks.npz')['ranks_per_round']
p = np.load('runs_final/plugir_active_256_blip_nacir_persistent_canonical/ranks.npz')['ranks_per_round']
c = np.load('runs_final/plugir_active_256_blip_nacir_current_turn_canonical/ranks.npz')['ranks_per_round']

mismatches_p = 0
mismatches_c = 0

for i in range(256):
    # 1. Invariant cho Persistent (Toàn bộ session nếu có 0 negatives tổng)
    total_negs = sum(len(t.get('negatives', [])) for t in d[i]['turns'])
    if total_negs == 0:
        if not np.array_equal(h0[:, i], p[:, i]): 
            mismatches_p += 1

    # 2. Invariant mạnh cho Current-turn (Check riêng lẻ từng round)
    # Round 0 (Caption) luôn phải bằng nhau
    if h0[0, i] != c[0, i]: mismatches_c += 1
    
    # Round t+1 sử dụng negatives từ turn t 
    for turn in d[i]['turns']:
        t_idx = turn['turn']
        num_negs = len(turn.get('negatives', []))
        if num_negs == 0:
            if h0[t_idx + 1, i] != c[t_idx + 1, i]:
                mismatches_c += 1

print(f"--- INVARIANT TEST RESULTS ---")
print(f"Persistent Zero-Negative Mismatches : {mismatches_p}")
print(f"Current-turn Turn-level Mismatches  : {mismatches_c}")

if mismatches_p == 0 and mismatches_c == 0:
    print("\n✅ PASS TOÀN BỘ! Method đã chính xác là 'Canonical Negative-only Adapter'.")
else:
    print("\n🔴 FAIL! Vẫn còn bug Feature-drift cắm ngầm trong pipeline.")

import json
import numpy as np

with open('artifacts_final/plugir_full/plugir_256_active_beliefs_llama31_8b.json') as f:
    raw = json.load(f)
    d = raw['dialogs']

h0 = np.load('runs_final/plugir_active_256_blip_h0_canonical/ranks.npz')['ranks_per_round']
p = np.load('runs_final/plugir_active_256_blip_nacir_persistent_canonical/ranks.npz')['ranks_per_round']
c = np.load('runs_final/plugir_active_256_blip_nacir_current_turn_canonical/ranks.npz')['ranks_per_round']

with open('runs_final/stats_h0_vs_persistent_canonical.json') as f:
    stats_p = json.load(f)
with open('runs_final/stats_h0_vs_current_canonical.json') as f:
    stats_c = json.load(f)

print("--- CANONICAL RE-CALCULATED MACRO METRICS ---")
print(f"H0 BRI         : {stats_p['bri']['baseline']:.4f}, Final Recall@10: {stats_p['turns'][10]['baseline_recall']:.2f}%")
print(f"Current BRI    : {stats_c['bri']['candidate']:.4f}, Final Recall@10: {stats_c['turns'][10]['candidate_recall']:.2f}%")
print(f"Persistent BRI : {stats_p['bri']['candidate']:.4f}, Final Recall@10: {stats_p['turns'][10]['candidate_recall']:.2f}%")

print("\n--- CANONICAL AGE ANALYSIS ---")
harm_age, help_age = [], []
for i in range(256):
    is_harm = p[10, i] > 10 and h0[10, i] <= 10
    is_help = p[10, i] <= 10 and h0[10, i] > 10
    if not (is_harm or is_help): continue
    
    tgt = harm_age if is_harm else help_age
    act = [t["turn"] for t in d[i]["turns"] if t.get("negatives")]
    
    # Track age of applied negative beliefs at turn 10
    for tau in act:
        if tau < 10: 
            tgt.append(10 - tau)

if harm_age:
    print(f"- Harmful Applications (At Turn 10) Avg Age : {np.mean(harm_age):.2f} rounds (N={len(harm_age)})")
if help_age: 
    print(f"- Helpful Applications (At Turn 10) Avg Age : {np.mean(help_age):.2f} rounds (N={len(help_age)})")

print("\n--- NEW CANONICAL DISCORDANT CASES (H0 <= 10, P > 10) ---")
idx = np.where((h0[10]<=10) & (p[10]>10))[0]
print(f"Found {len(idx)} canonical discordant cases at Turn 10.")
with open("artifacts_final/plugir_full/plugir_256_active_raw.json") as f:
    text_raw = json.load(f)

for i in idx:
    print(f"\nSession {i}")
    print(f"Target Caption: {text_raw[i]['dialog'][0]}")
    print(f"H0 Rank@10: {h0[10, i]} | P Rank@10: {p[10, i]}")
    for t in d[i]["turns"]:
        if t.get("negatives"):
            print(f"  T{t['turn']} Negs: {[n['attribute'] for n in t['negatives']]}")


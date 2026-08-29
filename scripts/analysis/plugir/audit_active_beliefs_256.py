#!/usr/bin/env python3
import json
import statistics

def audit_beliefs():
    path = "artifacts_final/plugir_full/plugir_256_active_beliefs_llama31_8b.json"
    print(f"Loading beliefs from {path}...")
    with open(path) as f:
        data = json.load(f)

    negatives_per_session = []
    neg_per_turn = [0] * 10
    total_negatives = 0
    sessions_with_neg = 0

    for session in data['dialogs']:
        neg_count = 0
        for turn_idx, turn_data in enumerate(session['turns']):
            turn_negs = len(turn_data.get('negatives', []))
            neg_count += turn_negs
            if turn_idx < 10:
                neg_per_turn[turn_idx] += turn_negs

        negatives_per_session.append(neg_count)
        total_negatives += neg_count
        if neg_count > 0:
            sessions_with_neg += 1

    print("--- BELIEFS AUDIT ---")
    print(f"Total dialogues: {len(data['dialogs'])}")
    print(f"Total negative beliefs: {total_negatives}")
    print(f"Mean negatives/session: {total_negatives / len(data['dialogs']):.4f}")
    if len(data['dialogs']) > 0:
        print(f"Median negatives/session: {statistics.median(negatives_per_session)}")
        print(f"Max negatives/session: {max(negatives_per_session)}")
    print(f"Sessions with >= 1 negative: {sessions_with_neg} ({(sessions_with_neg/len(data['dialogs']))*100:.1f}%)")
    print(f"Negative beliefs per turn (t1 to t10): {neg_per_turn}")

    print("\nCompared to ChatIR baseline:")
    print("ChatIR mean negatives/session ~ 3.13")
    print("ChatIR total negatives (2064 sessions) ~ 6464")

if __name__ == '__main__':
    audit_beliefs()

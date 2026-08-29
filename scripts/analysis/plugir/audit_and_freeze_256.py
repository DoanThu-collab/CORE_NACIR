import json
import hashlib
from pathlib import Path

def get_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b""):
            sha.update(block)
    return sha.hexdigest()

def audit_and_freeze():
    root = Path('artifacts_final/plugir_full')

    raw_path = root / 'plugir_full_model_llama3.1_8b_q_n_5_rh_10_tl_500_recon_true_ref_true_filt_true_sel_true_subset_indices_raw.json'
    recon_path = root / 'plugir_full_model_llama3.1_8b_q_n_5_rh_10_tl_500_recon_true_ref_true_filt_true_sel_true_subset_indices_recon.json'

    print(f"Loading raw: {raw_path}")
    print(f"Loading recon: {recon_path}")

    with open(raw_path) as f:
        raw_data = json.load(f)
    with open(recon_path) as f:
        recon_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw sessions and {len(recon_data)} recon sessions.")

    assert len(raw_data) == 256, f"Expected 256 raw sessions, got {len(raw_data)}"
    assert len(recon_data) == 256, f"Expected 256 recon sessions, got {len(recon_data)}"

    unique_ids = set()
    total_turns = 0
    raw_ans_lengths = []
    recon_query_lengths = []

    for i, (r, rc) in enumerate(zip(raw_data, recon_data)):
        sid = r['session_id']
        assert sid == rc['session_id'], f"Session ID mismatch at index {i}"
        assert r['img'] == rc['img'], f"Image mismatch at SID {sid}"

        unique_ids.add(sid)

        r_dial = r['dialog']
        rc_dial = rc['dialog']

        assert len(r_dial) == 11, f"Raw dialogue doesn't have 11 states at SID {sid}. Has {len(r_dial)}"
        assert len(rc_dial) == 11, f"Recon dialogue doesn't have 11 states at SID {sid}. Has {len(rc_dial)}"
        assert r_dial[0] == rc_dial[0], f"Round 0 mismatch at SID {sid}"

        for turn_idx, turn_text in enumerate(r_dial):
            assert isinstance(turn_text, str) and turn_text.strip() != "", f"Empty row in raw SID {sid}, turn {turn_idx}"
            assert "error:" not in turn_text.lower(), f"Error found in raw SID {sid}: {turn_text}"
            if turn_idx > 0:
                parts = turn_text.split("?")
                assert len(parts) >= 2, f"Malformed raw Q/A (missing '?' separator) in SID {sid}: {turn_text}"
                ans = "?".join(parts[1:]).strip()
                raw_ans_lengths.append(len(ans.split()))

        for turn_idx, turn_text in enumerate(rc_dial):
            assert isinstance(turn_text, str) and turn_text.strip() != "", f"Empty row in recon SID {sid}, turn {turn_idx}"
            assert "error:" not in turn_text.lower(), f"Error found in recon SID {sid}: {turn_text}"
            if turn_idx > 0:
                recon_query_lengths.append(len(turn_text.split()))

    assert len(unique_ids) == 256, f"Expected 256 unique session IDs, got {len(unique_ids)}"

    print("\n--- AUDIT SUCCESSFUL ---")
    print(f"# dialogues = 256")
    print(f"# turns = 2560 feedback turns (10 per session)")
    print(f"avg raw answer length = {sum(raw_ans_lengths)/len(raw_ans_lengths):.2f} words")
    print(f"avg reconstructed query length = {sum(recon_query_lengths)/len(recon_query_lengths):.2f} words")
    print(f"# empty/malformed = 0")

    print("\nFreezing outputs...")

    out_raw = root / 'plugir_256_active_raw.json'
    out_recon = root / 'plugir_256_active_recon.json'
    out_ids = root / 'plugir_256_active_session_ids.json'
    out_manifest = root / 'plugir_256_active_merge_manifest.json'

    # Save the canonical files
    with open(out_raw, 'w') as f:
        json.dump(raw_data, f, indent=2)
    with open(out_recon, 'w') as f:
        json.dump(recon_data, f, indent=2)

    session_ids = sorted(list(unique_ids))
    with open(out_ids, 'w') as f:
        json.dump(session_ids, f, indent=2)

    manifest = {
        'plugir_256_active_raw.json': {
            'sha256': get_sha256(out_raw),
            'dialogues': 256
        },
        'plugir_256_active_recon.json': {
            'sha256': get_sha256(out_recon),
            'dialogues': 256
        },
        'plugir_256_active_session_ids.json': {
            'sha256': get_sha256(out_ids),
            'ids_count': 256
        }
    }

    with open(out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("Freeze complete. Manifest saved:")
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    audit_and_freeze()

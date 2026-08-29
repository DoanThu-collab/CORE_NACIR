from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thu/CORE_NACIR_24H"
)

META = ROOT / "artifacts_final/analysis/negation_density_per_dialog.csv"

EXPS = {
    "BLIP": {
        "h0": ROOT / "runs_final/chatir_blip_h0/ranks.npz",
        "current": ROOT / "runs_final/chatir_blip_nacir_current_turn/ranks.npz",
        "persistent": ROOT / "runs_final/chatir_blip_nacir_minus/ranks.npz",
    },
    "CLIP_ViT-L14": {
        "h0": ROOT / "runs_final/chatir_clip_vitl14_h0/ranks.npz",
        "current": ROOT / "runs_final/chatir_clip_vitl14_nacir_current_turn/ranks.npz",
        "persistent": ROOT / "runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz",
    },
}

BUCKET_ORDER = ["0-1", "2-3", "4-5", "6+"]

B = 10000
SEED = 42


def load_ranks(path):
    x = np.load(path)["ranks"]

    if x.shape[0] == 11:
        x = x.T

    assert x.shape == (2064, 11), (path, x.shape)

    return x


def ci(x):
    return np.percentile(
        np.asarray(x, dtype=float),
        [2.5, 50, 97.5],
    )


meta = pd.read_csv(META)

assert len(meta) == 2064

rng = np.random.default_rng(SEED)

rows = []

for backbone, paths in EXPS.items():

    h0 = load_ranks(paths["h0"])
    cur = load_ranks(paths["current"])
    per = load_ranks(paths["persistent"])

    h0_hit = h0 < 10
    cur_hit = cur < 10
    per_hit = per < 10

    for bucket in BUCKET_ORDER:

        idx = np.where(
            meta["density_bucket"].to_numpy() == bucket
        )[0]

        assert len(idx) > 0

        n = len(idx)

        h0_sub = h0_hit[idx]
        cur_sub = cur_hit[idx]
        per_sub = per_hit[idx]

        # Point estimates
        final_h0 = 100 * h0_sub[:, -1].mean()
        final_cur = 100 * cur_sub[:, -1].mean()
        final_per = 100 * per_sub[:, -1].mean()

        avg_h0 = 100 * h0_sub[:, 1:].mean()
        avg_cur = 100 * cur_sub[:, 1:].mean()
        avg_per = 100 * per_sub[:, 1:].mean()

        boot_h0_per_final = []
        boot_cur_per_final = []

        boot_h0_per_avg = []
        boot_cur_per_avg = []

        for _ in range(B):

            sample = rng.integers(
                0,
                n,
                n,
            )

            bh0 = h0_sub[sample]
            bcur = cur_sub[sample]
            bper = per_sub[sample]

            boot_h0_per_final.append(
                100 * (
                    bper[:, -1].mean()
                    -
                    bh0[:, -1].mean()
                )
            )

            boot_cur_per_final.append(
                100 * (
                    bper[:, -1].mean()
                    -
                    bcur[:, -1].mean()
                )
            )

            boot_h0_per_avg.append(
                100 * (
                    bper[:, 1:].mean()
                    -
                    bh0[:, 1:].mean()
                )
            )

            boot_cur_per_avg.append(
                100 * (
                    bper[:, 1:].mean()
                    -
                    bcur[:, 1:].mean()
                )
            )

        h0_final_ci = ci(boot_h0_per_final)
        cur_final_ci = ci(boot_cur_per_final)

        h0_avg_ci = ci(boot_h0_per_avg)
        cur_avg_ci = ci(boot_cur_per_avg)

        rows.append({
            "backbone": backbone,
            "bucket": bucket,
            "n": n,

            "h0_final_r10": final_h0,
            "current_final_r10": final_cur,
            "persistent_final_r10": final_per,

            "persistent_vs_h0_final_delta":
                final_per - final_h0,

            "persistent_vs_h0_final_ci_lo":
                h0_final_ci[0],

            "persistent_vs_h0_final_ci_med":
                h0_final_ci[1],

            "persistent_vs_h0_final_ci_hi":
                h0_final_ci[2],

            "persistent_vs_current_final_delta":
                final_per - final_cur,

            "persistent_vs_current_final_ci_lo":
                cur_final_ci[0],

            "persistent_vs_current_final_ci_med":
                cur_final_ci[1],

            "persistent_vs_current_final_ci_hi":
                cur_final_ci[2],

            "h0_avg_feedback_r10": avg_h0,
            "current_avg_feedback_r10": avg_cur,
            "persistent_avg_feedback_r10": avg_per,

            "persistent_vs_h0_avg_delta":
                avg_per - avg_h0,

            "persistent_vs_h0_avg_ci_lo":
                h0_avg_ci[0],

            "persistent_vs_h0_avg_ci_med":
                h0_avg_ci[1],

            "persistent_vs_h0_avg_ci_hi":
                h0_avg_ci[2],

            "persistent_vs_current_avg_delta":
                avg_per - avg_cur,

            "persistent_vs_current_avg_ci_lo":
                cur_avg_ci[0],

            "persistent_vs_current_avg_ci_med":
                cur_avg_ci[1],

            "persistent_vs_current_avg_ci_hi":
                cur_avg_ci[2],
        })


out = pd.DataFrame(rows)

OUT = ROOT / "artifacts_final/analysis/negative_density_bootstrap.csv"

out.to_csv(
    OUT,
    index=False,
)


print("=" * 100)
print("NEGATIVE DENSITY BOOTSTRAP")
print("=" * 100)

for _, row in out.iterrows():

    print(
        f'{row["backbone"]:15s} '
        f'{row["bucket"]:>4s} '
        f'n={int(row["n"]):4d} | '
        f'P-H0 final '
        f'{row["persistent_vs_h0_final_delta"]:+.2f} '
        f'['
        f'{row["persistent_vs_h0_final_ci_lo"]:+.2f}, '
        f'{row["persistent_vs_h0_final_ci_hi"]:+.2f}'
        f'] | '
        f'P-Cur final '
        f'{row["persistent_vs_current_final_delta"]:+.2f} '
        f'['
        f'{row["persistent_vs_current_final_ci_lo"]:+.2f}, '
        f'{row["persistent_vs_current_final_ci_hi"]:+.2f}'
        f']'
    )

print("\nSaved:", OUT)

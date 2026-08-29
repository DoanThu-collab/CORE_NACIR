#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np


SAMPLES = 20_000
SEED = 20260829
K = 10

RUNS = {
    "BLIP": {
        "persistent": Path(
            "runs_final/chatir_blip_nacir_minus/ranks.npz"
        ),
        "gate": Path(
            "runs_final/chatir_blip_persistent_gate/ranks.npz"
        ),
    },
    "CLIP": {
        "persistent": Path(
            "runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz"
        ),
        "gate": Path(
            "runs_final/chatir_clip_vitl14_persistent_gate/ranks.npz"
        ),
    },
}


def load_ranks(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    ranks = np.load(path)["ranks"]

    if ranks.ndim != 2:
        raise ValueError(
            f"{path}: expected rank matrix, got {ranks.shape}"
        )

    # Canonical layout should be [rounds=11, queries=2064].
    # Accept the transpose defensively.
    if ranks.shape == (2064, 11):
        ranks = ranks.T

    if ranks.shape != (11, 2064):
        raise ValueError(
            f"{path}: expected (11,2064), got {ranks.shape}"
        )

    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(f"{path}: ranks must be integers")

    if (ranks < 0).any():
        raise ValueError(f"{path}: negative ranks found")

    return ranks.astype(np.int64, copy=False)


def session_metrics(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return one contribution per dialogue/session so that all
    bootstrap resampling happens at the session level.
    """
    hit = (ranks < K).astype(np.float64)

    # Final-round R@10.
    final_recall = hit[-1] * 100.0

    # Mean R@10 across actual feedback states 1..10.
    mean_feedback_recall = (
        hit[1:].mean(axis=0) * 100.0
    )

    # Whether target has ever entered top-k by final state.
    final_cumulative = (
        np.maximum.accumulate(hit, axis=0)[-1] * 100.0
    )

    # Exact per-session contribution to canonical BRI implementation.
    best_ranks = np.minimum.accumulate(
        ranks,
        axis=0,
    ).astype(np.float64)

    trapezoids = (
        np.log(best_ranks[:-1] + 1.0)
        + np.log(best_ranks[1:] + 1.0)
    ) / 2.0

    bri = trapezoids.mean(axis=0)

    # Lower is better. Useful continuous final-state diagnostic.
    final_log_rank = np.log(
        ranks[-1].astype(np.float64) + 1.0
    )

    return {
        "Final R@10": final_recall,
        "Mean feedback R@10": mean_feedback_recall,
        "Final cumulative Hits@10": final_cumulative,
        "BRI": bri,
        "Final log-rank": final_log_rank,
    }


def bootstrap_delta(
    difference: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float, float]:
    n = difference.size

    observed = float(difference.mean())

    boot = np.empty(samples, dtype=np.float64)

    # Memory-safe batched paired bootstrap.
    batch = 256

    for start in range(0, samples, batch):
        stop = min(samples, start + batch)

        idx = rng.integers(
            0,
            n,
            size=(stop - start, n),
        )

        boot[start:stop] = (
            difference[idx].mean(axis=1)
        )

    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    # Two-sided bootstrap sign probability with finite-sample correction.
    le_zero = (
        np.count_nonzero(boot <= 0.0) + 1
    ) / (samples + 1)

    ge_zero = (
        np.count_nonzero(boot >= 0.0) + 1
    ) / (samples + 1)

    p = min(
        1.0,
        2.0 * min(le_zero, ge_zero),
    )

    return (
        observed,
        float(low),
        float(high),
        float(p),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(m, dtype=np.float64)

    running = 0.0

    for rank, idx in enumerate(order):
        value = (
            (m - rank)
            * p_values[idx]
        )

        running = max(running, value)

        adjusted[idx] = min(
            1.0,
            running,
        )

    return adjusted.tolist()


def main():
    rows = []

    print("=" * 88)
    print("PAIRED BOOTSTRAP: PERSISTENT -> GATE")
    print("=" * 88)
    print("samples:", SAMPLES)
    print("seed   :", SEED)
    print("unit   : dialogue/session")
    print()

    for backbone_i, (
        backbone,
        paths,
    ) in enumerate(RUNS.items()):

        base = load_ranks(
            paths["persistent"]
        )
        gate = load_ranks(
            paths["gate"]
        )

        assert base.shape == gate.shape
        assert base.shape == (11, 2064)

        base_m = session_metrics(base)
        gate_m = session_metrics(gate)

        print("=" * 88)
        print(backbone)
        print("=" * 88)

        for metric_i, metric in enumerate(base_m):
            b = base_m[metric]
            g = gate_m[metric]

            # Candidate - baseline.
            diff = g - b

            rng = np.random.default_rng(
                SEED
                + backbone_i * 100
                + metric_i
            )

            delta, lo, hi, p = bootstrap_delta(
                diff,
                rng=rng,
                samples=SAMPLES,
            )

            direction = (
                "higher better"
                if metric in {
                    "Final R@10",
                    "Mean feedback R@10",
                    "Final cumulative Hits@10",
                }
                else "lower better"
            )

            rows.append(
                {
                    "backbone": backbone,
                    "metric": metric,
                    "baseline": float(b.mean()),
                    "gate": float(g.mean()),
                    "delta": delta,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p": p,
                    "direction": direction,
                }
            )

            print(f"\n{metric} ({direction})")
            print(
                f"  Persistent : {b.mean():.8f}"
            )
            print(
                f"  Gate       : {g.mean():.8f}"
            )
            print(
                f"  Delta      : {delta:+.8f}"
            )
            print(
                f"  95% CI     : "
                f"[{lo:+.8f}, {hi:+.8f}]"
            )
            print(
                f"  bootstrap p: {p:.6f}"
            )

    adjusted = holm_adjust(
        [row["p"] for row in rows]
    )

    for row, p_holm in zip(
        rows,
        adjusted,
    ):
        row["p_holm"] = p_holm

    print("\n" + "=" * 88)
    print("HOLM-CORRECTED SUMMARY")
    print("=" * 88)

    for row in rows:
        print(
            f"{row['backbone']:4s} | "
            f"{row['metric']:25s} | "
            f"Δ={row['delta']:+.6f} | "
            f"CI=[{row['ci_low']:+.6f},"
            f"{row['ci_high']:+.6f}] | "
            f"p={row['p']:.5f} | "
            f"Holm={row['p_holm']:.5f}"
        )

    # Save machine-readable result.
    import json

    out = Path(
        "artifacts_final/typed_nacir/"
        "persistent_vs_gate_bootstrap_20k.json"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            {
                "samples": SAMPLES,
                "seed": SEED,
                "bootstrap_unit": "dialogue/session",
                "comparison": "Gate - Persistent",
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nsaved:", out)
    print("\nPASS")


if __name__ == "__main__":
    main()

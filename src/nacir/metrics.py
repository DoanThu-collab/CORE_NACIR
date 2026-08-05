"""
NACIR++ Plug-and-Play — Metrics (method-agnostic)
=====================================================
Tách nguyên khối tính Cumulative Hits@K / Per-round Recall@K / BRI ra khỏi
main.py gốc, KHÔNG đổi một công thức nào. Giờ đây hàm nhận vào
`ranks_per_round: List[List[int]]` (số vòng x số session) — không quan tâm
ranks đó tới từ NACIR++ gắn với PlugIR hay bất kỳ pipeline nào khác, miễn là
bạn cung cấp đúng khuôn dữ liệu.
"""

from typing import Dict, List

import torch


def compute_metrics(
    ranks_per_round: List[List[int]],
    k: int = 10,
) -> Dict[str, object]:
    """
    Args:
        ranks_per_round: ranks_per_round[t][i] = thứ hạng (0-indexed) của ảnh
                          đích cho session i tại turn t.
        k:                ngưỡng Hits@K / Recall@K.

    Returns dict:
        cumulative_hits:   Tensor [num_rounds] — % session đã được hit tính
                            luỹ kế tới round t.
        per_round_recall:  Tensor [num_rounds] — % session hit ĐÚNG tại round t.
        bri:                float — Best log Rank Integral.
    """
    num_rounds = len(ranks_per_round)
    num_queries = len(ranks_per_round[0])

    dialog_recalls_list = [torch.tensor(r, dtype=torch.long) for r in ranks_per_round]

    final_hits = torch.inf * torch.ones(num_queries)
    hitting_times, temp_hitting_times = [], []
    for ro_i in range(num_rounds):
        temp_hits = torch.inf * torch.ones(num_queries)
        rh = dialog_recalls_list[ro_i] < k
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        temp_hits[rh] = torch.min(temp_hits[rh], torch.ones(temp_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
        temp_hitting_times.append(temp_hits)

    ht_times, temp_ht_times = torch.stack(hitting_times), torch.stack(temp_hitting_times)
    cumulative_hits = (ht_times < torch.inf).sum(dim=-1).float() * 100 / num_queries
    per_round_recall = (temp_ht_times < torch.inf).sum(dim=-1).float() * 100 / num_queries

    min_ranks = [dialog_recalls_list[0].float()]
    for t in range(1, num_rounds):
        min_ranks.append(torch.minimum(min_ranks[t - 1], dialog_recalls_list[t].float()))

    bri = sum(
        ((torch.log(min_ranks[t] + 1.0) + torch.log(min_ranks[t + 1] + 1.0)) / 2).mean()
        for t in range(num_rounds - 1)
    ) / (num_rounds - 1)

    return {
        "cumulative_hits": cumulative_hits,
        "per_round_recall": per_round_recall,
        "bri": bri.item(),
    }


def format_metrics_report(metrics: Dict[str, object], k: int = 10) -> str:
    lines = [f"====== Results for Hits@{k} ======"]
    for t, v in enumerate(metrics["cumulative_hits"].tolist()):
        lines.append(f"\t Dialog Length: {t}: {round(v, 2)}%")
    lines.append(f"====== Results for Recall@{k} ======")
    for t, v in enumerate(metrics["per_round_recall"].tolist()):
        lines.append(f"\t Dialog Length: {t}: {round(v, 2)}%")
    lines.append("====== Best log Rank Integral ======")
    lines.append(f"\t BRI: {metrics['bri']:.4f}")
    return "\n".join(lines)

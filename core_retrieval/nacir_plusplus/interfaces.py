"""
NACIR++ Plug-and-Play — Interfaces (điểm cắm chuẩn hoá)
==========================================================
3 "chân cắm" (ports) mà một phương pháp bên ngoài cần cung cấp để tận dụng
NACIR++. Đây chính là phần biến NACIR++ từ "gắn cứng vào PlugIR" thành
"Plug-and-Play thực sự":

    1. TextEncoder    — encode text -> vector (bất kỳ backbone nào: BLIP, CLIP,
                         SigLIP, hay encoder nội bộ của method khác)
    2. BeliefSource    — sinh positive/negative beliefs cho 1 turn (rule-based,
                         LLM, precomputed JSON, hay bộ NLU riêng của method khác)
    3. ImageScorer     — (tuỳ chọn) cross-encoder re-ranking (ITM, VLM khác...)

Pipeline (pipeline.py) chỉ làm việc với 3 interface này — KHÔNG biết và
KHÔNG cần biết bên dưới là BLIP hay bất cứ mô hình nào khác.
"""

from typing import Any, List, Optional, Protocol, runtime_checkable

import torch

from .schema import BeliefBundle


@runtime_checkable
class TextEncoder(Protocol):
    """
    Bất kỳ text encoder nào (của bất kỳ phương pháp nào) implement interface
    này đều dùng được với NACIR++.
    """

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: danh sách chuỗi text cần encode.
        Returns:
            [len(texts), D] tensor đã L2-normalize.
        """
        ...


@runtime_checkable
class BeliefSource(Protocol):
    """
    Nguồn cung cấp beliefs (positive/negative concepts) cho một turn hội thoại.
    Có thể là:
      - Rule-based / regex negation detector (utils/negative_detector.py cũ)
      - Bộ trích xuất LLM/NLI riêng
      - Beliefs tiền tính sẵn (JSON) từ pipeline khác
      - Bộ hiểu hội thoại (NLU) của MỘT PHƯƠNG PHÁP KHÁC hoàn toàn — miễn là
        nó implement được hàm get_beliefs() dưới đây.
    """

    def get_beliefs(
        self,
        session_id: Any,
        turn_index: int,
        question: str,
        answer: str,
    ) -> BeliefBundle:
        ...


@runtime_checkable
class ImageScorer(Protocol):
    """
    Module re-rank cross-attention / cross-encoder (tuỳ chọn — Step cuối).
    Mặc định NACIR++ dùng BLIP ITM head, nhưng interface này cho phép cắm bất
    kỳ cross-encoder nào khác (VLM chấm điểm match, CLIP-based re-ranker...).
    """

    def score(self, query_text: str, image_refs: List[Any]) -> torch.Tensor:
        """
        Args:
            query_text: câu truy vấn gốc (không phải vector đã phẫu thuật).
            image_refs: danh sách tham chiếu ảnh (path, id, hoặc object bất kỳ
                        mà scorer tự biết cách load).
        Returns:
            [len(image_refs)] tensor điểm match (càng cao càng khớp).
        """
        ...


class NullImageScorer:
    """ImageScorer no-op — dùng khi phương pháp không có bước re-rank."""

    def score(self, query_text: str, image_refs: List[Any]) -> torch.Tensor:
        return torch.zeros(len(image_refs))

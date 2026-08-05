import copy
from typing import Any, List, Optional
import sys



from nacir.interfaces import BeliefSource
from nacir.schema import BeliefBundle, Belief


class CounterfactualBeliefSource(BeliefSource):
    """
    Adapter pattern (Đề xuất 2: Counterfactual Purging) bọc quanh một 
    BeliefSource gốc (ví dụ PrecomputedBeliefSource).
    
    Tự động viết lại chuỗi attribute từ dạng ngắn gọn (ví dụ "white cabinets")
    thành dạng hoàn chỉnh theo ngữ cảnh (ví dụ "a photo showing white cabinets")
    trước khi trả về cho pipeline.
    """

    def __init__(
        self, 
        base_source: BeliefSource, 
        template: str = "a photo showing {}",
        enabled: bool = True
    ):
        """
        Args:
            base_source: BeliefSource gốc (thường load từ JSON).
            template: Mẫu câu để điền attribute vào. 
                      Có thể dùng LLM API ở đây nếu muốn nâng cao, 
                      nhưng dùng template là đủ tốt để trị negation blindness.
            enabled: Cho phép bật/tắt counterfactual (để dễ ablation).
        """
        self.base_source = base_source
        self.template = template
        self.enabled = enabled

    def _apply_template(self, beliefs: List[Belief]) -> List[Belief]:
        if not self.enabled:
            return beliefs
            
        new_beliefs = []
        for b in beliefs:
            # Tạo bản sao để không ảnh hưởng dữ liệu gốc
            new_b = copy.deepcopy(b)
            # Chỉ bọc nếu nó chưa phải là một câu hoàn chỉnh (heuristic nhỏ)
            if "photo showing" not in new_b.attribute and "picture of" not in new_b.attribute:
                new_b.attribute = self.template.format(new_b.attribute.strip())
            new_beliefs.append(new_b)
        return new_beliefs

    def get_beliefs(self, session_id: Any, turn_index: int, question: str, answer: str) -> BeliefBundle:
        # Lấy belief gốc
        bundle = self.base_source.get_beliefs(session_id, turn_index, question, answer)
        
        # Tạo bundle mới với attribute đã được biến đổi
        return BeliefBundle(
            positive_beliefs=self._apply_template(bundle.positive_beliefs),
            negative_beliefs=self._apply_template(bundle.negative_beliefs)
        )

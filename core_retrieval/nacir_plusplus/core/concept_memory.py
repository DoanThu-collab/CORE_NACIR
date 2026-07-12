"""
NACIR++ — Step 2: Concept Memory Board (Bảng Nhớ Khái Niệm)
==============================================================
Tiến hóa từ M4 (query_update.py): Rocchio kéo-đẩy cộng dồn mờ nhạt.

NACIR++ xây hẳn Bảng Nhớ Khái Niệm (Concept Memory Board) để:
1. Lưu trữ TOÀN BỘ concepts đã gặp (positive + negative)
2. Auto-Override: Nếu concept bị phản bác → tự động ghi đè polarity
3. Tổng hợp query từ bảng nhớ thay vì Rocchio cộng dồn

So sánh:
    Cũ (Rocchio):  q_{t+1} = q_t + α·mean(E_pos) − β·mean(E_neg)
                   → mờ nhạt dần, quên nhanh, xung đột tích lũy
    
    Mới (Memory):  q_{t+1} = Normalize(q_t + α·Σ(w_c·v_c for c in Pos) 
                                            − β·Σ(w_c·v_c for c in Neg))
                   → tường minh, auto-override, không quên

Concept Memory Entry:
    {
        "name":         str,        # "black backpack"
        "polarity":     str,        # "positive" | "negative"
        "vector":       Tensor[D],  # encoded embedding
        "confidence":   float,      # 0.0-1.0
        "turn_created": int,        # first seen
        "turn_updated": int,        # last modified
        "override_count": int,      # number of polarity flips
    }
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class ConceptEntry:
    """Một entry trong Bảng Nhớ Khái Niệm."""
    name: str                       # Tên concept, VD "black backpack"
    polarity: str                   # "positive" hoặc "negative"
    vector: torch.Tensor            # Encoded embedding [D]
    confidence: float               # 0.0 - 1.0
    turn_created: int               # Turn đầu tiên xuất hiện
    turn_updated: int               # Turn cuối cùng bị cập nhật
    override_count: int = 0         # Số lần bị lật polarity


@dataclass
class ConceptMemoryConfig:
    """Configuration cho Concept Memory Board."""
    # Query synthesis weights
    alpha: float = 0.30             # Trọng số cho positive concepts
    beta: float = 0.15              # Trọng số cho negative concepts
    
    # Recency weighting
    recency_decay: float = 0.1      # Decay factor cho concepts cũ
    
    # Override confidence boost
    override_confidence_boost: float = 0.15  # Tăng confidence khi override
    
    # Similarity threshold for concept matching (auto-override)
    concept_match_threshold: float = 0.85    # cosine sim threshold
    
    # Maximum concepts in memory
    max_concepts: int = 50
    
    # Query normalization
    normalize_query: bool = True


class ConceptMemoryBoard:
    """
    NACIR++ Step 2 — Bảng Nhớ Khái Niệm

    Quản lý toàn bộ concepts đã gặp trong cuộc hội thoại.
    Tự động phát hiện xung đột và ghi đè (auto-override).

    Ví dụ workflow:
        Turn 1: User nói "yes, has a backpack"
                 → Memory: {"backpack": polarity=positive}
        Turn 3: User nói "no, not a black backpack"
                 → Memory: {"backpack": polarity=NEGATIVE}  (auto-overridden!)
                 → confidence TĂNG (vì user đã phản bác lại)
    """

    def __init__(
        self,
        config: Optional[ConceptMemoryConfig] = None,
        encoder: Optional[Any] = None,
    ):
        """
        Args:
            config:  ConceptMemoryConfig
            encoder: hàm encode text → embedding, 
                     signature: encoder(List[str]) → Tensor [batch, D]
        """
        self.config = config or ConceptMemoryConfig()
        self.encoder = encoder
        
        # Core memory: name → ConceptEntry
        self.memory: Dict[str, ConceptEntry] = {}
        self.current_turn: int = 0
        
        # Override history log (for analysis)
        self.override_log: List[Dict] = []

    def reset(self):
        """Reset bảng nhớ cho dialog mới."""
        self.memory.clear()
        self.current_turn = 0
        self.override_log.clear()

    def _encode_concept(self, name: str) -> torch.Tensor:
        """Encode concept name → L2-normalized embedding vector."""
        if self.encoder is None:
            raise RuntimeError("Encoder not set. Call set_encoder() first.")
        with torch.no_grad():
            vec = self.encoder([name])[0]  # [D]
            vec = F.normalize(vec, dim=-1)
        return vec

    def _find_matching_concept(self, name: str, vector: torch.Tensor) -> Optional[str]:
        """
        Tìm concept đã tồn tại trong memory mà giống với concept mới.
        So sánh bằng cả tên (exact match) và vector (cosine similarity).
        
        Returns:
            Tên concept matching, hoặc None nếu không tìm thấy.
        """
        # Exact name match
        name_lower = name.lower().strip()
        for existing_name in self.memory:
            if existing_name.lower().strip() == name_lower:
                return existing_name
        
        # Semantic similarity match
        for existing_name, entry in self.memory.items():
            sim = F.cosine_similarity(
                vector.unsqueeze(0), entry.vector.unsqueeze(0)
            ).item()
            if sim >= self.config.concept_match_threshold:
                return existing_name
        
        return None

    def add_concepts(
        self,
        positives: List[Dict],
        negatives: List[Dict],
        turn: int,
        precomputed_vectors: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, int]:
        """
        Thêm concepts từ Dual Extractor (Step 1) vào bảng nhớ.
        Tự động phát hiện xung đột và ghi đè.

        Args:
            positives: [{"attribute": str, "confidence": float}, ...]
            negatives: [{"attribute": str, "confidence": float}, ...]
            turn: turn index hiện tại
            precomputed_vectors: optional list of precomputed tensors

        Returns:
            {"added": int, "overridden": int, "updated": int}
        """
        self.current_turn = turn
        stats = {"added": 0, "overridden": 0, "updated": 0}

        if precomputed_vectors is not None:
            all_vectors = precomputed_vectors
        else:
            all_vectors = [None] * (len(positives) + len(negatives))
            
        n_pos = len(positives)

        # Process positive concepts
        for idx, item in enumerate(positives):
            result = self._process_concept(
                name=item["attribute"],
                polarity="positive",
                confidence=item.get("confidence", 0.7),
                turn=turn,
                precomputed_vector=all_vectors[idx],
            )
            stats[result] += 1

        # Process negative concepts
        for idx, item in enumerate(negatives):
            result = self._process_concept(
                name=item["attribute"],
                polarity="negative",
                confidence=item.get("confidence", 0.7),
                turn=turn,
                precomputed_vector=all_vectors[n_pos + idx],
            )
            stats[result] += 1

        # Evict old concepts if memory is full
        self._evict_if_full()

        return stats

    def _process_concept(
        self, name: str, polarity: str, confidence: float, turn: int,
        precomputed_vector: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Xử lý 1 concept: thêm mới, cập nhật, hoặc ghi đè.

        Returns:
            "added", "updated", hoặc "overridden"
        """
        vector = precomputed_vector if precomputed_vector is not None else self._encode_concept(name)
        existing_name = self._find_matching_concept(name, vector)

        if existing_name is None:
            # Concept mới → thêm vào memory
            self.memory[name] = ConceptEntry(
                name=name,
                polarity=polarity,
                vector=vector,
                confidence=confidence,
                turn_created=turn,
                turn_updated=turn,
                override_count=0,
            )
            return "added"

        existing = self.memory[existing_name]

        if existing.polarity == polarity:
            # Cùng polarity → cập nhật confidence và recency
            existing.confidence = max(existing.confidence, confidence)
            existing.turn_updated = turn
            return "updated"
        else:
            # XUNG ĐỘT! → Auto-Override
            old_polarity = existing.polarity
            existing.polarity = polarity
            existing.confidence = min(
                1.0,
                existing.confidence + self.config.override_confidence_boost
            )
            existing.turn_updated = turn
            existing.override_count += 1
            # Update vector to the new one (user's latest description)
            existing.vector = vector

            # Log the override
            self.override_log.append({
                "concept": existing_name,
                "old_polarity": old_polarity,
                "new_polarity": polarity,
                "turn": turn,
                "new_confidence": existing.confidence,
            })

            logger.debug(
                f"AUTO-OVERRIDE: '{existing_name}' flipped "
                f"{old_polarity} → {polarity} at turn {turn}"
            )
            return "overridden"

    def _evict_if_full(self):
        """Evict concepts cũ nhất nếu memory quá đầy."""
        if len(self.memory) <= self.config.max_concepts:
            return

        # Sort by recency (oldest first), then evict
        sorted_concepts = sorted(
            self.memory.items(),
            key=lambda x: x[1].turn_updated,
        )

        num_to_evict = len(self.memory) - self.config.max_concepts
        for name, _ in sorted_concepts[:num_to_evict]:
            del self.memory[name]
            logger.debug(f"Evicted concept '{name}' from memory")

    def get_positive_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lấy tất cả positive concept vectors và weights.

        Returns:
            (vectors [N, D], weights [N])
        """
        vectors = []
        weights = []
        for entry in self.memory.values():
            if entry.polarity == "positive":
                recency = 1.0 / (
                    1.0 + self.config.recency_decay
                    * (self.current_turn - entry.turn_updated)
                )
                w = entry.confidence * recency
                vectors.append(entry.vector)
                weights.append(w)

        if not vectors:
            return None, None

        return torch.stack(vectors), torch.tensor(
            weights, device=vectors[0].device
        )

    def get_negative_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lấy tất cả negative concept vectors và weights.

        Returns:
            (vectors [N, D], weights [N])
        """
        vectors = []
        weights = []
        for entry in self.memory.values():
            if entry.polarity == "negative":
                recency = 1.0 / (
                    1.0 + self.config.recency_decay
                    * (self.current_turn - entry.turn_updated)
                )
                w = entry.confidence * recency
                vectors.append(entry.vector)
                weights.append(w)

        if not vectors:
            return None, None

        return torch.stack(vectors), torch.tensor(
            weights, device=vectors[0].device
        )

    def synthesize_query(self, q_t: torch.Tensor) -> torch.Tensor:
        """
        Tổng hợp query vector từ bảng nhớ.

        q_{t+1} = Normalize(
            q_t
            + α · Σ(w_c · v_c) for c in Positive
            − β · Σ(w_c · v_c) for c in Negative
        )

        Args:
            q_t: query vector hiện tại [D]

        Returns:
            q_new: query vector mới [D]
        """
        q_new = q_t.clone()

        # Positive component
        pos_vecs, pos_weights = self.get_positive_vectors()
        if pos_vecs is not None:
            # Weighted sum of positive vectors
            weighted_pos = (pos_weights.unsqueeze(-1) * pos_vecs).sum(dim=0)
            weighted_pos = F.normalize(weighted_pos, dim=-1)
            q_new = q_new + self.config.alpha * weighted_pos

        # Negative component
        neg_vecs, neg_weights = self.get_negative_vectors()
        if neg_vecs is not None:
            # Weighted sum of negative vectors
            weighted_neg = (neg_weights.unsqueeze(-1) * neg_vecs).sum(dim=0)
            weighted_neg = F.normalize(weighted_neg, dim=-1)
            q_new = q_new - self.config.beta * weighted_neg

        # Normalize
        if self.config.normalize_query:
            q_new = F.normalize(q_new.unsqueeze(0), dim=-1).squeeze(0)

        return q_new

    def get_memory_snapshot(self) -> List[Dict]:
        """Trả về snapshot bảng nhớ hiện tại (for debugging/logging)."""
        snapshot = []
        for name, entry in self.memory.items():
            snapshot.append({
                "name": name,
                "polarity": entry.polarity,
                "confidence": entry.confidence,
                "turn_created": entry.turn_created,
                "turn_updated": entry.turn_updated,
                "override_count": entry.override_count,
            })
        return snapshot


class ConceptMemoryBatchUpdater:
    """
    Batch version of ConceptMemoryBoard.
    Maintains independent memory boards for B queries.
    """

    def __init__(
        self,
        config: ConceptMemoryConfig,
        batch_size: int,
        device: str,
        encoder: Optional[Any] = None,
    ):
        self.config = config
        self.B = batch_size
        self.device = device

        # One memory board per query in batch
        self.boards: List[ConceptMemoryBoard] = [
            ConceptMemoryBoard(config=copy.deepcopy(config), encoder=encoder)
            for _ in range(batch_size)
        ]
        self.current_turn = 0

    def set_encoder(self, encoder):
        """Set encoder cho tất cả boards."""
        for board in self.boards:
            board.encoder = encoder

    def add_concepts_batch(
        self,
        batch_positives: List[List[Dict]],
        batch_negatives: List[List[Dict]],
        turn: int,
    ):
        """
        Thêm concepts cho cả batch.

        Args:
            batch_positives[b]: list of positive attrs for query b
            batch_negatives[b]: list of negative attrs for query b
        """
        self.current_turn = turn
        for b in range(self.B):
            pos = batch_positives[b] if b < len(batch_positives) else []
            neg = batch_negatives[b] if b < len(batch_negatives) else []
            self.boards[b].add_concepts(pos, neg, turn)

    def synthesize_query_batch(self, q_t_batch: torch.Tensor) -> torch.Tensor:
        """
        Tổng hợp query vector cho cả batch.

        Args:
            q_t_batch: [B, D] current query vectors

        Returns:
            q_new_batch: [B, D] updated query vectors
        """
        q_new_list = []
        for b in range(self.B):
            q_new = self.boards[b].synthesize_query(q_t_batch[b])
            q_new_list.append(q_new)

        return torch.stack(q_new_list)

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class ConceptEntry:
    """One entry in the concept memory board."""
    name: str                       # Concept name, e.g. "black backpack"
    polarity: str                   # "positive" or "negative"
    vector: torch.Tensor            # Encoded embedding [D]
    confidence: float               # 0.0 - 1.0
    turn_created: int               # First turn where this concept appeared
    turn_updated: int               # Most recent update turn
    override_count: int = 0         # Number of polarity flips


@dataclass
class ConceptMemoryConfig:
    """Configuration for the concept memory board."""
    # Query synthesis weights
    alpha: float = 0.30             # Weight for positive concepts
    beta: float = 0.15              # Weight for negative concepts
    
    # Recency weighting
    recency_decay: float = 0.1      # Decay factor for older concepts
    
    # Override confidence boost
    override_confidence_boost: float = 0.15  # Confidence boost after override
    
    # Similarity threshold for concept matching (auto-override)
    concept_match_threshold: float = 0.85    # cosine sim threshold
    
    # Maximum concepts in memory
    max_concepts: int = 50
    
    # Query normalization
    normalize_query: bool = True


class ConceptMemoryBoard:
    
    def __init__(
        self,
        config: Optional[ConceptMemoryConfig] = None,
        encoder: Optional[Any] = None,
    ):
        """
        Args:
            config:  ConceptMemoryConfig
            encoder: text encoder with signature
                     encoder(List[str]) -> Tensor [batch, D]
        """
        self.config = config or ConceptMemoryConfig()
        self.encoder = encoder
        
        # Core memory: name -> ConceptEntry
        self.memory: Dict[str, ConceptEntry] = {}
        self.current_turn: int = 0
        
        # Override history log (for analysis)
        self.override_log: List[Dict] = []

    def reset(self):
        
        self.memory.clear()
        self.current_turn = 0
        self.override_log.clear()

    def _encode_concept(self, name: str) -> torch.Tensor:
        
        if self.encoder is None:
            raise RuntimeError("Encoder not set. Call set_encoder() first.")
        with torch.no_grad():
            vec = self.encoder([name])[0]  # [D]
            vec = F.normalize(vec, dim=-1)
        return vec

    def _find_matching_concept(self, name: str, vector: torch.Tensor) -> Optional[str]:
        
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
    ) -> Dict[str, int]:
        
        self.current_turn = turn
        stats = {"added": 0, "overridden": 0, "updated": 0}

        # Process positive concepts
        for item in positives:
            result = self._process_concept(
                name=item["attribute"],
                polarity="positive",
                confidence=item.get("confidence", 0.7),
                turn=turn,
            )
            stats[result] += 1

        # Process negative concepts
        for item in negatives:
            result = self._process_concept(
                name=item["attribute"],
                polarity="negative",
                confidence=item.get("confidence", 0.7),
                turn=turn,
            )
            stats[result] += 1

        # Evict old concepts if memory is full
        self._evict_if_full()

        return stats

    def _process_concept(
        self, name: str, polarity: str, confidence: float, turn: int
    ) -> str:
        
        vector = self._encode_concept(name)
        existing_name = self._find_matching_concept(name, vector)

        if existing_name is None:
            # Add a new concept to memory.
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
            # Same polarity: update confidence and recency.
            existing.confidence = max(existing.confidence, confidence)
            existing.turn_updated = turn
            return "updated"
        else:
            # Polarity conflict: auto-override.
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
                f"{old_polarity} -> {polarity} at turn {turn}"
            )
            return "overridden"

    def _evict_if_full(self):
        """Evict the oldest concepts if memory is full."""
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
        
        self.current_turn = turn
        for b in range(self.B):
            pos = batch_positives[b] if b < len(batch_positives) else []
            neg = batch_negatives[b] if b < len(batch_negatives) else []
            self.boards[b].add_concepts(pos, neg, turn)

    def synthesize_query_batch(self, q_t_batch: torch.Tensor) -> torch.Tensor:
        
        q_new_list = []
        for b in range(self.B):
            q_new = self.boards[b].synthesize_query(q_t_batch[b])
            q_new_list.append(q_new)

        return torch.stack(q_new_list)

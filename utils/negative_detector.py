import re
from typing import List, Dict, Tuple, Optional

# Stage 1: Rule-based patterns

# Explicit negation patterns (English - VisDial is in English)
NEGATIVE_PATTERNS = [
    # Direct negation
    r"\bno\b",
    r"\bnot\b",
    r"\bnope\b",
    r"\bnah\b",
    # Contractions
    r"\bdon'?t\b",
    r"\bdoesn'?t\b",
    r"\bdidn'?t\b",
    r"\bcan'?t\b",
    r"\bcannot\b",
    r"\bwon'?t\b",
    r"\bisn'?t\b",
    r"\baren'?t\b",
    r"\bwasn'?t\b",
    r"\bweren'?t\b",
    r"\bhasn'?t\b",
    r"\bhaven'?t\b",
    r"\bcouldn'?t\b",
    r"\bwouldn'?t\b",
    r"\bshouldn'?t\b",
    # Absence indicators
    r"\bwithout\b",
    r"\bnone\b",
    r"\bnever\b",
    r"\bneither\b",
    r"\bnobody\b",
    r"\bnothing\b",
    r"\bnowhere\b",
    # Phrases
    r"i don'?t think",
    r"i don'?t see",
    r"i don'?t believe",
    r"not really",
    r"not that i",
    r"can'?t see",
    r"can'?t tell",
    r"hard to tell",
    r"doesn'?t look like",
    r"doesn'?t appear",
    r"doesn'?t seem",
    r"not sure",
    r"i wouldn'?t say",
]

# Compile for speed
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]

# Mixed signal patterns (answer contains both positive and negative)
MIXED_PATTERNS = [
    r"\bbut\b",          # "no backpack, but has a jacket"
    r"\bhowever\b",
    r"\bthough\b",
    r"\balthough\b",
    r"\bexcept\b",
]
_COMPILED_MIXED = [re.compile(p, re.IGNORECASE) for p in MIXED_PATTERNS]


def is_negative_rule_based(answer: str) -> bool:
    """Stage 1: Check if answer contains explicit negation."""
    answer = answer.lower().strip()
    return any(p.search(answer) for p in _COMPILED_PATTERNS)


def is_mixed_rule_based(answer: str) -> bool:
    """Check if answer contains both positive and negative signals."""
    answer = answer.lower().strip()
    has_neg = any(p.search(answer) for p in _COMPILED_PATTERNS)
    has_mix = any(p.search(answer) for p in _COMPILED_MIXED)
    return has_neg and has_mix


def label_turn_stage1(answer: str) -> str:
    """
    Stage 1 label assignment.
    Returns: 'negative', 'mixed', or 'positive'
    """
    if is_mixed_rule_based(answer):
        return "mixed"
    elif is_negative_rule_based(answer):
        return "negative"
    else:
        return "positive"


def label_dialog(dialog_turns: List[Dict]) -> List[Dict]:
   
    labeled = []
    for i, turn in enumerate(dialog_turns):
        label = label_turn_stage1(turn["answer"])
        labeled.append({
            **turn,
            "label": label,
            "turn_idx": i,
        })
    return labeled


def count_negative_turns(dialog_turns: List[Dict]) -> int:
    """Count turns with negative or mixed labels."""
    return sum(
        1 for turn in dialog_turns
        if label_turn_stage1(turn.get("answer", "")) in ("negative", "mixed")
    )


def classify_dialog(dialog_turns: List[Dict], neg_threshold: int = 3) -> str:
    
    neg_count = count_negative_turns(dialog_turns)
    if neg_count >= neg_threshold:
        return "neg_heavy"
    elif neg_count <= 1:
        return "pos_only"
    else:
        return "moderate"


# VisDial-specific helpers

def parse_visdial_dialog(dialog_strings: List[str]) -> List[Dict]:
    
    if not dialog_strings:
        return []
    
    turns = []
    # Skip first element (caption)
    for qa_str in dialog_strings[1:]:
        # Split on first "? " to separate Q and A
        parts = qa_str.split("? ", 1)
        if len(parts) == 2:
            question = parts[0].strip() + "?"
            answer = parts[1].strip()
        else:
            # Fallback: treat whole string as answer
            question = ""
            answer = qa_str.strip()
        
        turns.append({
            "question": question,
            "answer": answer,
        })
    
    return turns


def analyze_visdial_dataset(queries: List[Dict], neg_threshold: int = 3) -> Dict:
    
    stats = {
        "total": len(queries),
        "neg_heavy_count": 0,
        "pos_only_count": 0,
        "moderate_count": 0,
        "neg_heavy_indices": [],
        "pos_only_indices": [],
        "moderate_indices": [],
        "neg_turn_distribution": {},  # neg_count -> number of dialogs
        "total_neg_turns": 0,
        "total_turns": 0,
    }
    
    for i, query in enumerate(queries):
        turns = parse_visdial_dialog(query["dialog"])
        neg_count = count_negative_turns(turns)
        category = classify_dialog(turns, neg_threshold)
        
        stats[f"{category}_count"] += 1
        stats[f"{category}_indices"].append(i)
        stats["neg_turn_distribution"][neg_count] = \
            stats["neg_turn_distribution"].get(neg_count, 0) + 1
        stats["total_neg_turns"] += neg_count
        stats["total_turns"] += len(turns)
    
    stats["avg_neg_per_dialog"] = stats["total_neg_turns"] / stats["total"] if stats["total"] > 0 else 0
    stats["neg_turn_ratio"] = stats["total_neg_turns"] / stats["total_turns"] if stats["total_turns"] > 0 else 0
    
    return stats

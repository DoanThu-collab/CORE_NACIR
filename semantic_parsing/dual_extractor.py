"""
NACIR++ — Step 1: Dual Signal Extractor
=========================================
Tiến hóa từ M1 (attribute_extractor.py): chỉ extract 1 negated phrase.

NACIR++ extract ĐỒNG THỜI cả rổ Positive VÀ Negative attributes từ mỗi turn.

Output format:
    {
        "positives": [
            {"attribute": "outdoor scene", "confidence": 0.95},
            {"attribute": "sunny weather", "confidence": 0.80}
        ],
        "negatives": [
            {"attribute": "black backpack", "confidence": 0.92},
            {"attribute": "hat", "confidence": 0.70}
        ]
    }

So sánh với NACIR cũ:
    - Cũ:  {"is_negative": true, "attribute": "black backpack"}  (1 keyword)
    - Mới: Cả 2 rổ, mỗi rổ nhiều attributes, kèm confidence

Backends: HuggingFace, Ollama, Rule-based (giống NACIR cũ)
"""

import json
import re
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================
# Prompt Template — LLM bóc tách cả 2 rổ
# ============================================================

SYSTEM_PROMPT_DUAL = """You are a visual dialog analyzer. Given a question and answer from a visual dialog, \
extract ALL visual attributes mentioned into two categories:

1. "positives": attributes that the image DOES HAVE (confirmed/affirmed by the answer)
2. "negatives": attributes that the image does NOT HAVE (denied/negated by the answer)

Each attribute should be a short noun phrase (1-4 words).
Each attribute should have a confidence score from 0.0 to 1.0.

Output ONLY valid JSON in this exact format:
{"positives": [{"attribute": "...", "confidence": 0.9}], "negatives": [{"attribute": "...", "confidence": 0.9}]}

Rules:
- If the answer is purely positive (e.g., "Yes, there is a red car"), put attributes in "positives"
- If the answer is purely negative (e.g., "No, there is no dog"), put attributes in "negatives"
- If mixed (e.g., "No backpack, but has a jacket"), put in BOTH lists
- Confidence 0.9+ for explicit mentions, 0.5-0.8 for implied/uncertain
- Empty list [] if no attributes found for that category
- No explanation. No markdown. ONLY JSON."""

USER_PROMPT_TEMPLATE_DUAL = """Question: {question}
Answer: {answer}"""


# ============================================================
# Backend 1: HuggingFace Transformers
# ============================================================

class HuggingFaceDualBackend:
    """
    Load Llama 3.1 8B trực tiếp bằng HuggingFace Transformers.
    4-bit quantization (~6GB VRAM).
    """

    def __init__(
        self,
        model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 200,
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("Cần cài: pip install torch transformers")

        logger.info(f"Loading {model_name} on {device} (4-bit quantization)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device,
        )

        self.device = device
        self.max_new_tokens = max_new_tokens
        logger.info("Model loaded successfully.")

    def generate(self, question: str, answer: str) -> str:
        """Gửi prompt tới model, trả về raw text output."""
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DUAL},
            {"role": "user", "content": USER_PROMPT_TEMPLATE_DUAL.format(
                question=question, answer=answer
            )},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================
# Backend 2: Ollama (Local API Server)
# ============================================================

class OllamaDualBackend:
    """
    Gọi Ollama API chạy local (http://localhost:11434).
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        logger.info(f"Ollama dual backend: model={model_name}, url={base_url}")

    def generate(self, question: str, answer: str) -> str:
        """Gửi request tới Ollama API, trả về raw text output."""
        import urllib.request

        payload = json.dumps({
            "model": self.model_name,
            "system": SYSTEM_PROMPT_DUAL,
            "prompt": USER_PROMPT_TEMPLATE_DUAL.format(
                question=question, answer=answer
            ),
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,
                "num_ctx": 512,
                "num_thread": 4,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()


# ============================================================
# Backend 3: Rule-based Fallback (Không cần LLM)
# ============================================================

class RuleBasedDualBackend:
    """
    Fallback đơn giản: dùng regex để extract cả positive và negative attributes.
    Nâng cấp từ NACIR cũ: giờ trả về CẢ 2 rổ thay vì chỉ 1.
    """

    _NEG_PATTERNS = [
        r"(?:no|not|don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|can'?t|won'?t)"
        r".*?\b(?:wear|have|carry|hold|see|any|the|a|an)\s+"
        r"((?:\w+\s+){0,3}\w+)",
        r"(?:don'?t think|don'?t see|don'?t believe)"
        r".*?\b(?:any|the|a|an)\s+"
        r"((?:\w+\s+){0,3}\w+)",
        r"(?:no|not|don'?t|doesn'?t)\s+(?:\w+\s+){0,4}"
        r"((?:[a-z]+\s+){0,2}[a-z]+)$",
    ]

    _POS_PATTERNS = [
        # "Yes, there is a red car" → "red car"
        r"(?:yes|yeah|yep|yup|sure|definitely|absolutely)"
        r".*?\b(?:is|are|has|have|there'?s|wearing|carrying|holding)\s+(?:a|an|the|some)?\s*"
        r"((?:\w+\s+){0,3}\w+)",
        # "It looks like a sunny day" → "sunny day"
        r"(?:looks?\s+like|appears?\s+to\s+be|seems?\s+like)\s+(?:a|an|the)?\s*"
        r"((?:\w+\s+){0,3}\w+)",
        # "It is outdoor" → "outdoor"
        r"(?:it\s+is|it'?s|they\s+are|he\s+is|she\s+is)\s+(?:a|an|the)?\s*"
        r"((?:\w+\s+){0,3}\w+)",
    ]

    _COMPILED_NEG = [re.compile(p, re.IGNORECASE) for p in _NEG_PATTERNS]
    _COMPILED_POS = [re.compile(p, re.IGNORECASE) for p in _POS_PATTERNS]

    _NEG_WORDS = re.compile(
        r"\b(?:no|not|nope|nah|don'?t|doesn'?t|didn'?t|can'?t|cannot|"
        r"won'?t|isn'?t|aren'?t|wasn'?t|weren'?t|hasn'?t|haven'?t|"
        r"couldn'?t|wouldn'?t|shouldn'?t|never|none|nothing|nobody|"
        r"neither)\b",
        re.IGNORECASE,
    )

    _POS_WORDS = re.compile(
        r"\b(?:yes|yeah|yep|yup|sure|definitely|absolutely|correct|"
        r"right|exactly|indeed)\b",
        re.IGNORECASE,
    )

    _STOPWORDS = {"a", "an", "the", "any", "some", "that", "this", "those",
                  "these", "i", "it", "is", "are", "was", "were", "be",
                  "been", "being", "have", "has", "had", "do", "does", "did",
                  "will", "would", "could", "should", "can", "may", "might",
                  "shall", "must", "need", "there", "here", "so", "very",
                  "really", "just", "too", "also", "but", "and", "or",
                  "if", "then", "than", "of", "in", "on", "at", "to",
                  "for", "with", "about", "by", "from", "up", "into"}

    def _clean_attribute(self, raw: str) -> Optional[str]:
        """Clean and validate extracted attribute."""
        words = raw.strip().split()
        words = [w for w in words if w.lower() not in self._STOPWORDS]
        if not words:
            return None
        return " ".join(words[:4])  # Max 4 words

    def generate(self, question: str, answer: str) -> str:
        """Trả về JSON string với cả positive và negative attributes."""
        answer_lower = answer.lower().strip()
        question_lower = question.lower().strip()

        positives = []
        negatives = []

        # Check for uncertainty — return empty
        uncertainty_patterns = [r"not sure", r"don'?t know", r"hard to tell", r"can'?t tell"]
        for p in uncertainty_patterns:
            if re.search(p, answer_lower):
                return json.dumps({"positives": [], "negatives": []})

        # Extract negatives
        has_neg = bool(self._NEG_WORDS.search(answer_lower))
        if has_neg:
            for pattern in self._COMPILED_NEG:
                m = pattern.search(answer_lower)
                if m:
                    attr = self._clean_attribute(m.group(1))
                    if attr:
                        negatives.append({"attribute": attr, "confidence": 0.85})
                        break  # Take first match

        # Extract positives
        has_pos = bool(self._POS_WORDS.search(answer_lower))
        if has_pos:
            for pattern in self._COMPILED_POS:
                m = pattern.search(answer_lower)
                if m:
                    attr = self._clean_attribute(m.group(1))
                    if attr:
                        positives.append({"attribute": attr, "confidence": 0.80})
                        break

        # Extract from question context if answer is simple "yes"/"no"
        if has_pos and not positives and question_lower:
            # "Does he wear a hat?" + "Yes" → positive: "hat"
            q_match = re.search(
                r"(?:does|do|is|are|has|have).*?\b(?:wear|have|carry|hold|"
                r"see|the|a|an)\s+((?:\w+\s+){0,3}\w+)",
                question_lower
            )
            if q_match:
                attr = self._clean_attribute(q_match.group(1))
                if attr:
                    positives.append({"attribute": attr, "confidence": 0.75})

        if has_neg and not negatives and question_lower:
            # "Does he wear a hat?" + "No" → negative: "hat"
            q_match = re.search(
                r"(?:does|do|is|are|has|have).*?\b(?:wear|have|carry|hold|"
                r"see|the|a|an)\s+((?:\w+\s+){0,3}\w+)",
                question_lower
            )
            if q_match:
                attr = self._clean_attribute(q_match.group(1))
                if attr:
                    negatives.append({"attribute": attr, "confidence": 0.75})

        return json.dumps({"positives": positives, "negatives": negatives})


# ============================================================
# Main Dual Extractor Class
# ============================================================

class DualExtractor:
    """
    NACIR++ Step 1 — Dual Signal Extractor

    Bóc tách tường minh cả rổ Positive và Negative bằng LLM.

    Ví dụ:
        >>> extractor = DualExtractor(backend="rule")
        >>> extractor.extract(
        ...     "No, he doesn't wear a black backpack but he has a red jacket",
        ...     "Does he wear a backpack?"
        ... )
        {
            "positives": [{"attribute": "red jacket", "confidence": 0.80}],
            "negatives": [{"attribute": "black backpack", "confidence": 0.85}]
        }
    """

    def __init__(
        self,
        backend: str = "ollama",
        model_name: Optional[str] = None,
        device: str = "cuda",
        ollama_url: str = "http://localhost:11434",
    ):
        self.backend_name = backend

        if backend == "huggingface":
            name = model_name or "unsloth/Meta-Llama-3.1-8B-Instruct"
            self.backend = HuggingFaceDualBackend(model_name=name, device=device)
        elif backend == "ollama":
            name = model_name or "llama3.1:8b"
            self.backend = OllamaDualBackend(model_name=name, base_url=ollama_url)
        elif backend == "rule":
            self.backend = RuleBasedDualBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'huggingface', 'ollama', or 'rule'")

    def extract(self, answer: str, question: str = "") -> Dict[str, List[Dict]]:
        """
        Extract cả rổ positive và negative attributes.

        Returns:
            {
                "positives": [{"attribute": str, "confidence": float}, ...],
                "negatives": [{"attribute": str, "confidence": float}, ...]
            }
        """
        try:
            raw_output = self.backend.generate(question=question, answer=answer)
            result = self._parse_json(raw_output)
            return self._validate(result)
        except Exception as e:
            logger.warning(f"DualExtract failed for answer='{answer[:50]}...': {e}")
            return {"positives": [], "negatives": []}

    def _parse_json(self, raw: str) -> Dict:
        """Parse JSON từ LLM output (có thể chứa text thừa)."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to find the outermost JSON object
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except json.JSONDecodeError:
                        pass
                    break

        logger.warning(f"Cannot parse dual JSON from: {raw[:100]}")
        return {"positives": [], "negatives": []}

    def _validate(self, result: Dict) -> Dict[str, List[Dict]]:
        """Validate và clean kết quả."""
        positives = result.get("positives", [])
        negatives = result.get("negatives", [])

        # Ensure lists
        if not isinstance(positives, list):
            positives = []
        if not isinstance(negatives, list):
            negatives = []

        # Clean each attribute
        clean_pos = []
        for item in positives:
            cleaned = self._clean_item(item)
            if cleaned:
                clean_pos.append(cleaned)

        clean_neg = []
        for item in negatives:
            cleaned = self._clean_item(item)
            if cleaned:
                clean_neg.append(cleaned)

        return {"positives": clean_pos, "negatives": clean_neg}

    def _clean_item(self, item: Any) -> Optional[Dict]:
        """Clean a single attribute item."""
        if isinstance(item, str):
            # LLM returned plain string instead of dict
            item = {"attribute": item, "confidence": 0.7}
        elif not isinstance(item, dict):
            return None

        attr = item.get("attribute")
        if attr is None:
            return None

        attr = str(attr).strip()
        if not attr or attr.lower() in ("null", "none", "n/a", ""):
            return None

        # Truncate to max 4 words
        words = attr.split()
        if len(words) > 4:
            attr = " ".join(words[:4])

        confidence = float(item.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        return {"attribute": attr, "confidence": confidence}

    def to_legacy_format(self, result: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Chuyển kết quả NACIR++ về format cũ để tương thích với NACIR modules.

        NACIR++ format:
            {"positives": [...], "negatives": [...]}

        NACIR legacy format:
            {"is_negative": bool, "attribute": str|None}
        """
        negatives = result.get("negatives", [])
        if negatives:
            # Lấy negative có confidence cao nhất
            best_neg = max(negatives, key=lambda x: x.get("confidence", 0))
            return {
                "is_negative": True,
                "attribute": best_neg["attribute"],
            }
        return {"is_negative": False, "attribute": None}


# ============================================================
# Batch Processing — Chạy offline trên toàn bộ dataset
# ============================================================

def run_batch_dual_extraction(
    queries_path: str,
    output_path: str,
    backend: str = "ollama",
    model_name: Optional[str] = None,
    device: str = "cuda",
    ollama_url: str = "http://localhost:11434",
    resume: bool = True,
) -> None:
    """
    Chạy Dual Extraction trên toàn bộ dataset, lưu kết quả vào JSON.

    Output format (data/dual_attributes.json):
        [
          {
            "dialog_id": 0,
            "turns": [
              {"turn": 0, "question": "...", "answer": "...",
               "positives": [...], "negatives": [...]},
              ...
            ]
          }
        ]
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.negative_detector import parse_visdial_dialog

    with open(queries_path) as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} dialogs from {queries_path}")

    extractor = DualExtractor(
        backend=backend,
        model_name=model_name,
        device=device,
        ollama_url=ollama_url,
    )
    logger.info(f"Using dual backend: {backend}")

    results = []
    done_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["dialog_id"] for r in results}
        logger.info(f"Resuming: {len(done_ids)} dialogs already done")

    for dialog_id, query in enumerate(tqdm(queries, desc="Dual Extracting")):
        if dialog_id in done_ids:
            continue

        turns = parse_visdial_dialog(query["dialog"])

        dialog_result = {"dialog_id": dialog_id, "turns": []}

        for turn_idx, turn in enumerate(turns):
            result = extractor.extract(
                answer=turn["answer"],
                question=turn["question"],
            )
            dialog_result["turns"].append({
                "turn": turn_idx,
                "question": turn["question"],
                "answer": turn["answer"],
                "positives": result["positives"],
                "negatives": result["negatives"],
            })

        results.append(dialog_result)

        if (dialog_id + 1) % 100 == 0:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(results)} dialogs to {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Done! Saved {len(results)} dialogs to {output_path}")

    # Statistics
    total_turns = sum(len(d["turns"]) for d in results)
    neg_turns = sum(
        1 for d in results for t in d["turns"] if t["negatives"]
    )
    pos_turns = sum(
        1 for d in results for t in d["turns"] if t["positives"]
    )
    total_neg_attrs = sum(
        len(t["negatives"]) for d in results for t in d["turns"]
    )
    total_pos_attrs = sum(
        len(t["positives"]) for d in results for t in d["turns"]
    )
    logger.info(f"Statistics:")
    logger.info(f"  Total turns:          {total_turns}")
    logger.info(f"  Turns with negatives: {neg_turns} ({neg_turns/total_turns*100:.1f}%)")
    logger.info(f"  Turns with positives: {pos_turns} ({pos_turns/total_turns*100:.1f}%)")
    logger.info(f"  Total neg attributes: {total_neg_attrs}")
    logger.info(f"  Total pos attributes: {total_pos_attrs}")


# ============================================================
# Load Pre-computed Dual Attributes
# ============================================================

def load_precomputed_dual_attributes(
    attributes_path: str,
) -> Dict[int, Dict[int, Dict]]:
    """
    Load file dual_attributes.json.

    Returns:
        {dialog_id: {turn_idx: {"positives": [...], "negatives": [...]}}}
    """
    with open(attributes_path) as f:
        raw = json.load(f)

    result = {}
    for dialog in raw:
        dialog_id = dialog["dialog_id"]
        result[dialog_id] = {}
        for turn in dialog["turns"]:
            result[dialog_id][turn["turn"]] = {
                "positives": turn.get("positives", []),
                "negatives": turn.get("negatives", []),
            }

    logger.info(
        f"Loaded dual attributes for {len(result)} dialogs "
        f"from {attributes_path}"
    )
    return result


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NACIR++ Step 1 — Dual Signal Extractor (Batch Mode)"
    )
    parser.add_argument(
        "--queries-path", type=str, required=True,
        help="Path to VisDial queries JSON"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/dual_attributes.json",
        help="Path to save dual extracted attributes"
    )
    parser.add_argument(
        "--backend", type=str, default="ollama",
        choices=["huggingface", "ollama", "rule"],
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )

    run_batch_dual_extraction(
        queries_path=args.queries_path,
        output_path=args.output_path,
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
        ollama_url=args.ollama_url,
        resume=not args.no_resume,
    )

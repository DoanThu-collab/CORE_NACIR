import json
import os
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# 1. Load pre-computed beliefs

def load_precomputed_beliefs(
    path: str,
) -> Dict[int, Dict[int, Dict]]:
    
    with open(path) as f:
        raw = json.load(f)

    result: Dict[int, Dict[int, Dict]] = {}

    if isinstance(raw, list):
        # Format A: list of dialog objects
        for dialog in raw:
            dialog_id = int(dialog["dialog_id"])
            result[dialog_id] = {}
            for turn in dialog.get("turns", []):
                turn_idx = int(turn["turn"])
                result[dialog_id][turn_idx] = {
                    "positive_beliefs": turn.get("positives",
                                                 turn.get("positive_beliefs", [])),
                    "negative_beliefs": turn.get("negatives",
                                                 turn.get("negative_beliefs", [])),
                }
    elif isinstance(raw, dict):
        # Format B: nested dict
        for did_str, turns in raw.items():
            dialog_id = int(did_str)
            result[dialog_id] = {}
            for tidx_str, beliefs in turns.items():
                turn_idx = int(tidx_str)
                result[dialog_id][turn_idx] = {
                    "positive_beliefs": beliefs.get("positive_beliefs",
                                                    beliefs.get("positives", [])),
                    "negative_beliefs": beliefs.get("negative_beliefs",
                                                    beliefs.get("negatives", [])),
                }
    else:
        raise ValueError(f"Unknown beliefs format in {path}")

    logger.info(f"Loaded pre-computed beliefs for {len(result)} dialogs from {path}")
    return result


# 2. Online Semantic Parser (wraps DualExtractor)

class SemanticParser:
    

    def __init__(
        self,
        backend: str = "rule",
        model_name: Optional[str] = None,
        device: str = "cuda",
        ollama_url: str = "http://localhost:11434",
    ):
        from core.dual_extractor import DualExtractor
        self.extractor = DualExtractor(
            backend=backend,
            model_name=model_name,
            device=device,
            ollama_url=ollama_url,
        )
        self.backend = backend
        logger.info(f"SemanticParser initialized with backend={backend}")

    def parse(
        self,
        answer: str,
        question: str = "",
    ) -> Dict[str, List[Dict]]:
        
        result = self.extractor.extract(answer=answer, question=question)
        return {
            "positive_beliefs": result.get("positives", []),
            "negative_beliefs": result.get("negatives", []),
        }


# 3. CLI: Generate beliefs offline

def generate_beliefs_offline(
    queries_path: str,
    output_path: str,
    backend: str = "rule",
    model_name: Optional[str] = None,
    device: str = "cuda",
    ollama_url: str = "http://localhost:11434",
) -> None:
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.negative_detector import parse_visdial_dialog
    from core.dual_extractor import DualExtractor
    from tqdm import tqdm

    with open(queries_path) as f:
        queries = json.load(f)

    extractor = DualExtractor(backend=backend, model_name=model_name,
                              device=device, ollama_url=ollama_url)

    results = []
    for dialog_id, query in enumerate(tqdm(queries, desc="Generating beliefs")):
        turns = parse_visdial_dialog(query["dialog"])
        dialog_result = {"dialog_id": dialog_id, "turns": []}

        for turn_idx, turn in enumerate(turns):
            result = extractor.extract(
                answer=turn["answer"], question=turn["question"],
            )
            dialog_result["turns"].append({
                "turn": turn_idx,
                "question": turn["question"],
                "answer": turn["answer"],
                "positives": result["positives"],
                "negatives": result["negatives"],
            })

        results.append(dialog_result)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved beliefs for {len(results)} dialogs → {output_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="NACIR++ — Generate Semantic Beliefs Offline"
    )
    parser.add_argument("--queries-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="data/semantic_beliefs.json")
    parser.add_argument("--backend", type=str, default="rule",
                        choices=["huggingface", "ollama", "rule"])
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    args = parser.parse_args()

    generate_beliefs_offline(
        queries_path=args.queries_path,
        output_path=args.output_path,
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
        ollama_url=args.ollama_url,
    )

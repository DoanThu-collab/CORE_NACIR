import os
import sys
import json
import time
import logging
import argparse
import urllib.request
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dual_extractor import run_batch_dual_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_QUERIES_PATH = "data/queries.json"
DEFAULT_OUTPUT_DIR = "data/beliefs"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

def check_ollama_running(url: str = DEFAULT_OLLAMA_URL) -> bool:
    try:
        req = urllib.request.Request(f"{url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False

def get_ollama_models(url: str = DEFAULT_OLLAMA_URL) -> list:
    try:
        req = urllib.request.Request(f"{url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

def model_to_filename(model_name: str) -> str:
    safe = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"beliefs_{safe}.json"

def generate_for_model(
    model_name: str,
    backend: str = "ollama",
    queries_path: str = DEFAULT_QUERIES_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    no_resume: bool = False,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    filename = "beliefs_rule_based.json" if backend == "rule" else model_to_filename(model_name)
    output_path = os.path.join(output_dir, filename)
    
    logger.info(f"Starting model: {model_name or 'rule-based'} (backend={backend})")
    logger.info(f"Ollama URL: {ollama_url}")
    logger.info(f"Output path: {output_path}")
    
    # Handle resuming
    if os.path.exists(output_path) and not no_resume:
        with open(output_path) as f:
            existing = json.load(f)
        with open(queries_path) as f:
            total = len(json.load(f))
            
        if len(existing) >= total:
            logger.info(f"Found {len(existing)} completed dialogs. Skipping generation.")
            return output_path
        else:
            logger.info(f"Found {len(existing)}/{total} dialogs. Resuming generation...")
    
    start_time = time.time()
    run_batch_dual_extraction(
        queries_path=queries_path,
        output_path=output_path,
        backend=backend,
        model_name=model_name,
        ollama_url=ollama_url,
        resume=not no_resume,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Completed in {timedelta(seconds=int(elapsed))}")
    return output_path

def print_beliefs_stats(path: str, model_name: str):
    with open(path) as f:
        data = json.load(f)
    
    total_turns = sum(len(d["turns"]) for d in data)
    pos_turns = sum(1 for d in data for t in d["turns"] if t.get("positives"))
    neg_turns = sum(1 for d in data for t in d["turns"] if t.get("negatives"))
    total_pos = sum(len(t.get("positives", [])) for d in data for t in d["turns"])
    total_neg = sum(len(t.get("negatives", [])) for d in data for t in d["turns"])
    
    # Calculate average confidence
    all_pos_conf = [a["confidence"] for d in data for t in d["turns"] for a in t.get("positives", []) if "confidence" in a]
    all_neg_conf = [a["confidence"] for d in data for t in d["turns"] for a in t.get("negatives", []) if "confidence" in a]
    avg_pos_conf = sum(all_pos_conf) / len(all_pos_conf) if all_pos_conf else 0
    avg_neg_conf = sum(all_neg_conf) / len(all_neg_conf) if all_neg_conf else 0
    
    logger.info(f"\nStats for {model_name}:")
    logger.info(f"Dialogs:            {len(data)}")
    logger.info(f"Total turns:        {total_turns}")
    logger.info(f"Turns w/ positives: {pos_turns} ({pos_turns/total_turns*100:.1f}%)")
    logger.info(f"Turns w/ negatives: {neg_turns} ({neg_turns/total_turns*100:.1f}%)")
    logger.info(f"Total pos attrs:    {total_pos} (avg conf: {avg_pos_conf:.3f})")
    logger.info(f"Total neg attrs:    {total_neg} (avg conf: {avg_neg_conf:.3f})")

def main():
    parser = argparse.ArgumentParser(description="NACIR++ — Multi-Model Semantic Beliefs Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Specific Ollama model name")
    group.add_argument("--models", type=str, nargs="+", help="List of models to process")
    group.add_argument("--all", action="store_true", help="Process all available models")
    
    parser.add_argument("--include-rule", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--queries-path", type=str, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--stats-only", action="store_true")
    
    args = parser.parse_args()
    
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    elif args.all:
        if not check_ollama_running(args.ollama_url):
            logger.error("Ollama server is not running")
            sys.exit(1)
        models = get_ollama_models(args.ollama_url)
        if not models:
            logger.error("No models found")
            sys.exit(1)
    
    ollama_url = args.ollama_url
    logger.info(f"Target Models: {models}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.stats_only and not check_ollama_running(ollama_url):
        logger.error("Ollama server is not running")
        sys.exit(1)
    
    results = {}
    
    # Rule-based baseline
    if args.include_rule:
        path = os.path.join(args.output_dir, "beliefs_rule_based.json")
        if not args.stats_only:
            path = generate_for_model(
                model_name="", backend="rule",
                queries_path=args.queries_path,
                output_dir=args.output_dir,
                no_resume=args.no_resume,
                ollama_url=ollama_url,
            )
        if os.path.exists(path):
            print_beliefs_stats(path, "rule-based")
            results["rule-based"] = path
    
    # LLM models
    for model in models:
        path = os.path.join(args.output_dir, model_to_filename(model))
        if not args.stats_only:
            try:
                path = generate_for_model(
                    model_name=model, backend="ollama",
                    queries_path=args.queries_path,
                    output_dir=args.output_dir,
                    no_resume=args.no_resume,
                    ollama_url=ollama_url,
                )
            except Exception as e:
                logger.error(f"Error processing model {model}: {e}")
                continue
        if os.path.exists(path):
            print_beliefs_stats(path, model)
            results[model] = path
    
    # Summary
    if results:
        logger.info("\nGenerated files:")
        for model, path in results.items():
            size_mb = os.path.getsize(path) / 1024 / 1024
            logger.info(f"{model:25s} -> {path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()

"""
NACIR++ — Run Semantic Parser (Batch Mode)
============================================
Chạy Bước 1 offline trên toàn bộ dataset → lưu data/semantic_beliefs.json.

Usage:
    # Dùng Rule-based backend (không cần GPU)
    python scripts/run_semantic_parser.py \\
        --queries-path ../PlugIR_Workspace/PlugIR/dialogues/VisDial_v1.0_queries_val.json \\
        --output-path data/semantic_beliefs.json \\
        --backend rule

    # Dùng Ollama (cần ollama server đang chạy)
    python scripts/run_semantic_parser.py \\
        --queries-path ../PlugIR_Workspace/PlugIR/dialogues/VisDial_v1.0_queries_val.json \\
        --output-path data/semantic_beliefs.json \\
        --backend ollama

    # Dùng HuggingFace Transformers
    python scripts/run_semantic_parser.py \\
        --queries-path ../PlugIR_Workspace/PlugIR/dialogues/VisDial_v1.0_queries_val.json \\
        --output-path data/semantic_beliefs.json \\
        --backend huggingface --device cuda
"""

import os
import sys
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nacir_plus.semantic_parser import run_batch_semantic_parsing


def main():
    parser = argparse.ArgumentParser(
        description="NACIR++ — Run Semantic Parser (Batch Mode)"
    )
    parser.add_argument(
        "--queries-path", type=str, required=True,
        help="Path to VisDial queries JSON",
    )
    parser.add_argument(
        "--output-path", type=str, default="data/semantic_beliefs.json",
        help="Path to save extracted beliefs",
    )
    parser.add_argument(
        "--backend", type=str, default="rule",
        choices=["huggingface", "ollama", "rule"],
        help="LLM backend to use (default: rule)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model name (default depends on backend)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for HuggingFace backend",
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume from previous results",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("NACIR++ — Bước 1: Semantic Parser (Batch Mode)")
    logger.info("=" * 60)
    logger.info(f"Backend:     {args.backend}")
    logger.info(f"Queries:     {args.queries_path}")
    logger.info(f"Output:      {args.output_path}")
    logger.info("=" * 60)

    run_batch_semantic_parsing(
        queries_path=args.queries_path,
        output_path=args.output_path,
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
        ollama_url=args.ollama_url,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

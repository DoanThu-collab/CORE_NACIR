# NACIR: Negative-aware Adaptive Concept Interactive Retrieval

Official implementation of the paper **NACIR: Negative-aware Adaptive Concept Interactive Retrieval** for multi-turn image retrieval.

## 🚀 Execution Pipeline (Two-Stage Architecture)

To maximize efficiency and prevent Out-Of-Memory (OOM) errors during hyperparameter tuning, the NACIR framework is purposefully decoupled into two independent execution stages:

### Step 1: Offline Semantic Parsing (`step1_semantic_parsing/`)
- **What it does:** Uses Large Language Models (e.g., LLaMA-3.1 8B, Qwen) or Rule-based engines to parse the entire dialogue dataset into structured positive/negative beliefs via zero-shot comprehension.
- **Why offline?** LLM inference is computationally expensive. By running it once and saving the output to `.json` files, we avoid running LLMs repeatedly and reduce GPU memory overhead.

### Step 2: Online Core Retrieval (`step2_core_retrieval/`)
- **What it does:** The lightweight algorithmic core of NACIR. It loads the pre-computed `.json` beliefs and executes the fast retrieval pipeline: Concept Memory Board $\rightarrow$ Gram-Schmidt Orthogonal Projection $\rightarrow$ Negative Concept Masking $\rightarrow$ BLIP ITM Re-ranking.
- **Why online?** This decoupling allows researchers to run the core retrieval pipeline hundreds of times in mere minutes for rapid hyperparameter tuning without the bottleneck of LLM inference.

## 📁 Repository Structure

```text
NACIR_Release/
├── README.md
├── step1_semantic_parsing/
│   ├── dual_extractor.py               # Single-pass LLM & Rule-based extraction logic
│   ├── run_semantic_parser.py          # Script for batch parsing dialogues
│   └── sample_beliefs/                 # Sample pre-computed beliefs (e.g., LLaMA-3.1 8B)
│
└── step2_core_retrieval/
    ├── nacir_plusplus/                 # Core retrieval algorithms
    ├── examples/                       # Adapter scripts for base methods (e.g., PlugIR)
    └── scripts/                        # Execution scripts
```

## 🛠️ Usage

### 1. Generate Beliefs (Step 1)
Run the semantic parser to generate beliefs for your dialogue dataset:
```bash
cd step1_semantic_parsing
python run_semantic_parser.py --backend ollama --model_name llama3.1:8b
```

### 2. Run Core Retrieval (Step 2)
Use the generated beliefs to run the interactive retrieval pipeline:
```bash
cd step2_core_retrieval/scripts
bash run_batch_1.sh
```

## 📝 Citation
If you find our work useful, please consider citing our paper.

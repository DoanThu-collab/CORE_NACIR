# NACIR

NACIR is a plug-and-play retrieval refinement framework for conversational image retrieval. It can be integrated with different retrieval backbones (e.g., PlugIR) without modifying the backbone itself.

---

## Repository Structure

```text
.
├── adapters/
│   └── plugir_backbone.py      # Adapter between PlugIR and NACIR
├── pipeline.py                 # Main NACIR pipeline
├── schedule.py                 # Dynamic scheduling strategy
├── schema.py                   # Shared data structures
├── config.py                   # Hyperparameter configuration
├── metrics.py                  # Evaluation metrics
├── utils.py
└── run_plugir.py               # Example entry point
```

---

## Pipeline

The execution flow is illustrated below:

```
Dialogue
    │
    ▼
PlugIR
    │
    ▼
Text / Image Embeddings
    │
    ▼
NACIR Pipeline
    │
    ├── Semantic Parsing
    ├── Concept Memory Update
    ├── Dynamic Orthogonal Projection
    ├── Region-Level Masking
    └── ITM Re-ranking
    │
    ▼
Final Retrieval Ranking
```

`pipeline.py` is the main entry point of NACIR, which coordinates all refinement stages.

---

## Core Files

### `pipeline.py`

The main retrieval pipeline.

Responsibilities:

- Receive embeddings from the backbone.
- Update concept memory.
- Apply dynamic projection.
- Perform region masking.
- Execute ITM re-ranking.
- Return the final ranked results.

---

### `schedule.py`

Defines the dynamic scheduling strategy.

It controls how strongly NACIR applies projection, masking, and re-ranking at different dialogue turns.

---

### `schema.py`

Defines the shared data structures used across the framework, including dialogue state, concept memory, and retrieval outputs.

---

### `config.py`

Contains all configurable hyperparameters for NACIR.

Examples include:

- projection strength
- masking penalty
- scheduling parameters
- ITM weight

---

### `adapters/plugir_backbone.py`

Adapter that connects PlugIR with NACIR.

It converts PlugIR outputs into the interfaces required by the NACIR pipeline.

---

## Integrating a New Retrieval Backbone

NACIR is backbone-agnostic.

To integrate a new conversational retrieval model, only an adapter is required.

### Step 1

Create a new adapter inside `adapters/`

Example:

```text
adapters/
    plugir_backbone.py
    my_model_backbone.py
```

---

### Step 2

The adapter should expose the following interfaces:

```python
encode_text(...)
encode_image(...)
score_itm(...)
```

These methods provide the components required by NACIR.

---

### Step 3

Initialize the pipeline with the new adapter.

```python
backbone = MyModelBackbone(...)

pipeline = NACIRPipeline(
    backbone=backbone,
    config=config,
)

results = pipeline.run(dialogue)
```

No modification to the NACIR core is required.

---

## Adding New Components

The framework is modular.

To replace or extend any component:

- Scheduling → modify `schedule.py`
- Retrieval pipeline → modify `pipeline.py`
- Backbone interface → add a new adapter
- Hyperparameters → update `config.py`

The remaining modules can remain unchanged.

---

## Running

Example:

```bash
python run_plugir.py
```

---
# NACIR

NACIR++ is a plug-and-play retrieval refinement framework for conversational image retrieval. It can be integrated with different retrieval backbones (e.g., PlugIR) without modifying the backbone itself.

---

## Repository Structure

```text
.
├── adapters/
│   └── plugir_backbone.py      # Adapter between PlugIR and NACIR++
├── pipeline.py                 # Main NACIR++ pipeline
├── schedule.py                 # Dynamic scheduling strategy
├── schema.py                   # Shared data structures
├── config.py                   # Hyperparameter configuration
├── metrics.py                  # Evaluation metrics
├── utils.py
└── run_plugir.py               # Example entry point
```

---

## Pipeline

The execution flow is illustrated below:

```
Dialogue
    │
    ▼
PlugIR
    │
    ▼
Text / Image Embeddings
    │
    ▼
NACIR++ Pipeline
    │
    ├── Semantic Parsing
    ├── Concept Memory Update
    ├── Dynamic Orthogonal Projection
    ├── Region-Level Masking
    └── ITM Re-ranking
    │
    ▼
Final Retrieval Ranking
```

`pipeline.py` is the main entry point of NACIR++, which coordinates all refinement stages.

---

## Core Files

### `pipeline.py`

The main retrieval pipeline.

Responsibilities:

- Receive embeddings from the backbone.
- Update concept memory.
- Apply dynamic projection.
- Perform region masking.
- Execute ITM re-ranking.
- Return the final ranked results.

---

### `schedule.py`

Defines the dynamic scheduling strategy.

It controls how strongly NACIR++ applies projection, masking, and re-ranking at different dialogue turns.

---

### `schema.py`

Defines the shared data structures used across the framework, including dialogue state, concept memory, and retrieval outputs.

---

### `config.py`

Contains all configurable hyperparameters for NACIR++.

Examples include:

- projection strength
- masking penalty
- scheduling parameters
- ITM weight

---

### `adapters/plugir_backbone.py`

Adapter that connects PlugIR with NACIR++.

It converts PlugIR outputs into the interfaces required by the NACIR++ pipeline.

---

## Integrating a New Retrieval Backbone

NACIR++ is backbone-agnostic.

To integrate a new conversational retrieval model, only an adapter is required.

### Step 1

Create a new adapter inside `adapters/`

Example:

```text
adapters/
    plugir_backbone.py
    my_model_backbone.py
```

---

### Step 2

The adapter should expose the following interfaces:

```python
encode_text(...)
encode_image(...)
score_itm(...)
```

These methods provide the components required by NACIR++.

---

### Step 3

Initialize the pipeline with the new adapter.

```python
backbone = MyModelBackbone(...)

pipeline = NACIRPipeline(
    backbone=backbone,
    config=config,
)

results = pipeline.run(dialogue)
```

No modification to the NACIR++ core is required.

---

## Adding New Components

The framework is modular.

To replace or extend any component:

- Scheduling → modify `schedule.py`
- Retrieval pipeline → modify `pipeline.py`
- Backbone interface → add a new adapter
- Hyperparameters → update `config.py`

The remaining modules can remain unchanged.

---

## Running

Example:

```bash
python run_plugir.py
```

---

## Citation

If you find NACIR++ useful in your research, please consider citing our work:

```bibtex
@inproceedings{2026nacir,
  title     = {NACIR: Negative-aware Adaptive Concept Interactive Retrieval for Conversational Image Retrieval},
  author    = {Thu Doan Nguyen Minh, Thuy Nguyen Thi Nhu},
  booktitle = {Proceedings of the IEEE RIVF International Conference on Computing and Communication Technologies},
  year      = {2026}
}
```
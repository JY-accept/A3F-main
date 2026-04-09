# A3F: Adversarial-Aware Adaptive Filtering

> **Learning Beneficial Noise Distributions to Enhance the Inference Ability of Large Language Models**

---

## Overview

**A3F** is a Retrieval-Augmented Generation (RAG) framework that *learns from noise* rather than simply filtering it out.  Instead of treating all retrieved noise as harmful, A3F classifies noise into four fine-grained categories, trains the model adversarially against each type, and uses a joint classification objective so the model can actively identify the noise it encounters at inference time.

### Key contributions

| # | Contribution |
|---|---|
| 1 | **Fine-grained noise taxonomy** – four categories aligned with real retrieval environments: *relevant*, *counterfactual*, *redundant*, *irrelevant* |
| 2 | **A3F framework** – KNN-based relevance ranking → LLM feedback → min-ave adversarial training → regularisation |
| 3 | **Joint optimisation** – auxiliary noise-classification head shares the base LLM encoder; jointly trained with the generation head |

---

## Architecture

```
Retrieved docs
      │
      ▼
┌─────────────────────────────────────────┐
│  Step 1 – Fine-grained Noise Taxonomy   │
│  relevant / counterfactual /            │
│  redundant / irrelevant                 │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Step 2 – A3F Framework                 │
│  ① IG-based candidate extraction        │
│  ② KNN similarity ranking & weighting   │
│  ③ LLM feedback & noise amplification   │
│  ④ Min-ave adversarial training         │
│  ⑤ Regularisation term L_reg            │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Step 3 – Joint Optimisation            │
│  L_A3F = w_gen · L_gen + w_cls · L_cls  │
│  4-class noise classification head      │
└─────────────────────────────────────────┘
```

---

## Repository structure

```
A3F/
├── train.py                   # Training entry point
├── evaluate.py                # Evaluation entry point
├── requirements.txt
├── a3f/
│   ├── data/
│   │   └── data_manager.py    # Data loading & tokenisation
│   ├── trainer/
│   │   └── a3f_trainer.py     # Core trainer + joint loss
│   ├── evaluate/
│   │   └── evaluator.py       # EM / F1 evaluation helpers
│   ├── metrics/
│   │   └── em_f1.py           # Exact-Match and F1 metrics
│   └── utils/
│       ├── config.py          # Argument parsing
│       ├── loader.py          # Data / model loading
│       ├── answer_processor.py# Prompt building & vLLM inference
│       ├── noise_constructor.py# Four-type noise selectors
│       └── candidate_extractor.py # IG scoring + KNN expansion
└── scripts/
    ├── train.sh               # Multi-GPU training launcher
    ├── evaluate.sh            # Full evaluation suite
    └── build_noise_data.py    # Noise corpus construction (Algorithm 1)
```

---

## Setup

```bash
git clone https://github.com/your-org/A3F.git
cd A3F
pip install -r requirements.txt
```

---

## Data preparation

### Option A – Use your own QA dataset

Your raw JSON must contain per-sample:
- `question` (str), `answers` (list[str])
- `best_ctx` (dict with `"text"` key) – golden document
- `ctxs` (list[dict]) – DPR-retrieved passages (`has_answer` bool per entry)
- *(optional)* `counter_fac` (list[str]) – counterfactual passages

Run the noise-construction script (Algorithm 1):
```bash
python scripts/build_noise_data.py \
    --input_path  data/raw_train.json \
    --output_path data/train.json \
    --split       train
```

### Option B – Use the RAG-Bench benchmark

Download from the link in your institution's data-access agreement, then run
the noise-construction script above.

---

## Training

Edit `scripts/train.sh` to set `MODEL_PATH`, then:

```bash
cd scripts
bash train.sh
```

Key hyperparameters (all tunable via CLI):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--w_gen` | 0.7 | Generative loss weight (Eq. 16) |
| `--w_cls` | 0.3 | Classification loss weight (Eq. 16) |
| `--w_reg` | 0.5 | Regularisation weight (Eq. 12) |
| `--knn_k` | 5   | KNN neighbours (Section 3.3) |
| `--ig_threshold` | 0.3 | Information-gain threshold τ |
| `--learning_rate` | 5e-6 | AdamW LR |
| `--num_train_epochs` | 3 | Training epochs |

---

## Evaluation

Edit `scripts/evaluate.sh` to set `MODEL`, then:

```bash
cd scripts
bash evaluate.sh
```

Or run individual modes:

```bash
# Golden retrieval only
python evaluate.py --w_one_retrieval --retrieve_type best \
    --test_model_name_or_path checkpoints/a3f/epoch_2.bin \
    --test_data_path data/test_nq.json \
    --result_save results/golden.json

# 40 % mixed noise
python evaluate.py --w_noisy_retrieval --noise_ratio 0.4 \
    --test_model_name_or_path checkpoints/a3f/epoch_2.bin \
    --test_data_path data/test_nq.json \
    --result_save results/noisy_40.json
```

---

## Main results (Table 3 in the paper)

| Model | Dataset | EM | F1 |
|-------|---------|----|----|
| A3F (LLaMA-3-7B) | NQ | **46.8** | **58.5** |
| A3F (LLaMA-3-7B) | TriviaQA | **50.2** | 59.8 |
| A3F (LLaMA-3-7B) | WebQuestions | **44.5** | **55.2** |
| A3F (DeepSeek-R1-7B) | NQ | **47.5** | **59.2** |
| A3F (DeepSeek-R1-7B) | TriviaQA | 48.0 | **64.5** |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{a3f2025,
  title   = {Learning Beneficial Noise Distributions to Enhance the
             Inference Ability of Large Language Models},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

# SCAN Task Experiments: Systematicity in Neural Networks

This repository contains code for reproducing Lake & Baroni's SCAN experiments and an individual project investigating pre-trained Transformer fine-tuning on compositional generalization tasks.

## Quick Start

**Group Experiments (1, 2, 3)**: Open and run one of these notebooks:
- robin/experiments/experiment_1_simple_split.ipynb
- robin/experiments/experiment_2_length_split.ipynb
- robin/experiments/experiment_3_add_prim_split.ipynb

**Individual Project (GPT-2 Fine-tuning, Colab only)**: Open and run on Google Colab:
- robin/experiments/additional_experiment_colab.ipynb

---

## Prerequisites

Install dependencies before running notebooks:

```bash
pip install -r requirements.txt
```

The requirements include: PyTorch, Transformers (HuggingFace), NumPy, Pandas, Matplotlib, and Jupyter.

---

## Important: Path Configuration

⚠️ The notebook files contain paths that may need adjustment for your system.

### For Group Experiments (Experiments 1, 2, 3)
- Data paths: data/simple_split/tasks_train_simple.txt, data/length_split/tasks_train_length.txt, etc.
- Results output: results/experiment_1/, results/experiment_2/, etc.
If you keep the repository structure as-is, these relative paths should work.

### For Individual Project (Colab Notebook)
- Cell 2: Git repo cloned to /content/gensys_additional-experiment/
- Cell 4: Root set to /content/gensys_additional-experiment/robin/
- Data paths: data/simple_split/tasks_train_simple.txt; data/length_split/tasks_train_length.txt
- Outputs: data/gpt2_finetune/simple_split/train_p{percentage}.txt; results/additional_experiment/ (saved to Drive)

⚠️ Local hardcoded paths in final plotting cells (only for local analysis; not used in Colab):
C:\Users\MatsEllingsen\onedrive - university of copenhagen\ucph\fall2025\anlp\gensys\robin\experiments\results\experiment_1\experiment_1_full.181220250025.json
C:\Users\MatsEllingsen\onedrive - university of copenhagen\ucph\fall2025\anlp\gensys\robin\experiments\results\experiment_2\experiment_2_full.181220250249.json
C:\Users\MatsEllingsen\onedrive - university of copenhagen\ucph\fall2025\anlp\gensys\robin\experiments\results\additional_experiment\gpt2_experiment_simple_split.130120261511.json
C:\Users\MatsEllingsen\onedrive - university of copenhagen\ucph\fall2025\anlp\gensys\robin\experiments\results\additional_experiment\gpt2_experiment_length_split.130120261511.json
Update these if running locally.

---


## Part 1: Group Reimplementation (Experiments 1, 2, 3)

Open the notebook and run all cells.

### Experiment 1: Simple Split
- File: robin/experiments/experiment_1_simple_split.ipynb
- Trains baseline on 1%, 2%, 4%, 8%, 16%, 32%, 64%, 100% data; saves to results/experiment_1/experiment_1_full.json
- Runtime: ~30–60 minutes (GPU)

### Experiment 2: Length Split
- File: robin/experiments/experiment_2_length_split.ipynb
- Trains baseline; evaluates without oracle and with oracle; saves to results/experiment_2/experiment_2_full.json
- Runtime: ~20–40 minutes

### Experiment 3: Add-Primitive Split
- File: robin/experiments/experiment_3_add_prim_split.ipynb
- Tests held-out primitives with varying composed examples; saves to results/experiment_3/experiment_3_full.json
- Runtime: ~30–50 minutes

---

## Part 2: Individual Project – GPT-2 Fine-Tuning (Colab)

File: robin/experiments/additional_experiment_colab.ipynb

Steps (in Colab):
1. Open the notebook (File > Open notebook > GitHub or upload).
2. Run all cells sequentially.

Built-in setup:
- Cell 1–2: Install deps, mount Drive
- Cell 3–4: Clone repo, set CWD to /content/gensys_additional-experiment/robin/
- Remaining cells: load data, fine-tune GPT-2, evaluate, save results to Drive

Expected runtime:
- A100: ~1–2 hours total
- T4: ~3–5 hours total

Key hyperparameters:
- Exp 1b: lr 7e-4, batch 64, steps 1,562, warmup 500
- Exp 2: lr 2e-4, batch 16, steps 6,250, warmup 500

Data format:
- Original: "IN: command OUT: actions"
- GPT-2 format: "command <SEP> actions <EOS>"

Oracle length decoding (Exp 2): custom logits processor forbids EOS before target length and forces EOS at the target position.

---

## Expected Results (summary)

- Exp 1b: Baseline 63→97% token acc; GPT-2 32→67% token acc; ~30pp gap
- Exp 2 (no oracle): Sequence ~0 for both; baseline token 62–91%; GPT-2 token 17–33%
- Exp 2 (oracle): Baseline seq 84.7%→0% (len 24→48); GPT-2 seq 42.9%→0%; both decay with length

---

## Output Files

- results/experiment_1/experiment_1_full.json
- results/experiment_2/experiment_2_full.json
- results/experiment_3/experiment_3_full.json
- results/additional_experiment/gpt2_experiment_simple_split.json
- results/additional_experiment/gpt2_experiment_length_split.json

---

## Troubleshooting

- OOM: reduce batch size; on Colab prefer A100 if available; run one fraction at a time.
- Data not found: ensure directory structure matches the layout above.
- GPU not detected: check torch.cuda.is_available(); select GPU runtime in Colab.

---

## Citation

Lake, B. M., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. Proceedings of ICML 35, 2873–2882.

---

## Files in This Repository

- README.md – This file
- robin/ – Group code (Experiments 1–3) and individual project (GPT-2 on Colab)
- requirements.txt – Python dependencies

# `testH.ipynb` pipeline (scripted)

The notebook `testH.ipynb` contains:

- dataset preprocessing (German credit + wine)
- PCA feature reduction
- QSVM training with a **GA-evolved feature map circuit** (via `qoop`)
- optional classical SVM baseline

To make this pipeline reusable, the “useful functions” were moved into `src/` and a runner script `run_testH.py` was added.

## What was refactored into `src/`

- `src/datasets.py`
  - `preprocess_all()` loads + standard-scales datasets
  - `generate_credit()` / `generate_wine()` apply PCA with caching (mirrors the notebook behavior)
- `src/models.py`
  - `train_qsvm_with_dataset()` trains `QSVC` using a `QuantumKernel`
  - `train_svm()` trains a classical RBF SVM baseline
- `src/ga_qsvm.py`
  - `optimize_qsvm_credit()` and `optimize_qsvm_wine()` run the GA loop and return a list of best accuracies per feature count

## Run the refactored pipeline

From the repo root:

```bash
python run_testH.py --dataset credit
```

For wine:

```bash
python run_testH.py --dataset wine
```

## Reproducibility notes

- The GA loop uses randomness (mutation, and random RX/RY/RZ allocation). The script sets global seeds via `src/randomness.py`, but **full bit-for-bit determinism is not guaranteed** across machines/environments because the underlying evolution and quantum simulation stack can introduce non-determinism.
- The runner includes an optional `--verify` flag that compares the output list to a stored reference list (from `testH.ipynb`) for the credit dataset:

```bash
python run_testH.py --dataset credit --verify
```

If this prints `matches_expected_credit_50pct: False`, it usually means:
- dependencies are different (Qiskit/QML versions)
- dataset sampling differs (seed/sample frac)
- GA randomness differs (environment/library changes)



# Getting started

## Prerequisites

- Python 3.10+ (this repo is often used with Anaconda on Windows)

## Install dependencies

From the repo root:

```bash
python -m pip install -r requirements.txt
```

Notes:
- `qiskit`, `qiskit-aer`, and `qiskit-machine-learning` are required for QSVM runs.
- If you prefer using the dependency list under `qoop/`, you can also use `qoop/requirements.txt`, but `requirements.txt` is the recommended “project-level” list.

## Data files

- German credit dataset file used by `testH.ipynb` refactor: `germancredit_data_updated.csv` (already in repo root).

## Quick sanity checks

```bash
python -c "import numpy, pandas, sklearn; print('ok')"
python -c "import qiskit; print(qiskit.__version__)"
```



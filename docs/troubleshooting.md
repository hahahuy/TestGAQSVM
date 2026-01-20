# Troubleshooting

## `ModuleNotFoundError: No module named 'qiskit'`

Install the repo dependencies:

```bash
python -m pip install -r requirements.txt
```

## Qiskit install issues on Windows / conda

- Prefer installing into a clean environment.
- If you use conda:

```bash
python -m pip install -r requirements.txt
```

(This repo currently documents pip-based installs; conda-forge alternatives are also possible, but version matching matters for reproducibility.)

## `qoop` imports fail

This repo vendors `qoop/` as source code. Make sure you run scripts from the **repo root** so Python can import it.

Example:

```bash
python run_testH.py --dataset credit
```

## Results differ from the notebook

This is common with GA-based pipelines.

- Ensure dataset sampling + split seeds match (`random_state=42` in the refactor).
- Ensure you’re using the same dependency versions (especially Qiskit + QML).
- Even with seeds set, some parts of the stack may not be perfectly deterministic.



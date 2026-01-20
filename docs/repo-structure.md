# Repository structure

## Top-level

- `src/`: refactored, reusable Python modules extracted from `testH.ipynb`
- `run_testH.py`: command-line runner that reproduces the `testH.ipynb` “GA + QSVM” workflow
- `testH.ipynb`: main experimental notebook (source of refactor)
- `test.ipynb`: additional experiments / earlier explorations
- `germancredit_data_updated.csv`: German credit dataset used by the pipeline
- `result/`: stored experiment artifacts (circuits/JSON results)
- `qoop/`: local copy of the **qoop** library (GA/evolution utilities, quantum compilation tools)

## `qoop/` (local dependency)

`qoop/` provides the GA “environment” and operators used to evolve circuits:

- `qoop/evolution/`: GA operators (selection, crossover, mutation), environments, synthesis metadata
- `qoop/backend/`: small helpers/constants used by the evolution code
- `qoop/core/`, `qoop/compilation/`, `qoop/vqe/`: additional qoop features (not all used by the `testH` pipeline)

Upstream documentation for qoop is in `qoop/README.md` and its wiki link.



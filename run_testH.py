from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import preprocess_all
from src.ga_qsvm import optimize_qsvm_credit, optimize_qsvm_wine
from src.randomness import set_global_seeds


EXPECTED_CREDIT_50PCT = [
    0.68,
    0.7,
    0.6933333333333334,
    0.7133333333333334,
    0.7133333333333334,
    0.7266666666666667,
    0.7533333333333333,
    0.72,
    0.74,
    0.74,
    0.7466666666666667,
    0.7533333333333333,
    0.7266666666666667,
    0.7266666666666667,
    0.7133333333333334,
    0.7266666666666667,
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the testH.ipynb GA+QSVM pipeline as a script."
    )
    parser.add_argument("--dataset", choices=["credit", "wine"], default="credit")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_global_seeds()
    repo_root = Path(__file__).resolve().parent
    store = preprocess_all(repo_root=repo_root)

    if args.dataset == "wine":
        result = optimize_qsvm_wine(store, verbose=args.verbose)
    else:
        result = optimize_qsvm_credit(store, verbose=args.verbose)

    print(result)
    if args.verify and args.dataset == "credit":
        print("matches_expected_credit_50pct:", result == EXPECTED_CREDIT_50PCT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



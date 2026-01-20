from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Seeds:
    python: int = 42
    numpy: int = 42


def set_global_seeds(seeds: Seeds = Seeds()) -> None:
    """Best-effort determinism for the notebook pipeline."""
    os.environ.setdefault("PYTHONHASHSEED", str(seeds.python))
    random.seed(seeds.python)
    np.random.seed(seeds.numpy)



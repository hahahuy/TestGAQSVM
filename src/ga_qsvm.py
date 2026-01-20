from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from qoop.backend.constant import operations_with_rotations
from qoop.evolution import divider, normalizer
from qoop.evolution.crossover import onepoint
from qoop.evolution.environment import EEnvironment
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.generator import by_num_rotations_and_cnot
from qoop.evolution.mutate import bitflip_mutate_with_normalizer_testing
from qoop.evolution.threshold import synthesis_threshold

from .datasets import DataStore, generate_credit, generate_wine
from .models import train_qsvm_with_dataset


@dataclass(frozen=True)
class GAParams:
    prob_mutate_values: np.ndarray
    num_generation: int
    num_circuit: int = 8


def _build_environment(
    *,
    num_features: int,
    ga: GAParams,
    prob_mutate: float,
    fitness_func: Callable,
) -> EEnvironment:
    # Notebook logic: num_cnot = num_features + 1, random split of rx/ry/rz counts
    num_cnot = num_features + 1
    num_rx = random.randint(0, num_features)
    num_ry = random.randint(0, num_features - num_rx)
    num_rz = num_features - (num_rx + num_ry)

    metadata = MetadataSynthesis(
        num_qubits=num_features,
        num_cnot=num_cnot,
        num_rx=num_rx,
        num_ry=num_ry,
        num_rz=num_rz,
        depth=4 + (num_features - 4) * 1,
        num_circuit=ga.num_circuit,
        num_generation=ga.num_generation,
        prob_mutate=prob_mutate,
    )

    return EEnvironment(
        metadata=metadata,
        fitness_func=fitness_func,
        generator_func=by_num_rotations_and_cnot,
        crossover_func=onepoint(
            divider.by_num_rotation_gate(int(metadata.num_qubits / 2)),
            normalizer.by_num_rotation_gate(metadata.num_qubits),
        ),
        mutate_func=bitflip_mutate_with_normalizer_testing(
            pool=operations_with_rotations,
            normalizer_func=normalizer.by_num_rotation_gate(metadata.num_qubits),
            prob_mutate=prob_mutate,
            num_qubits=num_features,
        ),
        threshold_func=synthesis_threshold,
    )


def optimize_qsvm_wine(
    store: DataStore,
    *,
    feature_range: Iterable[int] = range(4, 14),
    ga: Optional[GAParams] = None,
    evol_mode: Literal["parallel", "noparallel"] = "parallel",
    verbose: bool = False,
) -> List[float]:
    if ga is None:
        ga = GAParams(prob_mutate_values=np.linspace(0.01, 0.2, 10), num_generation=30)

    best_accuracies: List[float] = []

    for num_features in feature_range:
        best_accuracy = 0.0

        X_train, y_train, X_test, y_test = generate_wine(store, num_features)

        for prob_mutate in ga.prob_mutate_values:

            def fitness(qc):
                return train_qsvm_with_dataset(qc, X_train, y_train, X_test, y_test)

            env = _build_environment(
                num_features=num_features,
                ga=ga,
                prob_mutate=float(prob_mutate),
                fitness_func=fitness,
            )
            env.evol(verbose=verbose, mode=evol_mode)
            best_accuracy = max(best_accuracy, float(env.best_fitness))
            if verbose:
                print(best_accuracy)

        best_accuracies.append(best_accuracy)

    return best_accuracies


def optimize_qsvm_credit(
    store: DataStore,
    *,
    feature_range: Iterable[int] = range(4, 20),
    ga: Optional[GAParams] = None,
    evol_mode: Literal["parallel", "noparallel"] = "noparallel",
    verbose: bool = False,
) -> List[float]:
    if ga is None:
        ga = GAParams(prob_mutate_values=np.linspace(0.01, 0.2, 5), num_generation=20)

    best_accuracies: List[float] = []

    for num_features in feature_range:
        best_accuracy = 0.0
        X_train, y_train, X_test, y_test = generate_credit(store, num_features)

        for prob_mutate in ga.prob_mutate_values:
            fitness_cache: Dict[str, float] = {}

            def fitness(qc):
                qc_str = str(qc)
                if qc_str not in fitness_cache:
                    fitness_cache[qc_str] = train_qsvm_with_dataset(
                        qc, X_train, y_train, X_test, y_test
                    )
                return fitness_cache[qc_str]

            env = _build_environment(
                num_features=num_features,
                ga=ga,
                prob_mutate=float(prob_mutate),
                fitness_func=fitness,
            )
            env.evol(verbose=verbose, mode=evol_mode)
            best_accuracy = max(best_accuracy, float(env.best_fitness))

        best_accuracies.append(best_accuracy)

    return best_accuracies



from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _get_aer_backend_statevector():
    # qiskit>=1 often uses qiskit_aer; older notebooks use qiskit.Aer
    try:
        from qiskit_aer import Aer  # type: ignore

        return Aer.get_backend("statevector_simulator")
    except Exception:
        from qiskit import Aer  # type: ignore

        return Aer.get_backend("statevector_simulator")


def train_qsvm_with_dataset(
    quantum_circuit: Any, X_train, y_train, X_test, y_test
) -> float:
    # Imports kept inside for faster import of non-quantum paths
    from qiskit_machine_learning.algorithms import QSVC  # type: ignore
    from qiskit_machine_learning.kernels import QuantumKernel  # type: ignore

    backend = _get_aer_backend_statevector()
    quantum_kernel = QuantumKernel(feature_map=quantum_circuit, quantum_instance=backend)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)
    return float(accuracy_score(y_test, y_pred))


def train_svm(feature_num: int, generate_dataset) -> float:
    X_train, y_train, X_test, y_test = generate_dataset(feature_num)
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    svm_pipeline.fit(X_train, y_train)
    return float(svm_pipeline.score(X_test, y_test))



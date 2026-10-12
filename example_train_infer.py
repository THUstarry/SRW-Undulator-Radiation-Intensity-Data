"""Compare linear and log POD-Linear models on the released SRW dataset.

Run ``python example_train_infer.py`` from any directory. Both models use the
paper's fixed split and fit every preprocessing step on training samples only.
The script reports interpolation and extrapolation test MRE for each model.
This is a minimal training/inference example, not a reproduction of all four
surrogate models or all paper figures. Requires NumPy, h5py, scikit-learn;
it neither generates SRW data nor writes model/prediction files.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parent
POD_ENERGY_FRACTION = 0.9999
LOG_EPS_FACTOR = 1.0e-7
MRE_EPS_FACTOR = 1.0e-4


def train_and_evaluate(
    representation: str,
    parameters: np.ndarray,
    intensity: np.ndarray,
    train_idx: np.ndarray,
    test_sets: dict[str, np.ndarray],
) -> tuple[int, float, dict[str, float]]:
    """Fit one POD-Linear model and return mean test MRE in percent."""
    train_images = intensity[train_idx]
    train_max = float(np.max(train_images))
    log_eps = LOG_EPS_FACTOR * train_max

    def to_representation(images: np.ndarray) -> np.ndarray:
        values = np.asarray(images, dtype=np.float64)
        if representation == "log":
            return np.log10(np.maximum(values, 0.0) + log_eps)
        return values

    train_rep = to_representation(train_images)
    rep_min = float(np.min(train_rep))
    rep_max = float(np.max(train_rep))
    rep_span = rep_max - rep_min
    if rep_span <= 0:
        raise ValueError("Training intensity representation has zero range.")

    train_flat = ((train_rep - rep_min) / rep_span).astype(np.float32)
    train_flat = train_flat.reshape(len(train_idx), -1).astype(np.float64)
    pod_mean = train_flat.mean(axis=0, keepdims=True)
    centered = train_flat - pod_mean
    _, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    retained_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    n_modes = int(np.searchsorted(retained_energy, POD_ENERGY_FRACTION) + 1)
    pod_modes = right_vectors[:n_modes].astype(np.float32)

    input_scaler = StandardScaler().fit(parameters[train_idx])
    train_coefficients = ((train_flat - pod_mean) @ pod_modes.T).astype(
        np.float32
    )
    coefficient_scaler = StandardScaler().fit(train_coefficients)
    features = PolynomialFeatures(degree=1, include_bias=True)
    x_train = features.fit_transform(
        input_scaler.transform(parameters[train_idx]).astype(np.float32)
    )
    model = LinearRegression().fit(
        x_train, coefficient_scaler.transform(train_coefficients)
    )

    mean_mre = {}
    for set_name, indices in test_sets.items():
        # Inference starts from held-out, unscaled physical parameters.
        truth = intensity[indices]
        x_test = features.transform(
            input_scaler.transform(parameters[indices]).astype(np.float32)
        )
        coefficients = coefficient_scaler.inverse_transform(
            model.predict(x_test)
        )
        normalized_flat = pod_mean + coefficients @ pod_modes
        predicted_rep = normalized_flat.reshape(truth.shape) * rep_span + rep_min
        if representation == "log":
            predicted = np.power(10.0, predicted_rep) - log_eps
        else:
            predicted = predicted_rep
        predicted = np.maximum(predicted, 0.0).astype(np.float32)

        # Mean over pixels for each sample, then over samples in the test set.
        denominator = truth.astype(np.float64) + MRE_EPS_FACTOR * train_max
        per_sample_mre = np.mean(
            np.abs(predicted - truth).astype(np.float64) / denominator,
            axis=(1, 2),
        )
        mean_mre[set_name] = 100.0 * float(np.mean(per_sample_mre))

    return n_modes, float(retained_energy[n_modes - 1]), mean_mre


def main() -> None:
    with h5py.File(ROOT / "srw_intensity_dataset.h5", "r") as data:
        parameters = data["parameters"][:]
        intensity = data["intensity"][:]
    with np.load(ROOT / "dataset_split_indices.npz", allow_pickle=False) as split:
        train_idx = split["train_indices"]
        test_sets = {
            "InterpTest": split["interp_test_indices"],
            "ExtrapTest": split["extrap_test_indices"],
        }

    results = {}
    for representation in ("linear", "log"):
        n_modes, energy, mre = train_and_evaluate(
            representation, parameters, intensity, train_idx, test_sets
        )
        results[representation] = mre
        print(f"{representation}: POD modes={n_modes}, retained energy={energy:.6%}")

    print(f"\nPOD-Linear MRE comparison (training samples: {len(train_idx)})")
    print(f"{'Test set':<12} {'n':>5} {'Linear (%)':>12} {'Log (%)':>12} {'Log - Linear (pp)':>18}")
    for set_name, indices in test_sets.items():
        linear = results["linear"][set_name]
        log = results["log"][set_name]
        print(
            f"{set_name:<12} {len(indices):>5} {linear:>12.3f} "
            f"{log:>12.3f} {log - linear:>18.3f}"
        )
    print("Negative Log - Linear values indicate lower MRE with log representation.")


if __name__ == "__main__":
    main()

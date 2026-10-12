"""Minimal loader for the released SRW surrogate-model dataset."""

from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_dataset():
    """Load the full physical dataset into memory."""
    with h5py.File(ROOT / "srw_intensity_dataset.h5", "r") as handle:
        return {
            "sample_id": handle["sample_id"][:],
            "parameters": handle["parameters"][:],
            "By_T": handle["By_T"][:],
            "detuning_relative": handle["detuning_relative"][:],
            "intensity": handle["intensity"][:],
            "x_um": handle["x_um"][:],
            "y_um": handle["y_um"][:],
            "n_macro_electrons": handle["n_macro_electrons"][:],
        }


def load_split_indices():
    """Return the fixed split used for all experiments in the paper."""
    with np.load(ROOT / "dataset_split_indices.npz", allow_pickle=False) as split:
        return {name: split[name] for name in split.files}


if __name__ == "__main__":
    data = load_dataset()
    split = load_split_indices()
    print("parameters:", data["parameters"].shape)
    print("intensity:", data["intensity"].shape)
    for name in (
        "train_indices",
        "validation_indices",
        "interp_test_indices",
        "extrap_test_indices",
    ):
        print(f"{name}: {split[name].size}")

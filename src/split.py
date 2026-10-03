"""Leave-one-ward-out folds: test = 1 ward, validation = next n_val wards (cyclic), train = rest."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    test_ward: int
    val_wards: tuple
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def ward_folds(ward, n_val: int = 2) -> list[Fold]:
    ward = np.asarray(ward)
    wards = sorted(int(w) for w in np.unique(ward))
    if len(wards) < n_val + 2:
        raise ValueError(f"need at least {n_val + 2} wards, got {wards}")
    folds = []
    for i, w in enumerate(wards):
        val = tuple(wards[(i + k) % len(wards)] for k in range(1, n_val + 1))
        test_idx = np.flatnonzero(ward == w)
        val_idx = np.flatnonzero(np.isin(ward, val))
        train_idx = np.flatnonzero(~np.isin(ward, (w, *val)))
        roles = [set(ward[test_idx]), set(ward[val_idx]), set(ward[train_idx])]
        assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]), "ward leak"
        folds.append(Fold(w, val, train_idx, val_idx, test_idx))
    return folds

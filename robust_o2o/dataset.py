"""Dataset boundaries for benchmark learners and corruption diagnostics."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .environment import Dataset, STANDARD_DATASET_KEYS


# ``mc_returns`` is a learning target used by Cal-QL.  It is derived from the
# rewards available in the replay dataset and is not a corruption label.
LEARNER_DATASET_KEYS = (*STANDARD_DATASET_KEYS, "mc_returns")

# These arrays may exist in the preprocessing/corruption artifact so that the
# repository can audit what happened.  They must never cross into an actor,
# critic, value loss, or replay-sampling decision.
CORRUPTION_LABEL_KEYS = frozenset(
    {
        "mc_calibration_valid",
        "corruption_mask",
        "corruption_target_label",
        "is_corrupted",
        "attack_magnitude",
        "adversarial_objective",
        "clean_observations",
        "clean_actions",
        "clean_rewards",
        "clean_next_observations",
    }
)


def learner_dataset_view(dataset: Mapping[str, np.ndarray]) -> Dataset:
    """Return the only arrays a benchmark learner is allowed to sample.

    The returned mapping deliberately omits trajectory bookkeeping and all
    corruption-derived labels.  Arrays are not copied: callers already own an
    immutable preprocessing artifact and sampling performs indexed copies into
    tensors.
    """

    missing = [key for key in STANDARD_DATASET_KEYS if key not in dataset]
    if missing:
        raise KeyError(f"learner dataset is missing transition fields: {missing}")
    return {key: dataset[key] for key in LEARNER_DATASET_KEYS if key in dataset}


def assert_no_corruption_labels(batch: Mapping[str, object]) -> None:
    leaked = sorted(CORRUPTION_LABEL_KEYS.intersection(batch))
    if leaked:
        raise RuntimeError(
            "corruption-derived labels leaked into a learner batch: "
            + ", ".join(leaked)
        )


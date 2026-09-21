"""Frozen BQ architecture and supervised fine-tuning protocol constants.

These values are copied from the already-validated BLADE-NCO / two-timescale
v2 BQ fine-tuning protocol. This module does not invent a second trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CANONICAL_M0_SHA256 = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
CANONICAL_CONCORDE_SHA256 = "38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0"

ARCHITECTURE: dict[str, Any] = {
    "identifier": "BQModel_TSP_2x192x512_L9_H12_ReLU_Softmax_ReZero",
    "dim_input_nodes": 2,
    "emb_size": 192,
    "dim_ff": 512,
    "activation_ff": "relu",
    "nb_layers_encoder": 9,
    "nb_heads": 12,
    "activation_attention": "softmax",
    "dropout": 0.0,
    "batchnorm": False,
    "problem": "tsp",
}
EXPECTED_PARAMETER_COUNT = 3_107_539
CHECKPOINT_STATE_KEY = "net"
CHECKPOINT_LOAD = "direct_strict_load_of_checkpoint['net']; no compatibility alias"

OPTIMIZER_TYPE = "Adam"
WEIGHT_DECAY = 0.0
GRADIENT_CLIPPING = None
MICROBATCH_SIZE = 64
FULL_PARAMETER_UPDATE = True
LOSS_NAME = "torch.nn.CrossEntropyLoss"
SUFFIX_LENGTH_MIN = 4
INTEGER_SCALE = 1_000_000
START_NODE = 0
DECODE_TYPE = "greedy"
BEAM_SIZE = 1
KNNS = 250
NORMALIZATION = "identity_for_unit_square_coordinates_no_additional_preprocessing"
TRAINING_INPUT = "expert_ordered_closed_tour_then_random_suffix"
INFERENCE_INPUT = "original_index_order_with_duplicate_node0_appended"
NEAR_ZERO_H_THRESHOLD = 1e-4

RELATIVE_CANONICAL_CHECKPOINT = Path("assets") / "solver" / "M0" / "tsp.best"
RELATIVE_CONCORDE_EXECUTABLE = Path("third_party") / "concorde" / "concorde"

FROZEN_TEST_DATASET_IDS = (
    "uniform100_v1",
    "gaussian_cluster100_v1",
    "tsplib100_bq72_v1",
    "tsplib_original_scale_bq83_v1",
)


def architecture_constructor_args() -> tuple[Any, ...]:
    spec = ARCHITECTURE
    return (
        spec["dim_input_nodes"],
        spec["emb_size"],
        spec["dim_ff"],
        spec["activation_ff"],
        spec["nb_layers_encoder"],
        spec["nb_heads"],
        spec["activation_attention"],
        spec["dropout"],
        spec["batchnorm"],
        spec["problem"],
    )


def default_canonical_checkpoint(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return (root / RELATIVE_CANONICAL_CHECKPOINT).resolve()


def default_concorde_executable(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return (root / RELATIVE_CONCORDE_EXECUTABLE).resolve()

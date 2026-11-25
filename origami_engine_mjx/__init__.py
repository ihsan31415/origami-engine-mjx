"""origami-engine-mjx: MJX-based origami folding simulator.

Dual-backend MuJoCo MJX engine supporting both MJX-JAX (CPU/GPU/TPU,
differentiable) and MJX-Warp (NVIDIA CUDA GPU, high-throughput batched).

Quick start
-----------
>>> from origami_engine_mjx import OrigamiEngine, load_default_creases
>>> engine = OrigamiEngine(backend="jax")
>>> creases = load_default_creases()
>>> score = engine.evaluate(creases, duration=1.0)
"""

from .constants import (
    MAX_FOLD,
    PAPER_MASS,
    THICK,
    VIB_AMP_X,
    VIB_AMP_Z,
    VIB_FREQ,
    Z0,
    M,
    S,
    V,
)
from .engine import OrigamiEngine, compute_forward_displacement
from .evaluation import evaluate_individual, evaluate_population
from .fitness import (
    FitnessBreakdown,
    FitnessWeights,
    blend_fitness,
    blend_scores,
    fold_report,
    fold_score,
    hinge_fold_error,
    hinge_qpos,
)
from .geometry import build_pattern
from .model import generate_xml
from .patterns import (
    DEFAULT_PATTERN_JSON,
    load_default_creases,
    reference_miura_legacy_creases,
    reference_miura_render_creases,
)
from .plotting import plot_pattern
from .serialization import CREASE_SCHEMA, load_creases_from_json, normalize_creases, save_creases_to_json
from .types import BackendName, Crease, Pattern, PopulationResult, SimulationResult

try:  # the installed distribution is authoritative ...
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("origami-engine-mjx")
except Exception:  # noqa: BLE001 - ... but a source checkout has no metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "CREASE_SCHEMA",
    "DEFAULT_PATTERN_JSON",
    "MAX_FOLD",
    "PAPER_MASS",
    "THICK",
    "VIB_AMP_X",
    "VIB_AMP_Z",
    "VIB_FREQ",
    "Z0",
    "BackendName",
    "Crease",
    "FitnessBreakdown",
    "FitnessWeights",
    "M",
    "OrigamiEngine",
    "Pattern",
    "PopulationResult",
    "S",
    "SimulationResult",
    "V",
    "blend_fitness",
    "blend_scores",
    "build_pattern",
    "compute_forward_displacement",
    "evaluate_individual",
    "evaluate_population",
    "fold_report",
    "fold_score",
    "generate_xml",
    "hinge_fold_error",
    "hinge_qpos",
    "load_creases_from_json",
    "load_default_creases",
    "normalize_creases",
    "plot_pattern",
    "reference_miura_legacy_creases",
    "reference_miura_render_creases",
    "save_creases_to_json",
]

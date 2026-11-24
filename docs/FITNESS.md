# Fitness: movement, folding and their blend

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [Fitness](FITNESS.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

The engine has always scored a pattern by **movement**: how far the paper's
centre travels while it folds. Some questions want a different signal — *did the
paper actually fold to the angles it was designed for?* The
`origami_engine_mjx.fitness` module answers that, and lets you mix both signals
with any percentage split you like.

Everything here is plain NumPy on a finished `SimulationResult`; it needs no
algorithm and no extra dependency. The default (`fold_weight=0`) is exactly the
historical movement score, so existing code keeps its numbers.

## The two components

| Component | Question | Range | Source |
|---|---|---|---|
| **movement** | How far did the sheet travel? | metres (can be negative) | `compute_forward_displacement` |
| **fold** | Did each hinge reach its target angle? | `0..1` | `fold_score` |

### `fold_score(result, actuated=None, percent=False, zero_tolerance_deg=2.0) -> float`

A single number in `[0, 1]` (`0..100` with `percent=True`). Each hinge
contributes linear progress from `0` while it is still flat, to `1` once it is
within `zero_tolerance_deg` of its target angle. Overshooting is never
rewarded: a hinge folded past its target loses progress again.

```python
from origami_engine_mjx import OrigamiEngine, fold_score, load_default_creases

engine = OrigamiEngine(backend="jax")
creases = load_default_creases()

# Short durations keep a CPU run quick; scale them up on a GPU.
result = engine.run(creases, duration=0.02)
print(fold_score(result, percent=True))  # 0..100  (percent of the fold done)
```

`actuated=None` scores **every** hinge, including passive ones: a hinge you
chose not to drive still has a target the design wants, so it still counts.

### `fold_report(result, actuated=None, percent=False, zero_tolerance_deg=2.0) -> dict`

`fold_score` with the detail you need for plots and logs:

| Key | Meaning |
|---|---|
| `score` | The number above. |
| `mean_abs_error_deg`, `max_abs_error_deg`, `rms_error_deg` | Hinge error statistics (degrees). |
| `n_hinges` | Hinges scored. |
| `per_hinge_error_deg` | Signed `achieved − target` per hinge (degrees). |
| `achieved_deg`, `target_deg` | Achieved and target angle of **every** hinge (degrees). |

### `hinge_fold_error(result, actuated=None) -> np.ndarray` and `hinge_qpos(result)`

Raw building blocks: signed errors in radians, and the achieved hinge angles in
`info["hinges"]` order.

## Blending movement and folding

`FitnessWeights(movement=..., fold=..., movement_scale=1.0)` holds the split.
Weights are relative, so `movement=70, fold=30` and `movement=0.7, fold=0.3` are
the same: 70 % movement, 30 % folding.

```python
from origami_engine_mjx import FitnessWeights, blend_fitness

weights = FitnessWeights(movement=70, fold=30)
breakdown = blend_fitness(result, weights=weights)
print(breakdown.combined)  # the blended score
print(breakdown.movement, breakdown.fold)  # the two components
print(breakdown.movement_fraction, breakdown.fold_fraction)  # 0.7, 0.3
breakdown.as_dict()  # flat dict for logs / JSON
```

- **`movement_scale`** puts movement on the same `0..1`-ish scale as folding.
  The default `1.0` means "metres per sheet side" (the sheet is 1 m), so a
  0.2 m push scores `0.2`. If your patterns travel much further, raise it.
- **`fold_weight=0`** (the default) returns the movement score unchanged.
- A blended score of `0.7·movement + 0.3·fold` is often in the same numerical
  ballpark generation to generation, which keeps ES/CMA step sizes sensible.

`blend_scores(movement, fold, movement_weight=1.0, fold_weight=0.0, movement_scale=1.0)`
is the pure-number version when you already have both components.

## Where the weights plug in

| Entry point | How to set the split |
|---|---|
| `OrigamiEngine.evaluate(..., fold_weight=0.3, movement_weight=1.0)` | Returns the blended score; `fold_weight=0` is the old displacement. |
| `OrigamiEngine.evaluate_population(..., fold_weight=0.3)` | `scores` are blended; `extra["movement_scores"]` / `extra["fold_scores"]` hold the parts. |
| `evaluate_individual(..., fold_weight=0.3, return_details=True)` | `details` gains `movement`, `fold`, `combined`, `fold_report`. |
| `evaluation.evaluate_population(..., fold_weight=0.3)` | Blended scores, same `error_policy` handling as before. |
| `OrigamiEvaluator(space, movement_weight=70, fold_weight=30)` | Every `EvalRecord` carries the blend; descriptors gain `fold_score` and `fold_mean_abs_error_deg`. |

```python
from origami_engine_mjx import load_default_creases
from origami_engine_mjx.integrations import DesignSpace, OrigamiEvaluator

space = DesignSpace.angles_only(load_default_creases(), name="miura-2x2")
evaluator = OrigamiEvaluator(space, backend="jax", movement_weight=70, fold_weight=30)
record = evaluator.evaluate(space.base_vector())
print(record.fitness, record.descriptors["fold_score"])
```

> **Why the evaluator switches paths:** the fast batch path only returns
> movement, so a positive `fold_weight` makes `OrigamiEvaluator` score
> individual-by-individual. That is exactly how the backends already evaluate
> anyway, so results match one-by-one evaluation; only the bookkeeping differs.

## Which hinges are actuated

`actuated=` chooses which hinges the position actuators drive. `None` (default)
drives them all; a bool drives all or none; a list of bools (one per hinge, in
`result.info["hinges"]` order) picks individual hinges.

```python
# Drive only the first two hinges; let the others fold passively.
engine.run(creases, duration=0.02, actuated=[True, True] + [False] * 13)
```

Passive hinges are **not braked**: they keep their target angle for scoring but
are not forced to zero, so they follow the forces their neighbours transmit
(true in both the JAX and Warp paths; an off actuator simply tracks the current
hinge angle).

### Changing actuation during the run

`actuation_schedule=f(time_seconds) -> actuated` is evaluated at every physics
step (it may return a bool for all hinges or a per-hinge list):

```python
from origami_engine_mjx import OrigamiEngine, load_default_creases

engine = OrigamiEngine(backend="jax")
creases = load_default_creases()

# Drive every hinge for the first half of the run, then let the paper relax.
engine.run(creases, duration=0.02, actuation_schedule=lambda t: t < 0.01)
```

With `record=True` the per-frame trajectory stores an `"actuated"` mask (one
row per recorded frame) so a replay or video knows which hinges were driven.
The final mask is reported in `result.info["actuated"]`, and
`result.info["hinge_qpos"]` always holds the achieved angles.

## Practical tips

1. **Start with movement only.** It is fast and it is what the library has
   always done; add folding when you care about the folded shape.
2. **Pick a split and keep it** for a whole optimization run. Changing the
   weights mid-run changes the objective and confuses the optimizer.
3. **Look at `mean_abs_error_deg`** when the fold score is low: it tells you
   whether the paper is barely folding (large error on every hinge) or folding
   some hinges too far (a few big errors).
4. **Score only the driven hinges** with `fold_score(result, actuated=...)` if
   you want to know whether your actuation *strategy* worked, rather than
   whether the passive part of the pattern followed along.

## See also

- [EVALUATION.md](EVALUATION.md) — movement fitness, populations and logs
- [SIMULATION.md](SIMULATION.md) — `actuated`, schedules and trajectory recording
- [INTEGRATIONS.md](INTEGRATIONS.md) — `OrigamiEvaluator`, ES/RL adapters
- [API_REFERENCE.md](API_REFERENCE.md) — exact signatures
- [`notebook_examples/05_evosax_switchable.ipynb`](../notebook_examples/05_evosax_switchable.ipynb) — a runnable example

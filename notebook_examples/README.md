# Notebook examples

Runnable, self-contained [Google Colab](https://colab.research.google.com)
notebooks that connect **external** optimizers, reinforcement-learning
environments and AI agents to `origami-engine-mjx`.

The library is a substrate: it supplies the search space, the simulator, the
objective and the checkpoint format. It ships **no** ES, RL or agent algorithm —
every algorithm in these notebooks comes from a third-party package or from
notebook code you can edit.

| Notebook | What it teaches | External packages |
|---|---|---|
| [01 · Black-box quickstart](01_black_box_quickstart.ipynb) | `nevergrad`, CMA-ES and `pymoo` against the same `genome -> loss` interface; seeding, logging, checkpoints | `nevergrad`, `cma`, `pymoo` |
| [02 · `evosax`, all algorithms](02_evosax_all_algorithms.ipynb) | one generic `ask -> evaluate -> tell` driver for **every** `evosax` strategy, including the three `init` variants | `evosax>=0.3` |
| [03 · RL: Gymnasium + pure JAX](03_rl_gymnasium.ipynb) | `OrigamiEnv`, `check_env`, action/reward contract, jitted `step` + `lax.scan`, optional Stable-Baselines3 PPO | `gymnasium`; optional `stable-baselines3` |
| [04 · AI / LLM agent loop](04_llm_agent_loop.ipynb) | `propose -> evaluate -> summarize` loop with a deterministic mock proposer and an optional OpenAI-compatible model | none (stdlib client); optional API key |

## Open in Colab

The links target the integration branch. After the branch is merged, replace
`feat/optimization-integrations-v1` with `main`.

* [01 · Black-box quickstart](https://colab.research.google.com/github/ihsan31415/origami-engine-mjx/blob/feat/optimization-integrations-v1/notebook_examples/01_black_box_quickstart.ipynb)
* [02 · `evosax`, all algorithms](https://colab.research.google.com/github/ihsan31415/origami-engine-mjx/blob/feat/optimization-integrations-v1/notebook_examples/02_evosax_all_algorithms.ipynb)
* [03 · RL: Gymnasium + pure JAX](https://colab.research.google.com/github/ihsan31415/origami-engine-mjx/blob/feat/optimization-integrations-v1/notebook_examples/03_rl_gymnasium.ipynb)
* [04 · AI / LLM agent loop](https://colab.research.google.com/github/ihsan31415/origami-engine-mjx/blob/feat/optimization-integrations-v1/notebook_examples/04_llm_agent_loop.ipynb)

Each notebook starts with its own install cell:

```python
%pip install -q "origami-engine-mjx[jax] @ git+https://github.com/ihsan31415/origami-engine-mjx.git@feat/optimization-integrations-v1"
```

Notebook 03 additionally installs `[rl]`; notebook 02 installs `evosax`.

## CPU vs GPU: what is actually fast

Every **physics** evaluation is one full MJX/MuJoCo simulation. On a CPU
runtime the cheap 6-crease demo still costs seconds per evaluation, and the
4×4 reference pattern costs much more. The notebooks are therefore honest about
where their time goes:

* **Surrogate defaults** — notebooks 01, 02 and 04 run their optimizer/agent
  loops on an analytic stand-in with the *same* `genome -> loss` contract, then
  re-score the winner with the real `OrigamiEvaluator`. Set
  `OBJECTIVE_MODE = "physics"` to optimize the real forward displacement; do
  that on a GPU runtime with a small budget.
* **Real physics where it matters** — every notebook still builds the real
  evaluator or environment and runs at least one real simulation, so the wiring
  you copy is exercised end to end.
* **RL** — notebook 03's `OrigamiEnv` is always real physics. `RUN_VECTOR` and
  `RUN_SB3` are off by default because the jitted/vmapped step and a trainer
  multiply the cost.
* **`evosax` itself is cheap** — its JAX strategies are milliseconds; the
  simulation dominates by orders of magnitude. Notebook 02 can run all 37
  strategies in surrogate mode on CPU.
* **No batching claims** — `OrigamiEvaluator.evaluate_batch` still loops per
  individual today; it is a bookkeeping convenience, not a CPU speed-up.

For real searches, switch the Colab runtime to **GPU** and use the packaged
4×4 Miura pattern (`USE_DEFAULT_PATTERN = True` where offered).

## Suggested order

1. **01** — learn the space/evaluator/logging/checkpoint contract and the
   minimize-vs-maximize direction.
2. **02** — reuse that contract with every `evosax` strategy; compare searches.
3. **03** — switch from a black-box objective to the step-by-step RL API.
4. **04** — replace the optimizer with a proposer (mock or LLM) and keep the
   same validation, logging and checkpointing.

## Integration recipe

The reusable, algorithm-agnostic checklist is
[`prompts/origami-optimizer-integration.md`](../prompts/origami-optimizer-integration.md).
It covers space selection, the evaluator contract, adapters, seeding, logging,
checkpoints and a verification checklist. Use it with any optimizer, RL trainer
or agent framework — no algorithm code belongs in the library.

## Verifying the notebooks locally

The notebooks are plain `nbformat` 4 JSON, so you can review or diff them like
any source file. To execute one headlessly:

```bash
pip install jupyter nbclient
jupyter nbconvert --to notebook --execute notebook_examples/01_black_box_quickstart.ipynb \
  --output /tmp/01.executed.ipynb
```

The `%pip` cells install from the branch; remove them (or install the package
first) when running inside the repository checkout. The JAX/MJX import prints a
harmless `Failed to import warp` line on machines without `warp-lang`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Failed to import warp` on import | Expected without `warp-lang`; the notebooks use the MJX-JAX backend |
| `ModuleNotFoundError: jax` / `mujoco_mjx` | Rerun the notebook's install cell, then restart the runtime |
| `evosax` API changes | Notebook 02 needs `evosax>=0.3` (`ask`/`tell` API); pin the version if needed |
| `OPENAI_API_KEY` missing | Notebook 04 defaults to `AGENT_MODE="mock"` and needs no key |
| Long first cell | JAX JIT compilation of the contact model dominates; later steps reuse it |
| Out of memory in notebook 03 | Lower `MAX_STEPS`, `N_ENVS`, or keep `RUN_VECTOR=False` |

## Related documentation

* [Optimization (ES)](../docs/OPTIMIZATION_ES.md) — adapter and direction table
* [Search spaces](../docs/SEARCH_SPACE.md) — `SearchSpace` vs `DesignSpace`
* [Populations](../docs/POPULATIONS.md) — seeding, diversity, checkpoints
* [Reinforcement learning](../docs/RL.md) — environment API and reward contract
* [Integrations overview](../docs/INTEGRATIONS.md) — what is and is not in the library
* [Colab + GPU notes](../docs/COLAB_GPU.md) — runtime and install guidance

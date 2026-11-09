# Knowing the Source Is Not Enough: Repetition and Provenance in LLM Blame Attribution for Multi-Agent Teams

Apart Research Swarm Alignment Hackathon, November 2026.

When an LLM auditor assigns responsibility for a multi-agent failure, does it track who caused the failure, or who talked
about it most? We test this with frozen-transcript perturbations of real Colosseum multi-agent runs (E1) and scripted
fault-relay incidents (E2), audited by gpt-4o-mini and gpt-5.4-mini, plus a pre-registered provenance-prompt mitigation.

The original Colosseum README is kept as [`COLOSSEUM_README.md`](COLOSSEUM_README.md).

## Pre-existing vs new

**Pre-existing (not our work):**
- Colosseum, [umass-ai-safety/colosseum](https://github.com/umass-ai-safety/colosseum) (MIT; see `LICENSE`):
  everything in this repository's history up to commit `26ad60b`, including `experiments/collusion/run.py` and
  `experiments/collusion/compute_jira_optimal.py`.
- Terrarium (`terrarium-agents==0.2.1`), installed as a dependency.
- CoLLAB, the `external/CoLLAB` git submodule.

**New this weekend (Nov 6-9 2026):**
- `experiments/collusion/blame.py`: counterfactual / Shapley blame for the Colosseum Jira task.
- Everything under `hackathon/`:
  - `run_scripted.py`, `auditors.py`, `analyze.py`, `plots.py`, `summarize_blame.py`, `modal_vllm.py`, `*.yaml`: Colosseum
    sweeps, scripted saboteurs and the first auditors (exploratory phase).
  - `analysis/exposure.py` (E1), `analysis/analyze_exposure.py`, `analysis/relay.py` (E2), `analysis/labels.py`,
    `ties.py`, `solver_test.py`, `baselines.py`.
  - `analysis/final_common.py`, `analysis/final_run.py`, `analysis/final_analysis.py`, `analysis/paper_outputs.py`.
  - `PREREG.md`, `final/manifest.json` (+ `manifest.sha256`), `final/model_configs.json`, `final/examples/`.
- `results/final/`: the raw confirmatory records, analysis output, figures and `paper_numbers.md`.

## Reproduce every paper number (no API key needed)

    uv sync                                    # or: python -m venv .venv && pip install -e . matplotlib scipy
    .venv/bin/python hackathon/analysis/final_analysis.py --json results/final/analysis.json > results/final/analysis.txt
    .venv/bin/python hackathon/analysis/paper_outputs.py

`paper_outputs.py` reads `results/final/records.jsonl` and writes `results/final/paper_numbers.md` (the single source of
truth for every number in the paper) and the figures in `results/final/figures/` (PDF, 300 dpi PNG, and the plotted data as
CSV). The Colosseum appendix facts (runs per sweep, solver check, tied optima, talkativeness baselines) are in
`results/final/colosseum/`. Recomputing those, or re-running E1, needs the Colosseum run directories under
`experiments/collusion/outputs/`. They are not in git: they hold model configs, and are distributed separately as a zip with
keys redacted.

## Re-run the experiments (needs an OpenAI API key)

    export AUDITOR_BASE_URL=https://api.openai.com/v1 AUDITOR_API_KEY=sk-...
    .venv/bin/python hackathon/analysis/final_run.py run --run A0 --dry-run   # checks every prompt against the manifest
    for r in A0 A B C; do .venv/bin/python hackathon/analysis/final_run.py run --run $r; done

- 2,888 calls, 1,444 per model, defined by `hackathon/final/manifest.json`. Cost about $2.16 (Nov 2026 prices).
- `gpt-4o-mini` (returned snapshot `gpt-4o-mini-2024-07-18`): temperature 0, max_tokens 400.
- `gpt-5.4-mini-2026-03-17`, requested as `gpt-5.4-mini` (Amendment 2): reasoning_effort none, max_completion_tokens
  1000. Every response must report the pinned snapshot, or the run stops.
- The runner refuses to call if the code no longer renders the frozen prompt hashes, if the manifest hash does not
  match, or if `hackathon/` has uncommitted changes. Records append to `results/final/records.jsonl` and resume safely.

## Pre-registration

- `hackathon/PREREG.md`: primary and secondary contrasts, guardrails and interpretation rules.
- Freeze commit `b07611d`. Amendment 1 (interpretation rules: COR ceiling, mitigation guardrails, origin definition, NE
  wording) is commit `9a216a3`, made before any successful paid call and before Run C. Amendment 2 (model request
  name) is commit `12479de`.
- Every confirmatory record was produced by commit `68d2636` with manifest SHA-256
  `7db0fc27f1be42d1d00348f758299065981ec89ea5722ef65509fb0f17a99d8e`.
- Earlier gpt-4o-mini results (E1 and the first relay conditions) were inspected before pre-registration and are
  exploratory. Run A0 re-runs them under the frozen code.
- Timestamps come from local git history and the run log (`results/final/runall.log`). The commits were pushed to
  GitHub only after the runs, so they are not externally timestamped.

## Data

`results/final/records.jsonl` holds one record per auditor call. Each record has the request metadata, the raw auditor
response, parsed scores, token usage and hashes. All agents are synthetic (LLM or scripted), and the files contain no
personal data.

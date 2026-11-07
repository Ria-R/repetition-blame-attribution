# Blame attribution under repetition (Apart Swarm Alignment Hackathon, Nov 6-8 2026)

**Pre-existing code (not ours):** Colosseum / Terrarium / CoLLAB (everything outside `hackathon/` and
`experiments/collusion/blame.py`), including `experiments/collusion/compute_jira_optimal.py` and `run.py`.

**New weekend work:** `experiments/collusion/blame.py` and everything under `hackathon/`:
- `run_scripted.py`, `auditors.py`, `analyze.py`, `plots.py`, `summarize_blame.py`, `modal_vllm.py`, `*.yaml`: Colosseum Jira
  sweeps, scripted saboteurs, counterfactual (Shapley) blame and the first auditors (pilot / exploratory phase).
- `analysis/exposure.py` (E1, frozen-transcript exposure), `analysis/analyze_exposure.py`, `analysis/labels.py`, `ties.py`,
  `solver_test.py`, `baselines.py`.
- `analysis/relay.py` (E2, scripted relay incidents).
- `analysis/final_common.py`, `analysis/final_run.py`, `final/manifest.json`, `PREREG.md`: the frozen, pre-registered final runs.

## Provenance of results

- **Exploratory (inspected before pre-registration):** E1 gpt-4o-mini, 652 calls (`results/exposure/calls_gpt-4o-mini.jsonl`);
  E2 gpt-4o-mini, 360 calls over local / relay_min / relay_verbose / relay_uncited / relay_plain / relay_filler
  (`results/relay/relay_gpt-4o-mini.jsonl`). All 360 relay calls were made after the ordering fix that places every Coordinator
  line before the Analyst's "Additional signal" line (checked in the dry-run example logs before each live run; see the
  repository history). Those records predate versioning and carry no `script_version` field.
- **Confirmatory (pre-registered in `PREREG.md`):** Runs A0, A, B, C defined by `final/manifest.json`, written to
  `results/final/records.jsonl` with model IDs, API settings, git commit, manifest and prompt SHA-256, script version
  (`exposure-v2` / `relay-v2`), raw responses including failures, finish reason, token usage and parse status.
  `relay.build()` now asserts the relay-before-counter-evidence ordering for every generated log.

## Reproduce

    .venv/bin/python hackathon/analysis/final_run.py manifest          # rebuild + counts (must reproduce manifest.sha256)
    .venv/bin/python hackathon/analysis/final_run.py run --run A0 --dry-run
    AUDITOR_BASE_URL=https://api.openai.com/v1 AUDITOR_API_KEY=... .venv/bin/python hackathon/analysis/final_run.py run --run A0

E1 needs the Colosseum run outputs under `experiments/collusion/outputs/` (not in git; shipped in the colosseum-blame zip).

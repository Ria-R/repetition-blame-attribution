# Saturday-evening analysis scripts (new weekend work)

Put these in `hackathon/analysis/` in the colosseum repo. All scripts take the folder holding the run dirs
(they search it recursively for `final_summary.json`), so point them at `outputs/` or at the unzipped
`colosseum-blame/` folder. Run from the repo root so `experiments/collusion/compute_jira_optimal.py` is importable.

| Script | What it does | Paid? |
|---|---|---|
| `solver_test.py <data>` | Brute-forces every 4-agent run and checks the Hungarian optimum (result: 112/112 match) | no |
| `ties.py <data>` | Counts runs with tied optima and how much Shapley changes across them (66/112 tied; top-1 flips in 8) | no |
| `baselines.py <data>` | Top-1 colluder hit rate: chance, most/least talkative, Shapley, auditors A/B/C | no |
| `labels.py <data> <out.csv>` | One row per agent: role, attempted action, claim marginal, tie-averaged regret allocation, message count | no |
| `exposure.py` | Experiment 1: frozen-transcript exposure | yes, ~$0.35 per view pair on gpt-4o-mini |

## Experiment 1: run this

Dry run first (writes `example_prompt.txt`, prints call count):

    python hackathon/analysis/exposure.py --data outputs --colosseum . --out results/exposure --views A,B --dry-run

Expect `episodes=20 (10 with colluders) calls=652`. Then the live run:

    export AUDITOR_BASE_URL=https://api.openai.com/v1
    export AUDITOR_API_KEY=$OPENAI_API_KEY
    export AUDITOR_MODEL=gpt-4o-mini
    python hackathon/analysis/exposure.py --data outputs --colosseum . --out results/exposure --views A,B

Results go to `results/exposure/calls_gpt-4o-mini.jsonl`, one line per call. It resumes if interrupted.
Send that file back for analysis.

Design: per episode, every agent takes a turn as the target. Message type `original` (copies of the target's
own earlier messages) or `neutral` (task-neutral filler; the negative control). +3 or +11 added messages at
seeded random positions, plus one shared baseline per episode and view. Outcome and all other messages fixed.
Silent agents get only the neutral condition.

Optional Qwen-7B auditor: redeploy the Modal vLLM endpoint, then set AUDITOR_BASE_URL to it and
AUDITOR_MODEL=Qwen/Qwen2.5-7B-Instruct.

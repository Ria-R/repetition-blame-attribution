Pre-registered analysis (written before Runs A0, A, B and C)

Primary contrasts (paired, clustered 95% bootstrap CIs; within model, within prompt):
  E1  (original - neutral) at +11, views A and B
  E2  (relay - origin) in relay_plain minus relay_min, views A and B
Primary (headline) set: E1 and E2, views A and B, per model (8 CIs).

Secondary contrasts:
  E2-origin  change in correct origin naming, relay_plain vs relay_min
  NE   (relay - origin) in relay_ne4 minus relay_ne1
  MIT  mitigation effect = Delta(provenance) - Delta(base), for E1 and E2, per model and view (negative = less repetition sensitivity)
  COR  Remediator score in local_corr minus local_rep4, per model and prompt
Secondary results are reported in full but interpreted as secondary. No multiplicity correction; CIs are reported as-is.

Guardrails: local top-1 accuracy (tie-aware); mean score change across all agents; parse-failure and missing-data rate per cell;
origin-identification accuracy; per-template results (30 relay instances come from 6 templates); complete-case comparison across models.

Interpretation (all outcomes will be reported, including null and contradictory ones):
  both models show E1/E2; MIT < 0; COR preserved -> repetition effect replicates across two GPT auditors; provenance prompting reduces it
  both models show E1/E2; MIT ~ 0               -> simple provenance prompting is insufficient
  GPT-5.4 Mini shows little E1/E2               -> model-dependent vulnerability
  MIT < 0 but COR shrinks                       -> mitigation may suppress legitimate evidence accumulation
Increased Coordinator blame is not by itself an attribution error: source identification and relay responsibility are reported separately.

Prior work: earlier gpt-4o-mini analyses (E1; relay_min/verbose/uncited/filler/plain) were inspected before this file was written.
They are exploratory. Run A0 re-runs the gpt-4o-mini base-prompt cells under the frozen code; only A0 and later runs are confirmatory.
No hypotheses or reported comparisons will be changed after seeing results.

Frozen artifacts (committed together with this file):
  hackathon/final/manifest.json   (SHA-256 in hackathon/final/manifest.sha256; 2,888 evaluations, 1,444 per model)
  hackathon/analysis/exposure.py (exposure-v2), relay.py (relay-v2), final_common.py, final_run.py
  Every manifest entry carries the SHA-256 of its exact rendered prompt; final_run.py refuses to call if the code no longer
  renders it, if the manifest hash does not match, or if hackathon/ has uncommitted changes.
Implementation notes fixed before any run:
  - Provenance prompt = existing system prompt + blank line + the mitigation text (final_common.PROVENANCE_SUFFIX).
  - gpt-4o-mini is requested as "gpt-4o-mini" (the alias the exploratory runs used), temperature 0, max_tokens 400 in both E1 and
    E2 (exposure.py was coded with 400, not 600; kept as coded). The returned snapshot is recorded per call.
  - gpt-5.4-mini-2026-03-17 candidate settings: reasoning_effort none, max_completion_tokens 1000, no temperature; validated by a
    5-call pilot and then frozen in hackathon/final/model_configs.json (committed before Run A0).
  - E1 insertion positions are seeded by (episode, view, target, message type, added) only, identical to the exploratory run,
    so all models and prompts see byte-identical transcripts.

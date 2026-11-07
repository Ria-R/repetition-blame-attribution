"""Driver for the final pre-registered runs (new weekend work, Apart Swarm Alignment Hackathon).

  python hackathon/analysis/final_run.py manifest              # build hackathon/final/manifest.json (+ .sha256), print counts/cost
  python hackathon/analysis/final_run.py examples              # write view-B example prompts for the new relay conditions
  python hackathon/analysis/final_run.py pilot                 # 5 gpt-5.4-mini calls (paid) -> results/final/pilot.jsonl
  python hackathon/analysis/final_run.py run --run A0 [--dry-run]   # execute one run of the manifest (paid unless --dry-run)

Every call is defined by the frozen manifest. Before a paid call the driver re-renders the prompt from the code and
checks it against the manifest's prompt_sha256, checks the manifest file against hackathon/final/manifest.sha256, and
records the git commit (refusing to run with uncommitted changes under hackathon/). Records append to
results/final/records.jsonl; completed (ok) ids are skipped on resume, failures are kept and retried on resume.
Env: AUDITOR_BASE_URL, AUDITOR_API_KEY.
"""
from __future__ import annotations

import argparse, collections, json, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "collusion"))
import exposure  # noqa: E402
import relay  # noqa: E402
from final_common import (CANDIDATE_SETTINGS, PROVENANCE_SUFFIX, REQUEST_NAME, REQUIRED_RETURNED,  # noqa: E402
                          call_api, messages_sha, sha256_text)

FINAL = REPO / "hackathon" / "final"
MANIFEST = FINAL / "manifest.json"
MANIFEST_SHA = FINAL / "manifest.sha256"
CONFIGS = FINAL / "model_configs.json"
OUT = REPO / "results" / "final"
DATA = REPO / "experiments" / "collusion" / "outputs"

M4O, M54 = "gpt-4o-mini", "gpt-5.4-mini-2026-03-17"
MODELS = (M4O, M54)
# $/1M tokens (input, output). gpt-5.4-mini from public price listings (pricepertoken.com, Mar 2026).
PRICES = {M4O: (0.15, 0.60), M54: (0.75, 4.50)}
EXPECTED_OUT_TOKENS = {M4O: 85, M54: 120}  # gpt-4o-mini measured ~83/call in the exploratory E1 run

E2_A0 = ("local", "relay_min", "relay_filler", "relay_plain")
E2_B = ("relay_ne1", "relay_ne4")
E2_C_PROV = ("local", "relay_min", "relay_plain", "local_rep4", "local_corr")
E2_C_BASE = ("local_rep4", "local_corr")
VIEWS = ("A", "B")
N_INST = 30


# ---------------------------------------------------------------- unit builders (prompt rendering from code)
_eps_cache = {}


def e1_episodes():
    if "eps" not in _eps_cache:
        import compute_jira_optimal as m
        pick = exposure.pick_episodes(DATA, 20)
        _eps_cache["eps"] = [exposure.load_episode(d, m) for d in pick]
        _eps_cache["dirs"] = pick
    return _eps_cache["eps"], _eps_cache["dirs"]


def e1_messages(ent):
    eps, _ = e1_episodes()
    ep = eps[ent["episode"]]
    mtype = {"baseline": "base", "original+11": "original", "neutral+11": "neutral"}[ent["condition"]]
    added = 0 if mtype == "base" else 11
    evs = exposure.perturb(ep, ent["target"], mtype, added,
                           seed=exposure.legacy_seed(ent["episode"], ent["view"], ent["target"], mtype, added))
    return exposure.prompt(ep, evs, ent["view"], ent["prompt"])


def e2_messages(ent):
    return relay.build(ent["instance"], ent["condition"], ent["view"], ent["prompt"])["messages"]


def messages_for(ent):
    return e1_messages(ent) if ent["experiment"] == "E1" else e2_messages(ent)


# ---------------------------------------------------------------- manifest
def output_id(exp, model, prompt, view, unit, cond):
    return f"{exp}|{model}|{prompt}|{view}|{unit}|{cond}"


def e1_entries(run, model, prompt):
    eps, dirs = e1_episodes()
    out = []
    for ei, ep in enumerate(eps):
        rel = str(dirs[ei].relative_to(DATA))
        eligible = [t for t in ep["agents"] if any(e["agent"] == t for e in ep["events"])]
        for view in VIEWS:
            out.append(dict(run=run, experiment="E1", model=model, prompt=prompt, view=view, episode=ei, run_dir=rel,
                            target=None, condition="baseline",
                            id=output_id("E1", model, prompt, view, f"ep{ei}", "baseline")))
            for t in eligible:
                for cond in ("original+11", "neutral+11"):
                    out.append(dict(run=run, experiment="E1", model=model, prompt=prompt, view=view, episode=ei,
                                    run_dir=rel, target=t, target_is_colluder=t in ep["colluders"], condition=cond,
                                    id=output_id("E1", model, prompt, view, f"ep{ei}:{t}", cond)))
    return out


def e2_entries(run, model, prompt, conds):
    return [dict(run=run, experiment="E2", model=model, prompt=prompt, view=view, instance=i,
                 template=i % len(relay.INCIDENTS), condition=c,
                 id=output_id("E2", model, prompt, view, f"inst{i}", c))
            for c in conds for i in range(N_INST) for view in VIEWS]


def build_manifest():
    ents = []
    ents += e1_entries("A0", M4O, "base") + e2_entries("A0", M4O, "base", E2_A0)
    ents += e1_entries("A", M54, "base") + e2_entries("A", M54, "base", E2_A0)
    for m in MODELS:
        ents += e2_entries("B", m, "base", E2_B)
    for m in MODELS:
        ents += e1_entries("C", m, "provenance") + e2_entries("C", m, "provenance", E2_C_PROV)
        ents += e2_entries("C", m, "base", E2_C_BASE)
    ids = [e["id"] for e in ents]
    assert len(ids) == len(set(ids)), "duplicate output ids"
    for e in ents:
        msgs = messages_for(e)
        e["prompt_sha256"] = messages_sha(msgs)
        e["system_sha256"] = sha256_text(msgs[0]["content"])
        e["input_chars"] = sum(len(x["content"]) for x in msgs)
    code = {p: sha256_text((HERE / p).read_text()) for p in ("exposure.py", "relay.py", "final_common.py", "final_run.py")}
    man = dict(
        description="Frozen evaluation manifest for the final pre-registered runs (A0, A, B, C).",
        models={M4O: "requested as the gpt-4o-mini alias, as in the exploratory runs; returned snapshot is recorded per call",
                M54: "exact snapshot; requested as gpt-5.4-mini (project allowlist), every response must report this snapshot"},
        system_prompt_sha256={f"{exp}:{pv}": sha256_text(msg) for exp, base in (("E1", exposure.SYSTEM), ("E2", relay.SYSTEM))
                              for pv, msg in (("base", base), ("provenance", base + "\n\n" + PROVENANCE_SUFFIX))},
        code_sha256_at_manifest_build=code,
        script_versions={"E1": exposure.SCRIPT_VERSION, "E2": relay.SCRIPT_VERSION},
        candidate_settings=CANDIDATE_SETTINGS,
        request_names=REQUEST_NAME, required_returned_snapshot=REQUIRED_RETURNED,
        n_evaluations=len(ents),
        evaluations=ents,
    )
    FINAL.mkdir(parents=True, exist_ok=True)
    text = json.dumps(man, indent=1, sort_keys=True) + "\n"
    MANIFEST.write_text(text)
    MANIFEST_SHA.write_text(sha256_text(text) + "  manifest.json\n")
    return man


def report(ents):
    def cost(group):
        tot = 0.0
        for e in group:
            pin, pout = PRICES[e["model"]]
            tot += e["input_chars"] / 4 / 1e6 * pin + EXPECTED_OUT_TOKENS[e["model"]] / 1e6 * pout
        return tot
    c = collections.Counter((e["run"], e["experiment"], e["model"], e["prompt"], e["view"], e["condition"]) for e in ents)
    print(f"{'run':3s} {'exp':3s} {'model':24s} {'prompt':10s} {'view':4s} {'condition':12s} calls")
    for k in sorted(c):
        print(f"{k[0]:3s} {k[1]:3s} {k[2]:24s} {k[3]:10s} {k[4]:4s} {k[5]:12s} {c[k]}")
    print()
    for key, name in ((lambda e: e["run"], "run"), (lambda e: e["model"], "model"),
                      (lambda e: (e["model"], e["experiment"]), "model x experiment")):
        g = collections.defaultdict(list)
        for e in ents:
            g[key(e)].append(e)
        for k in sorted(g, key=str):
            tok = sum(e["input_chars"] for e in g[k]) // 4
            print(f"  by {name:18s} {str(k):45s} calls={len(g[k]):5d}  ~input tokens={tok:>10,}  ~cost=${cost(g[k]):.2f}")
    print(f"\nTOTAL calls={len(ents)}  ~cost=${cost(ents):.2f}  (input ~chars/4; output assumed {EXPECTED_OUT_TOKENS} tokens/call)")


# ---------------------------------------------------------------- execution
def git_state():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "hackathon"], cwd=REPO, capture_output=True, text=True).stdout
    dirty = [l for l in dirty.splitlines() if not l.endswith(".local.yaml") and "__pycache__" not in l]
    return commit, dirty


def load_frozen():
    text = MANIFEST.read_text()
    want = MANIFEST_SHA.read_text().split()[0]
    got = sha256_text(text)
    if got != want:
        sys.exit(f"manifest hash mismatch: {got} != {want}")
    return json.loads(text), got


def parse(ent, text):
    if ent["experiment"] == "E1":
        agents = e1_episodes()[0][ent["episode"]]["agents"]
        sc, cul = exposure.parse(text or "", agents)
        return dict(scores=sc, culprits=cul)
    b = relay.build(ent["instance"], ent["condition"], ent["view"], ent["prompt"])
    sc, origin, cul = relay.parse(text or "", b["agents"])
    return dict(scores=sc, culprits=cul, origin_named=origin, who=b["who"], truth=b["truth"])


def execute(ents, settings_by_model, manifest_sha, out_path, workers, label):
    commit, dirty = git_state()
    if dirty:
        sys.exit("uncommitted changes under hackathon/ -- commit before a paid run:\n" + "\n".join(dirty))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in out_path.open():
            r = json.loads(l)
            if r.get("status") == "ok":
                done.add(r["id"])
    todo = [e for e in ents if e["id"] not in done]
    print(f"[{label}] {len(ents)} in scope, {len(done & {e['id'] for e in ents})} already done, {len(todo)} to call")
    for e in todo:  # verify every prompt against the manifest BEFORE spending anything
        if messages_sha(messages_for(e)) != e["prompt_sha256"]:
            sys.exit(f"prompt drift for {e['id']}: code no longer renders the frozen prompt")
    lock = threading.Lock()
    fails = []

    def one(e):
        msgs = messages_for(e)
        st = settings_by_model[e["model"]]
        res = call_api(e["model"], msgs, st)
        rec = {k: v for k, v in e.items() if k != "input_chars"}
        rec.update(model_requested=REQUEST_NAME.get(e["model"], e["model"]), model_id=e["model"], model_returned=res["model_returned"], api_settings=st,
                   git_commit=commit, manifest_sha256=manifest_sha, script_version=(exposure.SCRIPT_VERSION if e["experiment"] == "E1" else relay.SCRIPT_VERSION),
                   raw=res["raw"], finish_reason=res["finish_reason"], usage=res["usage"], error=res["error"])
        if res["ok"]:
            p = parse(e, res["raw"])
            rec.update(p)
            rec["parse_ok"] = all(v is not None for v in p["scores"].values())
            rec["status"] = "ok"
        else:
            rec.update(parse_ok=False, status="failed")
        with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["status"] != "ok":
                fails.append(e["id"])
        return rec

    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(one, todo))
    n_parse = sum(1 for r in recs if r["status"] == "ok" and not r["parse_ok"])
    print(f"[{label}] done: {len(recs)} calls, {len(fails)} API failures, {n_parse} parse failures")
    return recs, fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("manifest", "examples", "pilot", "run"))
    p.add_argument("--run", choices=("A0", "A", "B", "C"))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    if a.cmd == "manifest":
        man = build_manifest()
        report(man["evaluations"])
        print("manifest:", MANIFEST, "sha256:", MANIFEST_SHA.read_text().split()[0])
        return
    if a.cmd == "examples":
        ex = FINAL / "examples"
        ex.mkdir(parents=True, exist_ok=True)
        for cond in ("relay_ne1", "relay_ne4", "local_rep4", "local_corr"):
            (ex / f"{cond}_viewB_base.txt").write_text(relay.build(2, cond, "B", "base")["messages"][1]["content"])
        (ex / "E2_system_provenance.txt").write_text(relay.build(0, "local", "B", "provenance")["messages"][0]["content"])
        (ex / "E1_system_provenance.txt").write_text(exposure.system_prompt(exposure.SYSTEM, "provenance"))
        print("wrote", sorted(x.name for x in ex.iterdir()))
        return

    man, msha = load_frozen()
    ents = man["evaluations"]
    if a.cmd == "pilot":
        # 5 gpt-5.4-mini calls: E1 baseline A/B, E1 original+11, E2 relay_plain A, E2 local B (base prompt, Run A cells)
        want = [output_id("E1", M54, "base", "A", "ep0", "baseline"), output_id("E1", M54, "base", "B", "ep0", "baseline"),
                next(e["id"] for e in ents if e["run"] == "A" and e["experiment"] == "E1" and e["condition"] == "original+11"),
                output_id("E2", M54, "base", "A", "inst0", "relay_plain"), output_id("E2", M54, "base", "B", "inst0", "local")]
        sel = [next(e for e in ents if e["id"] == w) for w in want]
        if a.dry_run:
            print("pilot would call:", *want, sep="\n  "); return
        recs, fails = execute(sel, {M54: CANDIDATE_SETTINGS[M54]}, msha, OUT / "pilot.jsonl", 5, "pilot")
        for r in recs:
            print(json.dumps({k: r.get(k) for k in ("id", "status", "model_returned", "finish_reason", "usage", "parse_ok", "error")}))
        return

    sel = [e for e in ents if e["run"] == a.run and (a.model is None or e["model"] == a.model)]
    if a.dry_run:
        bad = [e["id"] for e in sel if messages_sha(messages_for(e)) != e["prompt_sha256"]]
        report(sel)
        print(f"prompt check: {len(sel) - len(bad)}/{len(sel)} prompts re-render to the frozen sha256")
        return
    if not CONFIGS.exists():
        sys.exit(f"{CONFIGS} missing: run the pilot and freeze model settings first")
    cfg = json.loads(CONFIGS.read_text())
    missing = {e["model"] for e in sel} - set(cfg)
    if missing:
        sys.exit(f"no frozen settings for {missing}")
    recs, fails = execute(sel, cfg, msha, OUT / "records.jsonl", a.workers, f"run {a.run}")
    if fails:
        sys.exit(f"{len(fails)} failed calls recorded; stopping (rerun resumes)")


if __name__ == "__main__":
    main()

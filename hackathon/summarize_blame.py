"""Aggregate blame_summary.json files by condition (new weekend work).
Usage: python hackathon/summarize_blame.py <outputs/tag/timestamp>"""
import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path

rows = [json.loads(p.read_text()) for p in Path(sys.argv[1]).rglob("blame_summary.json")]
groups = defaultdict(list)
for r in rows:
    groups[(len(r["colluders"]), r["prompt_variant"])].append(r)
m = lambda xs: f"{st.mean(xs):6.1f}" if xs else "   n/a"
print(f"{'c':>2} {'variant':10} {'n':>3} {'regret':>7} {'opt':>6} {'sh_coll':>7} {'sh_non':>7} {'top1=coll':>9} {'blame_sd':>8}")
for (c, v), rs in sorted(groups.items()):
    sc = [d["shapley"] for r in rs for d in r["agents"].values() if d["is_colluder"]]
    sn = [d["shapley"] for r in rs for d in r["agents"].values() if not d["is_colluder"]]
    top = [r["agents"][max(r["agents"], key=lambda a: r["agents"][a]["shapley"])]["is_colluder"] for r in rs if r["system_regret"] > 1e-6]
    sd = [st.pstdev([d["shapley"] for d in r["agents"].values()]) for r in rs]
    print(f"{c:>2} {v:10} {len(rs):>3} {m([r['system_regret'] for r in rs])} {m([r['optimal_reward'] for r in rs])} "
          f"{m(sc)} {m(sn)} {f'{sum(top)}/{len(top)}' if c else '   -':>9} {m(sd)}")

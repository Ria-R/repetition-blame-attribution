import json,sys,collections,random,statistics
from pathlib import Path
root=Path(sys.argv[1]); random.seed(0)
acc=collections.defaultdict(lambda:collections.defaultdict(list))
for fs in root.rglob("final_summary.json"):
    d=fs.parent; sweep=d.parts[-2]
    cfg=json.loads((d/"run_config.json").read_text())
    col=set(cfg.get("colluders") or [])
    if not col: continue
    ag=list(cfg["roles"].keys()); n=len(ag)
    bb=json.loads((d/"blackboards.json").read_text())
    best=max(bb,key=lambda b:len(b.get("participants") or []))
    cnt=collections.Counter(e["agent"] for e in best["events"] if e.get("kind")=="communication")
    tc={a:cnt.get(a,0) for a in ag}
    key=(sweep.split("_n8")[0] if "n8" not in sweep else sweep)
    def hit(a): return a in col
    A=acc[sweep]
    A["chance"].append(len(col)/n)
    # tie-aware expected hit
    def exp(score,rev):
        m=max(score.values()) if rev else min(score.values())
        tops=[a for a in ag if score[a]==m]; return sum(hit(a) for a in tops)/len(tops)
    A["most_talk"].append(exp(tc,True)); A["least_talk"].append(exp(tc,False))
    bs=json.loads((d/"blame_summary.json").read_text())
    sh={a:bs["agents"][a]["shapley"] for a in ag}; A["shapley"].append(exp(sh,True))
    for x in ("A","B","C","AL","BL"):
        f=d/f"audit_{x}.json"
        if f.exists():
            r=json.loads(f.read_text()); A[x].append(1.0 if r["ranking"][0] in col else 0.0)
print(f"{'sweep':24s}{'n':>4}"+"".join(f"{k:>11s}" for k in ["chance","most_talk","least_talk","shapley","A","B","C","AL","BL"]))
for sw,A in sorted(acc.items()):
    print(f"{sw:24s}{len(A['chance']):>4}"+"".join(f"{statistics.mean(A[k]):>11.2f}" if A[k] else f"{'-':>11s}" for k in ["chance","most_talk","least_talk","shapley","A","B","C","AL","BL"]))

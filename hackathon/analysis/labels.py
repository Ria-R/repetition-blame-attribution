import sys,json,itertools,math,csv,collections,random,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments"/"collusion"))
from compute_jira_optimal import _reconstruct_instance,_load_instance_from_agent_prompts,_load_weights,_evaluate,_load_actual_solution,solve_optimal_assignment
root=Path(sys.argv[1]); out=Path(sys.argv[2])
def shap(ag,R,act,opt):
    cache={}
    r0=R(act)
    def v(S):
        if S not in cache: cache[S]=R({a:(opt[a] if a in S else act[a]) for a in ag})-r0
        return cache[S]
    n=len(ag);ph={}
    for a in ag:
        o=[b for b in ag if b!=a];s=0
        for k in range(n):
            wt=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
            for S in itertools.combinations(o,k): s+=wt*(v(frozenset(S)|{a})-v(frozenset(S)))
        ph[a]=s
    return ph
rows=[]
for fs in sorted(root.rglob("final_summary.json")):
    d=fs.parent; sweep=d.relative_to(root).parts[0]  # top-level sweep folder (run names repeat across sweeps)
    cfg=json.loads((d/"run_config.json").read_text()); col=set(cfg.get("colluders") or [])
    try: inst=_reconstruct_instance(d)
    except Exception: inst=_load_instance_from_agent_prompts(d)
    w=_load_weights(d,None); ag=inst.agent_names; n=len(ag)
    R=lambda a:_evaluate(a,instance=inst,weights=w).joint_reward
    act=_load_actual_solution(d).assignment; act={a:act.get(a) for a in ag}
    T=[None,*inst.task_ids]
    if n<=4:
        best=-1e18;opts=[]
        for c in itertools.product(T,repeat=n):
            a=dict(zip(ag,c));v=R(a)
            if v>best+1e-9:best=v;opts=[a]
            elif abs(v-best)<=1e-9:opts.append(a)
    else:
        o=solve_optimal_assignment(instance=inst,weights=w); opts=[o.assignment]
    S=[shap(ag,R,act,o) for o in opts]
    ph={a:statistics.mean(s[a] for s in S) for a in ag}
    regret=R(opts[0])-R(act)
    cnt=collections.Counter(t for t in act.values() if t)
    bb=json.loads((d/"blackboards.json").read_text())
    best_b=max(bb,key=lambda b:len(b.get("participants") or []))
    msgs=collections.Counter(e["agent"] for e in best_b["events"] if e.get("kind")=="communication")
    for a in ag:
        harm=R(act)-R({**act,a:None}) # marginal contribution of a's claim (neg = harmful)
        rows.append(dict(sweep=sweep,run=d.name,n=n,seed=cfg.get("seed"),variant=cfg.get("prompt_variant"),agent=a,
            role_colluder=int(a in col),attempted_duplicate=int(act[a] is not None and cnt[act[a]]>1),
            attempted_skip=int(act[a] is None),claim_marginal=round(-harm,2),
            regret_alloc=round(ph[a],2),n_tied_optima=len(opts),system_regret=round(regret,2),msgs=msgs.get(a,0)))
out.parent.mkdir(exist_ok=True,parents=True)
with open(out,"w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(len(rows),"agent-rows from",len({r['run'] for r in rows}),"runs")

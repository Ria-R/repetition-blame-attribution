import sys,json,itertools,math,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments"/"collusion"))
from compute_jira_optimal import _reconstruct_instance,_load_instance_from_agent_prompts,_load_weights,_evaluate,_load_actual_solution
root=Path(sys.argv[1])
def shap(ag,R,act,opt):
    cache={}
    def v(S):
        if S not in cache: cache[S]=R({a:(opt[a] if a in S else act[a]) for a in ag})-R(act)
        return cache[S]
    n=len(ag);ph={}
    for a in ag:
        o=[b for b in ag if b!=a];s=0
        for k in range(n):
            wt=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
            for S in itertools.combinations(o,k): s+=wt*(v(frozenset(S)|{a})-v(frozenset(S)))
        ph[a]=s
    return ph
tot=0;tied=0;top1_unstable=0;spread=[];res=[]
for fs in root.rglob("final_summary.json"):
    d=fs.parent
    cfg=json.loads((d/"run_config.json").read_text())
    if cfg.get("num_agents")!=4: continue
    try: inst=_reconstruct_instance(d)
    except Exception: inst=_load_instance_from_agent_prompts(d)
    w=_load_weights(d,None); ag=inst.agent_names; T=[None,*inst.task_ids]
    R=lambda a:_evaluate(a,instance=inst,weights=w).joint_reward
    act=_load_actual_solution(d).assignment; act={a:act.get(a) for a in ag}
    best=-1e18;opts=[]
    for c in itertools.product(T,repeat=4):
        a=dict(zip(ag,c));v=R(a)
        if v>best+1e-9:best=v;opts=[a]
        elif abs(v-best)<=1e-9:opts.append(a)
    tot+=1
    if len(opts)>1:
        tied+=1
        S=[shap(ag,R,act,o) for o in opts]
        tops=[max(s,key=s.get) for s in S]
        if len(set(tops))>1: top1_unstable+=1
        spread.append(max(max(s[a] for s in S)-min(s[a] for s in S) for a in ag))
print("4-agent runs",tot,"tied-opt runs",tied,"of which top-1 Shapley agent changes with tie choice",top1_unstable)
import statistics
print("median max per-agent Shapley spread across tied optima",round(statistics.median(spread),2),"max",round(max(spread),2))

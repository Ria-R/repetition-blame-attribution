import sys,json,itertools,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments"/"collusion"))
from compute_jira_optimal import _reconstruct_instance,_load_instance_from_agent_prompts,_load_weights,_evaluate,solve_optimal_assignment,_load_actual_solution
root=Path(sys.argv[1]); bad=0;n=0;ties=0;skipped=0;shap_bad=0
import random
random.seed(0)
dirs=[fs.parent for fs in root.rglob("final_summary.json")]
for d in dirs:
    try:
        inst=_reconstruct_instance(d)
    except Exception:
        inst=_load_instance_from_agent_prompts(d)
    w=_load_weights(d,None)
    ag=inst.agent_names; T=[None,*inst.task_ids]
    if len(ag)>5:
        skipped+=1; continue
    R=lambda a:_evaluate(a,instance=inst,weights=w).joint_reward
    best=-1e18;cnt=0
    for combo in itertools.product(T,repeat=len(ag)):
        v=R(dict(zip(ag,combo)))
        if v>best+1e-9: best=v;cnt=1
        elif abs(v-best)<=1e-9: cnt+=1
    opt=solve_optimal_assignment(instance=inst,weights=w)
    n+=1
    if abs(opt.joint_reward-best)>1e-6: bad+=1; print("MISMATCH",d.name,opt.joint_reward,best)
    if cnt>1: ties+=1
print("tested",n,"mismatch",bad,"runs with tied optima",ties,"skipped(n>5)",skipped)

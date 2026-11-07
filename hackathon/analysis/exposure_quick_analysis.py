import json,random,statistics as st,collections,sys
L=[json.loads(l) for l in open('results/exposure/calls_gpt-4o-mini.jsonl')]
base={(l['ep'],l['view']):l for l in L if l['mtype']=='base'}
def ci(xs,n=4000):
    r=random.Random(0); ms=sorted(st.mean(r.choices(xs,k=len(xs))) for _ in range(n)); return ms[int(.025*n)],ms[int(.975*n)]
def rank(sc,a):  # rank of a (1=top), ties averaged
    v=sc[a]; return 1+sum(x>v for x in sc.values())+0.5*sum(x==v for k,x in sc.items() if k!=a)
g=collections.defaultdict(list)
for l in L:
    if l['mtype']=='base': continue
    b=base[(l['ep'],l['view'])]; t=l['target']
    d=l['scores'][t]-b['scores'][t]
    # others' mean change
    oth=st.mean(l['scores'][a]-b['scores'][a] for a in l['scores'] if a!=t)
    top=float(max(l['scores'],key=lambda a:(l['scores'][a],a==t))==t) - float(max(b['scores'],key=lambda a:(b['scores'][a],a==t))==t)
    cul=float(t in l['culprits'])-float(t in b['culprits'])
    for key in [(l['view'],l['mtype'],l['added']),(l['view'],l['mtype'],l['added'],'col' if l.get('target_is_colluder') else 'hon')]:
        g[key].append((d,oth,top,cul,rank(b['scores'],t)-rank(l['scores'],t)))
print("view type add [grp] n | d_target [95%CI] | d_others | d_rankup | d_top1 | d_named_culprit")
for k in sorted(g,key=str):
    xs=g[k]; d=[x[0] for x in xs]; lo,hi=ci(d)
    print(k,len(xs),f"| {st.mean(d):+.2f} [{lo:+.2f},{hi:+.2f}] | {st.mean(x[1] for x in xs):+.2f} | {st.mean(x[4] for x in xs):+.2f} | {st.mean(x[2] for x in xs):+.2f} | {st.mean(x[3] for x in xs):+.2f}")
# baseline sanity: does baseline B top-1 hit a colluder?
for v in 'AB':
    bs=[b for (e,vv),b in base.items() if vv==v and b['colluders']]
    print(v,'baseline top1 colluder',sum(max(b['scores'],key=b['scores'].get) in b['colluders'] for b in bs),'/',len(bs))

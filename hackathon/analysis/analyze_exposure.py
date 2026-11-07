"""Clustered analysis of the frozen-transcript exposure experiment (new weekend work).

  python analyze_exposure.py calls_gpt-4o-mini.jsonl labels.csv [--boot 5000]

Reports, with 95% CIs from a bootstrap that resamples whole EPISODES (rows within an episode are correlated):
  1. mean paired change in the target's score, by view x message type x added
  2. original - neutral contrast, paired within (episode, view, target, added)
  3. does the change track harm? Spearman between d_target and the target's regret allocation
"""
import csv, json, random, statistics, sys, collections


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3: return float("nan")
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def cluster_ci(rows, stat, B, rng):
    eps = sorted({r["ep"] for r in rows}); by = collections.defaultdict(list)
    for r in rows: by[r["ep"]].append(r)
    est = stat(rows); bs = []
    for _ in range(B):
        samp = [r for e in rng.choices(eps, k=len(eps)) for r in by[e]]
        v = stat(samp)
        if v == v: bs.append(v)
    bs.sort()
    return est, bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1], len(eps)


def main():
    path, lab = sys.argv[1], sys.argv[2]
    B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 5000
    rng = random.Random(0)
    recs = [json.loads(l) for l in open(path)]
    alloc = {(r["sweep"] + "/" + r["run"], r["agent"]): float(r["regret_alloc"]) for r in csv.DictReader(open(lab))}
    base = {(r["ep"], r["view"]): r["scores"] for r in recs if r["mtype"] == "base"}
    rows = []
    for r in recs:
        if r["mtype"] == "base": continue
        b = base.get((r["ep"], r["view"])); t = r["target"]
        if not b or b.get(t) is None or r["scores"].get(t) is None: continue
        rows.append(dict(ep=r["ep"], view=r["view"], mtype=r["mtype"], added=r["added"], target=t,
                         col=r["target_is_colluder"], d=r["scores"][t] - b[t],
                         alloc=alloc.get((r["tag"], t), float("nan"))))
    n_alloc = sum(1 for r in rows if r["alloc"] == r["alloc"])
    print(f"{len(rows)} perturbed calls with valid paired scores; {n_alloc} matched to labels.csv "
          f"(episode-clustered bootstrap, B={B})\n")
    mean = lambda R: statistics.mean(x["d"] for x in R) if R else float("nan")

    print("1. d_target = score(perturbed) - score(baseline)")
    for key in sorted({(r["view"], r["mtype"], r["added"]) for r in rows}):
        R = [r for r in rows if (r["view"], r["mtype"], r["added"]) == key]
        e, lo, hi, n = cluster_ci(R, mean, B, rng)
        print(f"  view {key[0]} {key[1]:8s} +{key[2]:<2d}  {e:+.2f} [{lo:+.2f},{hi:+.2f}]  n={len(R)} eps={n}")

    print("\n2. original - neutral, paired on the same target")
    idx = {(r["ep"], r["view"], r["target"], r["added"], r["mtype"]): r for r in rows}
    pairs = []
    for (ep, v, t, k, m), r in idx.items():
        if m == "original" and (ep, v, t, k, "neutral") in idx:
            pairs.append(dict(ep=ep, view=v, added=k, t=t, d=r["d"] - idx[(ep, v, t, k, "neutral")]["d"]))
    for key in sorted({(p["view"], p["added"]) for p in pairs}):
        P = [p for p in pairs if (p["view"], p["added"]) == key]
        e, lo, hi, n = cluster_ci(P, mean, B, rng)
        print(f"  view {key[0]} +{key[1]:<2d}  {e:+.2f} [{lo:+.2f},{hi:+.2f}]  n={len(P)}")

    print("\n3. does the change track harm? Spearman(d_target, target's regret allocation)")
    for key in sorted({(r["view"], r["mtype"], r["added"]) for r in rows}):
        R = [r for r in rows if (r["view"], r["mtype"], r["added"]) == key and r["alloc"] == r["alloc"]]
        st = lambda S: spearman([s["d"] for s in S], [s["alloc"] for s in S])
        e, lo, hi, n = cluster_ci(R, st, min(B, 2000), rng)
        print(f"  view {key[0]} {key[1]:8s} +{key[2]:<2d}  rho={e:+.2f} [{lo:+.2f},{hi:+.2f}]  n={len(R)}")

    print("\n4. ceiling check: harm vs baseline score")
    for r in rows: r["b"] = base[(r["ep"], r["view"])][r["target"]]
    for v in ("A", "B"):
        R = [r for r in rows if r["view"] == v and r["mtype"] == "original" and r["added"] == 11]
        print(f"  view {v}: rho(baseline, alloc)={spearman([r['b'] for r in R],[r['alloc'] for r in R]):+.2f}  "
              f"mean baseline={statistics.mean(r['b'] for r in R):.1f}  at ceiling(>=9)={sum(r['b']>=9 for r in R)}/{len(R)}")

    print("\n5a. complete-case: same targets in both message types (targets with own messages only)")
    have_orig = {(r["ep"], r["target"]) for r in rows if r["mtype"] == "original"}
    cc = [r for r in rows if (r["ep"], r["target"]) in have_orig]
    for key in sorted({(r["view"], r["mtype"], r["added"]) for r in cc}):
        R = [r for r in cc if (r["view"], r["mtype"], r["added"]) == key]
        e, lo, hi, n = cluster_ci(R, mean, B, rng)
        print(f"  view {key[0]} {key[1]:8s} +{key[2]:<2d}  {e:+.2f} [{lo:+.2f},{hi:+.2f}]  n={len(R)}")
    print("  dropped (silent) targets (ep, name, is_colluder):",
          sorted({(r['ep'], r['target'], r['col']) for r in rows} - {(r['ep'], r['target'], r['col']) for r in cc}))

    print("\n5b. difference-in-differences: (orig - neutral)_A - (orig - neutral)_B, paired on (episode, target, added)")
    for k in sorted({p["added"] for p in pairs}):
        pa = {(p["ep"], p["t"]): p["d"] for p in pairs if p["view"] == "A" and p["added"] == k}
        pb = {(p["ep"], p["t"]): p["d"] for p in pairs if p["view"] == "B" and p["added"] == k}
        D = [dict(ep=e, d=pa[(e, t)] - pb[(e, t)]) for (e, t) in pa if (e, t) in pb]
        e, lo, hi, n = cluster_ci(D, mean, B, rng)
        print(f"  +{k:<2d}  {e:+.2f} [{lo:+.2f},{hi:+.2f}]  n={len(D)}")

    print("\n5c. displacement: change in (target - max other) gap, top-suspect changes, mean change of the others")
    D = []
    for r in recs:
        if r["mtype"] == "base" or None in r["scores"].values(): continue
        b = base.get((r["ep"], r["view"]))
        if not b or None in b.values(): continue
        s, t = r["scores"], r["target"]
        gap = lambda x: x[t] - max(v for a, v in x.items() if a != t)
        top = lambda x: {a for a, v in x.items() if v == max(x.values())}
        D.append(dict(ep=r["ep"], view=r["view"], mtype=r["mtype"], added=r["added"], dgap=gap(s) - gap(b),
                      became_top=float(t in top(s) and t not in top(b)), top_changed=float(top(s) != top(b)),
                      d_others=statistics.mean(s[a] - b[a] for a in s if a != t),
                      d_all=statistics.mean(s[a] - b[a] for a in s)))
    print(f"  {'cell':22s} {'d_gap':>22s} {'became_top':>20s} {'top_set_changed':>16s} {'d_others':>20s} {'d_mean_all':>10s}")
    for key in sorted({(x["view"], x["mtype"], x["added"]) for x in D}):
        R = [x for x in D if (x["view"], x["mtype"], x["added"]) == key]
        g = cluster_ci(R, lambda S: statistics.mean(x["dgap"] for x in S), B, rng)
        bt = cluster_ci(R, lambda S: statistics.mean(x["became_top"] for x in S), B, rng)
        do = cluster_ci(R, lambda S: statistics.mean(x["d_others"] for x in S), B, rng)
        print(f"  {key[0]} {key[1]:8s} +{key[2]:<2d}        {g[0]:+.2f} [{g[1]:+.2f},{g[2]:+.2f}]  {bt[0]:.2f} [{bt[1]:.2f},{bt[2]:.2f}]"
              f"  {statistics.mean(x['top_changed'] for x in R):16.2f}  {do[0]:+.2f} [{do[1]:+.2f},{do[2]:+.2f}]  {statistics.mean(x['d_all'] for x in R):+10.2f}")

    print("\n5d. ceiling bands: mean d_target by baseline band (+11 only)")
    for v in ("A", "B"):
        for m in ("original", "neutral"):
            R = [r for r in rows if r["view"] == v and r["mtype"] == m and r["added"] == 11]
            for lo_b, hi_b in ((0, 3), (4, 6), (7, 10)):
                S = [r for r in R if lo_b <= r["b"] <= hi_b]
                if len({r["ep"] for r in S}) >= 3:
                    e, l, h, n = cluster_ci(S, mean, 2000, rng)
                    rho = spearman([r["d"] for r in S], [r["alloc"] for r in S])
                    print(f"  {v} {m:8s} baseline {lo_b}-{hi_b}: d={e:+.2f} [{l:+.2f},{h:+.2f}] n={len(S)} eps={n}  rho(d,alloc)={rho:+.2f}")

if __name__ == "__main__":
    main()

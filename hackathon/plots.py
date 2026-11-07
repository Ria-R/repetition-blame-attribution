"""Plots for the results doc (new weekend work). Usage: python hackathon/plots.py <outdir> <outputs root>/<tag> ..."""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analyze import boot_ci, load  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

LABEL = {
    ("pilot_n4", 2, "control"): "LLM colluders\ncontrol prompt", ("pilot_n4", 2, "simple"): "LLM colluders\nsimple",
    ("pilot_n4", 2, "deception"): "LLM colluders\ndeception", ("pilot_n4_aggr", 2, "aggressive"): "LLM colluders\naggressive",
    ("scripted_steal", 2, "control"): "Scripted\nsteal", ("scripted_bait", 2, "control"): "Scripted\nbait",
    ("scripted_steal_silent", 2, "control"): "Scripted steal\n(silent)", ("scripted_bait_silent", 2, "control"): "Scripted bait\n(silent)",
}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = load([Path(p) for p in sys.argv[2:]])
    n4 = [r for r in rows if r["n"] == 4]
    base = [r["regret"] for r in n4 if r["c"] == 0]
    conds = [k for k in LABEL if any((r["tag"], r["c"], r["variant"]) == k for r in n4)]

    # 1) regret by condition vs the no-colluder baseline
    fig, ax = plt.subplots(figsize=(9, 4))
    lo, hi = boot_ci(base)
    ax.axhspan(lo, hi, color=GRID, alpha=0.7, lw=0)
    ax.axhline(st.mean(base), color=INK2, lw=1, ls="--")
    ax.text(len(conds) - 0.5, st.mean(base) + 2, f"no colluders: {st.mean(base):.0f} (n={len(base)})", ha="right", color=INK2, fontsize=9)
    for i, k in enumerate(conds):
        xs = [r["regret"] for r in n4 if (r["tag"], r["c"], r["variant"]) == k]
        l, h = boot_ci(xs)
        ax.plot([i, i], [l, h], color=S[0], lw=2)
        ax.plot(i, st.mean(xs), "o", color=S[0], ms=8, mec=SURF, mew=2)
        ax.text(i + 0.12, st.mean(xs), f"{st.mean(xs):.0f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(conds)), [LABEL[k] for k in conds], fontsize=8)
    ax.set_ylabel("Reward lost vs exact optimum (opt ≈ 108)")
    ax.set_title("Saboteurs do not add measurable damage over honest-team noise (4 agents, mean, 95% bootstrap CI)", fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "regret_by_condition.png", dpi=160); plt.close(fig)

    # 2) top-1 hits a colluder: counterfactual truth vs auditors
    keys = [("Counterfactual (Shapley)", None), ("A: messages", "A"), ("B: messages + outcome", "B"), ("C: value share", "C")]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    for j, (name, x) in enumerate(keys):
        vals = []
        for k in conds:
            rs = [r for r in n4 if (r["tag"], r["c"], r["variant"]) == k]
            if x is None:
                v = st.mean(float(max(r["shapley"], key=r["shapley"].get) in r["colluders"]) for r in rs)
            else:
                rs = [r for r in rs if f"audit_{x}" in r]
                v = st.mean(float(r[f"audit_{x}"]["ranking"][0] in r["colluders"]) for r in rs) if rs else float("nan")
            vals.append(v)
        pos = [i + (j - 1.5) * w for i in range(len(conds))]
        ax.bar(pos, vals, width=w - 0.03, color=S[j], label=name)
    ax.axhline(0.5, color=INK2, lw=1, ls="--"); ax.text(-0.45, 0.52, "chance (2 of 4)", fontsize=8, color=INK2)
    ax.set_xticks(range(len(conds)), [LABEL[k] for k in conds], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Top-ranked agent is a colluder")
    ax.set_title("Counterfactual blame finds scripted saboteurs; LLM auditors (Qwen-7B) fall below chance", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.22))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "top1_accuracy.png", dpi=160); plt.close(fig)

    # 3) auditor A blames whoever talks most
    xs, ys = [], []
    for r in rows:
        a = r.get("audit_A")
        if not a:
            continue
        bb = json.loads((Path(r["dir"]) / "blackboards.json").read_text())
        pub = max(bb, key=lambda b: len(b["participants"]))
        cnt = {g: 0 for g in a["scores"]}
        for e in pub["events"]:
            if e["kind"] == "communication" and e["agent"] in cnt:
                cnt[e["agent"]] += 1
        tot = sum(cnt.values()) or 1
        for g in cnt:
            xs.append(cnt[g] / tot); ys.append(a["scores"][g])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=18, color=S[0], alpha=0.5, edgecolors="none")
    r = st.correlation(xs, ys)
    ax.set_xlabel("Agent's share of public messages"); ax.set_ylabel("Auditor A responsibility score (0-10)")
    ax.set_title(f"Auditor A blames the talkative (r = {r:.2f}, {len(xs)} agent-runs)", fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "talkativeness_bias.png", dpi=160); plt.close(fig)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()

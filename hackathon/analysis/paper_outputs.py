"""Paper figures and paper_numbers.md from the frozen records (new weekend work, Apart Swarm Alignment Hackathon).

  .venv/bin/python hackathon/analysis/paper_outputs.py

Reads results/final/records.jsonl (never modified) through final_analysis.py, writes
  results/final/figures/{fig1_relay_schematic,fig2_primary,fig3_provenance_4omini,figA1_controls}.{pdf,png,csv}
  results/final/paper_numbers.md
No API calls.
"""
from __future__ import annotations

import contextlib, csv, hashlib, io, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import final_analysis as fa  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

RES = REPO / "results" / "final"
FIG = RES / "figures"
M4, M5 = fa.MODELS
# Okabe-Ito colour-blind-safe palette; same colour + marker per model in every figure
STYLE = {M4: dict(color="#0072B2", marker="o", label="gpt-4o-mini"),
         M5: dict(color="#D55E00", marker="s", label="gpt-5.4-mini")}
PRICES = {M4: (0.15, 0.60), M5: (0.75, 4.50)}
XLAB = "Difference in responsibility score (0-10 scale)"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, name, rows, header):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    with (FIG / f"{name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
    box = dict(Monitor=(0.2, 4.6), Coordinator=(3.6, 4.6), Remediator=(7.0, 4.6), Analyst=(3.6, 1.9), Scribe=(7.0, 1.9))
    W, H = 2.8, 1.0
    for name, (x, y) in box.items():
        bystander = name in ("Analyst", "Scribe")
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05", fc="white", ec="black",
                                    lw=0.9, ls="--" if bystander else "-"))
        ax.text(x + W / 2, y + H / 2, name, ha="center", va="center", fontsize=8)
    arr = dict(arrowstyle="-|>", lw=0.9, color="black")
    ax.annotate("", xy=(3.6, 5.1), xytext=(3.0, 5.1), arrowprops=arr)
    ax.text(3.3, 5.75, "1. diagnosis", ha="center", fontsize=6.5)
    ax.annotate("", xy=(7.0, 5.1), xytext=(6.4, 5.1), arrowprops=arr)
    ax.text(6.7, 5.75, "2. forwards,\ncites Monitor", ha="center", fontsize=6.5)
    ax.text(5.0, 6.9, "x1 in relay_min, x4 in relay_plain", ha="center", fontsize=6.5, style="italic")
    ax.annotate("", xy=(7.6, 4.6), xytext=(4.6, 2.9), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", ls="--"))
    ax.text(4.2, 3.55, "3. counter-\nevidence", ha="right", fontsize=6.5)
    ax.text(8.4, 4.25, "4. acts", ha="center", fontsize=6.5)
    ax.text(0.1, 0.7, "View A: auditor sees the shared log only.\nView B: shared log + verified postmortem.",
            ha="left", va="center", fontsize=6.5)
    rows = [["Monitor", "Coordinator", "diagnosis (wrong root cause in relay conditions)", 1],
            ["Coordinator", "Remediator", "forward citing Monitor (x1 relay_min, x4 relay_plain)", 2],
            ["Analyst", "log", "counter-evidence (posted after all Coordinator lines)", 3],
            ["Remediator", "system", "acts on the forwarded report", 4]]
    save(fig, "fig1_relay_schematic", rows, ["from", "to", "message", "order"])


# ------------------------------------------------------------------------------------------------ forest helper
def forest(ax, entries, title):
    """entries: list of (label, model, (mean, lo, hi, n, k))"""
    ys = list(range(len(entries)))[::-1]
    for y, (lab, m, r) in zip(ys, entries):
        s = STYLE[m]
        ax.errorbar(r[0], y, xerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt=s["marker"], color=s["color"], ms=5,
                    capsize=2.5, lw=1.1, mfc=s["color"])
    ax.axvline(0, ls="--", color="grey", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([e[0] for e in entries])
    ax.set_title(title, fontsize=9)


def cells(idx_fn):
    out = []
    for m in (M4, M5):
        for v in ("A", "B"):
            out.append((f"{STYLE[m]['label']}, view {v}", m, v, idx_fn(m, v)))
    return out


# ------------------------------------------------------------------------------------------------ Fig 2
def fig2(e1d, idx):
    left = cells(lambda m, v: fa.e1_contrast(e1d, m, "base", v))
    right = cells(lambda m, v: fa.e2_pair(idx, m, "base", v, "relay_plain", "relay_min", fa.disp))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    fig.supxlabel(XLAB, fontsize=9, y=-0.04)
    forest(axs[0], [(l, m, r) for l, m, v, r in left], "E1: original - neutral repetition (+11)")
    forest(axs[1], [(l, m, r) for l, m, v, r in right], "E2: (relay - origin), relay_plain - relay_min")
    handles = [plt.Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ls="", label=STYLE[m]["label"]) for m in (M4, M5)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.22))
    rows = [["E1 original-neutral +11", STYLE[m]["label"], v, *[round(x, 4) for x in r[:3]], r[3], r[4]] for l, m, v, r in left]
    rows += [["E2 plain-min (relay-origin)", STYLE[m]["label"], v, *[round(x, 4) for x in r[:3]], r[3], r[4]] for l, m, v, r in right]
    save(fig, "fig2_primary", rows, ["contrast", "model", "view", "mean", "ci_lo", "ci_hi", "n", "clusters"])


# ------------------------------------------------------------------------------------------------ Fig 3
def prop(idx, m, p, v, c, f):
    cl = {i: [f(idx[(m, p, v, c, i)])] for i in range(30) if (m, p, v, c, i) in idx}
    return fa.boot(cl)


def fig3(idx):
    s = STYLE[M4]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.subplots_adjust(wspace=0.45)
    rows = []
    off = {"base": -0.12, "provenance": 0.12}
    for p in ("base", "provenance"):
        for j, v in enumerate(("A", "B")):
            r1 = prop(idx, M4, p, v, "relay_plain", fa.origin_ok)
            r2 = fa.e2_pair(idx, M4, p, v, "relay_plain", "relay_min", fa.disp)
            for ax, r in ((axs[0], r1), (axs[1], r2)):
                ax.errorbar(j + off[p], r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt=s["marker"], color=s["color"],
                            mfc=s["color"] if p == "base" else "white", ms=5.5, capsize=2.5, lw=1.1)
            rows.append(["origin named = Monitor, relay_plain", p, v, *[round(x, 4) for x in r1[:3]], r1[3], r1[4]])
            rows.append(["(relay-origin) relay_plain - relay_min", p, v, *[round(x, 4) for x in r2[:3]], r2[3], r2[4]])
    for ax in axs:
        ax.set_xticks([0, 1]); ax.set_xticklabels(["View A\n(log only)", "View B\n(log + postmortem)"]); ax.set_xlim(-0.5, 1.5)
    axs[0].set_ylabel("Proportion naming the Monitor as origin"); axs[0].set_ylim(0, 1.05)
    axs[0].set_title("Source identification (relay_plain)", fontsize=9)
    axs[1].set_ylabel("Difference in responsibility\nscore (0-10 scale)"); axs[1].axhline(0, ls="--", color="grey", lw=0.8)
    axs[1].set_title("Repetition sensitivity: relay_plain - relay_min", fontsize=9)
    handles = [plt.Line2D([], [], color=s["color"], marker=s["marker"], ls="", mfc=s["color"], label="base prompt"),
               plt.Line2D([], [], color=s["color"], marker=s["marker"], ls="", mfc="white", label="provenance prompt")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Provenance prompting improves source identification but increases repetition sensitivity in "
                 "responsibility scores", fontsize=8.5, y=1.06)
    save(fig, "fig3_provenance_4omini", rows, ["measure", "prompt", "view", "mean", "ci_lo", "ci_hi", "n", "clusters"])


# ------------------------------------------------------------------------------------------------ Fig A1
def figA1(idx):
    left = cells(lambda m, v: fa.e2_pair(idx, m, "base", v, "relay_filler", "relay_min", fa.disp))
    right = cells(lambda m, v: fa.e2_pair(idx, m, "base", v, "relay_ne4", "relay_ne1", fa.disp))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    fig.supxlabel(XLAB, fontsize=9, y=-0.04)
    forest(axs[0], [(l, m, r) for l, m, v, r in left], "Volume only: relay_filler - relay_min")
    forest(axs[1], [(l, m, r) for l, m, v, r in right], "Non-endorsing: relay_ne4 - relay_ne1")
    handles = [plt.Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ls="", label=STYLE[m]["label"]) for m in (M4, M5)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.22))
    rows = [["(relay-origin) filler - min", STYLE[m]["label"], v, *[round(x, 4) for x in r[:3]], r[3], r[4]] for l, m, v, r in left]
    rows += [["(relay-origin) ne4 - ne1", STYLE[m]["label"], v, *[round(x, 4) for x in r[:3]], r[3], r[4]] for l, m, v, r in right]
    save(fig, "figA1_controls", rows, ["contrast", "model", "view", "mean", "ci_lo", "ci_hi", "n", "clusters"])


# ------------------------------------------------------------------------------------------------ paper_numbers.md
def sh(b: bytes):
    return hashlib.sha256(b).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout


def paper_numbers(recs, analysis_text):
    commits = sorted({r["git_commit"] for r in recs})
    man_sha = sorted({r["manifest_sha256"] for r in recs})
    man = json.loads((REPO / "hackathon" / "final" / "manifest.json").read_text())
    cfg = json.loads((REPO / "hackathon" / "final" / "model_configs.json").read_text())
    prereg_run = git("show", f"{commits[0]}:hackathon/PREREG.md")
    rets = {}
    cost = 0.0
    for r in recs:
        rets.setdefault(r["model"], set()).add(r.get("model_returned"))
        pi, po = PRICES[r["model"]]
        cost += r["usage"].get("prompt_tokens", 0) * pi / 1e6 + r["usage"].get("completion_tokens", 0) * po / 1e6
    n_fail = sum(r["status"] != "ok" for r in recs)
    n_parse = sum(r["status"] == "ok" and not r["parse_ok"] for r in recs)
    req = {r["model"]: r["model_requested"] for r in recs}
    L = ["# Paper numbers (single source of truth)", "",
         "Generated by `hackathon/analysis/paper_outputs.py` from `results/final/records.jsonl`; do not edit by hand.",
         "All contrasts: paired, 95% bootstrap CIs resampling whole clusters (E1 episode, E2 incident instance), "
         f"{fa.B} resamples, seed 0. Format identical to analysis.txt.", "",
         "## Run facts", "",
         f"- Total calls: {len(recs)} (manifest: {man['n_evaluations']}); API failures: {n_fail}; parse failures: {n_parse}",
         f"- Cost from recorded token usage: ${cost:.2f} (gpt-4o-mini $0.15/$0.60, gpt-5.4-mini $0.75/$4.50 per 1M in/out)",
         *[f"- {m}: requested as `{req[m]}`, returned snapshot(s): {', '.join(sorted(map(str, rets[m])))}; frozen API settings: "
           f"`{json.dumps(cfg[m])}`" for m in fa.MODELS],
         f"- Git commit of the code that produced every record: `{', '.join(commits)}`",
         f"- Manifest SHA-256 (as recorded in every record): `{', '.join(man_sha)}`",
         f"- PREREG.md SHA-256 at that commit: `{sh(prereg_run.encode())}`",
         *[f"- System prompt SHA-256 {k}: `{v}`" for k, v in sorted(man["system_prompt_sha256"].items())],
         "- Per-call rendered-prompt SHA-256: field `prompt_sha256` in every record (checked against the manifest before each call).",
         "- Origin naming is defined for relay conditions only (Amendment 1); the origin_ok column for local, local_rep4 and "
         "local_corr in section 2 is scored against the Monitor and is not a pre-registered measure.", ""]
    # split analysis text into its sections, each as a code block, keep format identical
    secs = re.split(r"\n(?==+\n)", "\n" + analysis_text)
    L += ["## Analysis output (results/final/analysis.txt, including the A2 EXPLORATORY section)", ""]
    for s in secs:
        s = s.strip("\n")
        if not s:
            continue
        lines = s.splitlines()
        title = lines[1] if lines[0].startswith("=") and len(lines) > 1 else "Header"
        L += [f"### {title}", "", "```", s, "```", ""]
    col = RES / "colosseum"
    L += ["## Colosseum facts (appendix)", "",
          "Data: the 142 Colosseum run directories under experiments/collusion/outputs excluding the 2 scripted_smoke test runs "
          "(identical to the colosseum-blame zip).", ""]
    counts = {}
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "experiments" / "collusion" / "outputs"
    for fs in sorted(data.rglob("final_summary.json")):
        sw = fs.parent.relative_to(data).parts[0]
        if sw == "scripted_smoke":
            continue
        counts[sw] = counts.get(sw, 0) + 1
    L += ["Runs per sweep: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" (total {sum(counts.values())})", ""]
    for name, f in (("Solver brute-force check (solver_test.py)", "solver_test.txt"),
                    ("Tied optima and Shapley top-1 flips (ties.py)", "ties.txt"),
                    ("Top-1 colluder hit rate baselines, tie-aware (baselines.py; Qwen-7B auditors A/B/C)", "baselines.txt")):
        txt = (col / f).read_text().strip() if (col / f).exists() else "(missing)"
        L += [f"### {name}", "", "```", txt, "```", ""]
    bl = (col / "baselines.txt").read_text().splitlines() if (col / "baselines.txt").exists() else []
    rows = [l.split() for l in bl[1:] if l.strip()]
    if rows:
        n = sum(int(r[1]) for r in rows)
        lt = sum(int(r[1]) * float(r[4]) for r in rows) / n
        mt = sum(int(r[1]) * float(r[3]) for r in rows) / n
        ch = sum(int(r[1]) * float(r[2]) for r in rows) / n
        L += [f"Least-talkative-agent baseline, pooled over all {n} runs with designated colluders in sweeps "
              f"{', '.join(r[0] for r in rows)} (includes the silent policies scripted_bait_silent and scripted_steal_silent): "
              f"{lt:.2f} (= {round(lt * n, 2)}/{n}); most-talkative {mt:.2f}; chance {ch:.2f}.", ""]
    (RES / "paper_numbers.md").write_text("\n".join(L) + "\n")


def main():
    recs = fa.load(RES / "records.jsonl")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e1d = fa.e1_index(recs)
        idx = fa.e2_idx(recs)
    fig1(); fig2(e1d, idx); fig3(idx); figA1(idx)
    paper_numbers(recs, (RES / "analysis.txt").read_text())
    print("wrote", sorted(p.name for p in FIG.iterdir()), "and", RES / "paper_numbers.md")


if __name__ == "__main__":
    main()

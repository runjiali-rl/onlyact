"""Table 1 (linear probing of action information) from probe result files.

Reads <results>/<representation>/<dataset>_seed{0,1,2}.json written by probe/probe.py and prints / writes the LaTeX table:
mean over the three probe seeds (R^2; Procgen: macro-F1), Avg. = mean over the 12 continuous-action datasets,
Delta = JustAct minus the best other method. The LaTeX uses the paper macros \\best, \\second, \\mname and the colours
oursblue, deltapos, deltaneg.

    python tables/table1_probe.py --results results/table1 --out table1.tex
"""
import argparse
import json
import os
import statistics

SEEN = {"libero", "droid", "spatialvid_nav"}  # videos seen (without labels) by the JustAct encoder during training
GROUPS = [("Robot manipulation", ["libero", "droid", "rlbench", "robocasa", "droid_iprl", "metaworld", "utaustin_mutex", "vp2_robosuite", "vp2_robodesk"]),
          ("Camera navigation", ["spatialvid_nav", "arkitscenes", "realestate10k"]),
          ("Game (F1)", ["procgen_plan"])]
NAME = {"libero": "LIBERO", "droid": "DROID", "rlbench": "RLBench", "robocasa": "RoboCasa", "droid_iprl": "DROID-IPRL", "metaworld": "MetaWorld",
        "utaustin_mutex": "MUTEX", "vp2_robosuite": "Robosuite", "vp2_robodesk": "RoboDesk", "spatialvid_nav": "SpatialVID",
        "arkitscenes": "ARKitScenes", "realestate10k": "RE10K", "procgen_plan": "Procgen"}
METHODS = [("random", "Random"), ("adaworld_lam_retrained", "AdaWorld$^\\dagger$~\\cite{gao2025adaworld}"),
           ("dreamdojo_lam_retrained", "DreamDojo$^\\dagger$~\\cite{gao2026dreamdojo}"),
           ("vjepa2_frozen", "V-JEPA~2~\\cite{vjepa2}"), ("justact", "\\textbf{\\mname} (ours)")]
OTHERS = ["adaworld_lam_retrained", "dreamdojo_lam_retrained", "vjepa2_frozen"]
CAPTION = ("\\caption{\\textbf{Linear probing of action information.} A linear layer predicts the ground-truth action from each frozen representation. "
           "We report test $R^2$ on held-out scenes (Procgen: macro-F1 over 15 discrete actions), averaged over 3 probe seeds (std $\\le$ 0.02). "
           "$^{\\circ}$Videos seen by our encoder during training, without action labels; all other datasets are entirely unseen. "
           "$^\\dagger$Retrained on the same 168K training pairs as our encoder. Avg.: mean $R^2$ over the 12 continuous-action datasets. "
           "$\\Delta$: ours minus the best other method. \\best{Bold}: best; \\second{underline}: second best.}")


def load(results, seeds):
    E, S = {}, {}
    for _, cs in GROUPS:
        for c in cs:
            E[c], S[c] = {}, {}
            for k, _ in METHODS:
                vals = []
                for s in seeds:
                    r = json.load(open(os.path.join(results, k, f"{c}_seed{s}.json")))
                    vals.append(r["f1_macro"] if c == "procgen_plan" else r["r2"])
                # The paper prints means that were first rounded to 4 decimals.
                E[c][k] = round(sum(vals) / len(vals), 4)
                S[c][k] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return E, S


def render(E):
    cont = [c for _, cs in GROUPS for c in cs if c != "procgen_plan"]
    avg = {k: sum(E[c][k] for c in cont) / len(cont) for k, _ in METHODS}
    cols = [c for _, cs in GROUPS for c in cs] + ["avg"]
    val = lambda c, k: avg[k] if c == "avg" else E[c][k]
    num = lambda x: f"{x:.3f}".replace("-", "$-$") if x < 0 else f"{x:.3f}"

    def fmt(c, k):
        if k == "random":
            return f"\\textcolor{{gray}}{{{num(val(c, k))}}}"
        order = sorted(OTHERS + ["justact"], key=lambda m: -val(c, m))
        s = num(val(c, k))
        return f"\\best{{{s}}}" if k == order[0] else f"\\second{{{s}}}" if k == order[1] else s

    def delta(c):
        d = val(c, "justact") - max(val(c, m) for m in OTHERS)
        return f"\\textcolor{{{'deltapos' if d >= 0 else 'deltaneg'}}}{{{'+' if d >= 0 else '$-$'}{abs(d):.2f}}}"

    head = lambda c: "Avg." if c == "avg" else NAME[c] + ("$^{\\circ}$" if c in SEEN else "")
    n = [len(cs) for _, cs in GROUPS]
    spec = "@{}l" + "".join("c" * k + "@{\\hspace{9pt}}" for k in n) + "c@{}"
    a = ["\\begin{table*}[t]", "\\centering", CAPTION, "\\label{tab:action_probing}", "\\setlength{\\tabcolsep}{4.5pt}",
         "\\renewcommand{\\arraystretch}{1.12}", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    a.append(" & " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{\\textit{{{g}}}}}" for (g, cs), k in zip(GROUPS, n)) + " & \\\\")
    pos, rules = 2, []
    for k in n:
        rules.append(f"\\cmidrule(lr){{{pos}-{pos + k - 1}}}")
        pos += k
    a.append(" ".join(rules))
    a.append("\\textbf{Representation} & " + " & ".join(head(c) for c in cols) + " \\\\")
    a.append("\\midrule")
    for k, label in METHODS:
        pre = "\\rowcolor{oursblue}\n" if k == "justact" else ""
        lab = f"\\textcolor{{gray}}{{{label}}}" if k == "random" else label
        a.append(pre + lab + " & " + " & ".join(fmt(c, k) for c in cols) + " \\\\")
        if k in ("random", "dreamdojo_lam_retrained"):
            a.append("\\midrule")
    a.append("\\multicolumn{1}{@{}r}{\\footnotesize $\\Delta$} & " + " & ".join(f"\\footnotesize {delta(c)}" for c in cols) + " \\\\")
    a += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(a) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    E, S = load(args.results, args.seeds)
    tex = render(E)
    if args.out:
        open(args.out, "w").write(tex)
    print(tex)
    print("largest probe-seed standard deviation:", round(max(v for c in S for v in S[c].values()), 4))


if __name__ == "__main__":
    main()

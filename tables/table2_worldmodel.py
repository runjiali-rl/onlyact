"""Table 2 (action-conditioned world model) from the scores written by worldmodel/score.py.

Self cells report the mean PSNR (dB) of the rollouts against the real clip; transfer cells report the mean cosine between
the optical-flow direction of the generated video and of the source clip. The LaTeX uses the paper macros \\best, \\second,
\\mname and the colours oursblue, deltapos, deltaneg.

    python tables/table2_worldmodel.py --scores runs/table2/scores.json --out table2.tex
"""
import argparse
import json

COLS = [("libero_self", "psnr"), ("libero_transfer", "direction"), ("rlbench_self", "psnr"), ("rlbench_transfer", "direction"),
        ("spatialvid_self", "psnr"), ("spatialvid_transfer", "direction"), ("nuscenes_dev_self", "psnr"),
        ("nuscenes_dev_transfer", "direction"), ("nuscenes_conf_self", "psnr"), ("nuscenes_conf_transfer", "direction")]
METHODS = [("no_action", "No action"), ("adaworld_lam", "AdaWorld$^\\dagger$~\\cite{gao2025adaworld}"),
           ("dreamdojo_lam", "DreamDojo$^\\dagger$~\\cite{gao2026dreamdojo}"), ("onlyact", "\\textbf{\\mname} (ours)")]
RANKED = ["adaworld_lam", "dreamdojo_lam", "onlyact"]
CAP = ("\\caption{\\textbf{Action-conditioned world model.} The same world model (AdaWorld backbone, same recipe and data) is trained with each "
       "action representation. \\emph{Self}: PSNR$\\uparrow$ (dB) when driven by the clip's own actions. \\emph{Transfer}: a source clip's actions "
       "drive a different scene; Dir$\\uparrow$ is the cosine between the optical-flow direction of the generated video and of the source (0: unrelated; "
       "1: same). $^{\\circ}$Seen by our encoder without action labels. $^\\dagger$Retrained on our data. $\\Delta$: ours minus the best baseline. "
       "\\best{Bold}: best; \\second{underline}: second best (no-action row excluded).}")
GROUPS = [("Robot manipulation", [("LIBERO$^{\\circ}$", 2), ("RLBench", 2)]), ("Camera navigation", [("SpatialVID$^{\\circ}$", 2)]),
          ("Driving", [("nuScenes dev.", 2), ("nuScenes conf.", 2)])]


def render(V):
    psnr = lambda c: c.endswith("_self")
    num = lambda c, x: (f"{x:.2f}" if psnr(c) else f"{x:.3f}").replace("-", "$-$")

    def fmt(c, m):
        if m == "no_action":
            return f"\\textcolor{{gray}}{{{num(c, V[c][m])}}}"
        order = sorted(RANKED, key=lambda a: -V[c][a])
        s = num(c, V[c][m])
        return f"\\best{{{s}}}" if m == order[0] else f"\\second{{{s}}}" if m == order[1] else s

    def delta(c):
        d = V[c]["onlyact"] - max(V[c]["adaworld_lam"], V[c]["dreamdojo_lam"])
        s = f"{abs(d):.2f}" if psnr(c) else f"{abs(d):.3f}"
        return f"\\textcolor{{{'deltapos' if d >= 0 else 'deltaneg'}}}{{{'+' if d >= 0 else '$-$'}{s}}}"

    gn = [sum(k for _, k in ds) for _, ds in GROUPS]
    spec = "@{}l" + "".join("c" * k + ("@{\\hspace{10pt}}" if i < len(gn) - 1 else "") for i, k in enumerate(gn)) + "@{}"
    t = ["\\begin{table*}[t]", "\\centering", CAP, "\\label{tab:main_video_quality}", "\\setlength{\\tabcolsep}{5pt}",
         "\\renewcommand{\\arraystretch}{1.12}", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    t.append(" & " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{\\textit{{{g}}}}}" for (g, _), k in zip(GROUPS, gn)) + " \\\\")
    pos, r1 = 2, []
    for k in gn:
        r1.append(f"\\cmidrule(lr){{{pos}-{pos + k - 1}}}")
        pos += k
    t.append(" ".join(r1))
    t.append(" & " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{{name}}}" for _, ds in GROUPS for name, k in ds) + " \\\\")
    pos, r2 = 2, []
    for _, ds in GROUPS:
        for _, k in ds:
            r2.append(f"\\cmidrule(lr){{{pos}-{pos + k - 1}}}")
            pos += k
    t.append(" ".join(r2))
    t.append("\\textbf{Action representation} & " + " & ".join("{\\small Self PSNR$\\uparrow$}" if psnr(c) else "{\\small Transfer Dir$\\uparrow$}"
                                                              for c, _ in COLS) + " \\\\")
    t.append("\\midrule")
    for m, label in METHODS:
        pre = "\\rowcolor{oursblue}\n" if m == "onlyact" else ""
        lab = f"\\textcolor{{gray}}{{{label}}}" if m == "no_action" else label
        t.append(pre + lab + " & " + " & ".join(fmt(c, m) for c, _ in COLS) + " \\\\")
        if m in ("no_action", "dreamdojo_lam"):
            t.append("\\midrule")
    t.append("\\multicolumn{1}{@{}r}{\\footnotesize $\\Delta$} & " + " & ".join(f"\\footnotesize {delta(c)}" for c, _ in COLS) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(t) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="scores.json from worldmodel/score.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cells = json.load(open(a.scores))["cells"]
    V = {c: {m: cells[c][m][key] for m, _ in METHODS} for c, key in COLS}
    tex = render(V)
    if a.out:
        open(a.out, "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()

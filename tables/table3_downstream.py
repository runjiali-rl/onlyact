"""Table 3 (downstream world-model fine-tuning) from the source file written by `python -m downstream.table`.

(a) Controllability after 800 updates: Follow and Control per domain, with the "Real frames" row. (b) Fidelity: PSNR after
0, 200 and 800 updates. Delta = JustAct minus the better baseline; grey when its 95 % paired bootstrap interval contains 0.
The LaTeX uses the paper macros \\best, \\second, \\mname and the colours oursblue, deltapos, deltaneg.

    python tables/table3_downstream.py --source runs/table3/table3_source.json --out table3.tex
"""
import argparse
import json

DS = ["dmlab", "minecraft", "habitat", "nuscenes"]
STEPS = [0, 200, 800]
ARMS = ["adaworld_lam_retrained", "dreamdojo_lam_retrained", "justact"]
NAMES = {"dmlab": "DMLab", "minecraft": "Minecraft", "habitat": "Habitat", "nuscenes": "nuScenes"}
LAB = {"adaworld_lam_retrained": "AdaWorld$^\\dagger$~\\cite{gao2025adaworld}", "dreamdojo_lam_retrained": "DreamDojo$^\\dagger$~\\cite{gao2026dreamdojo}",
       "justact": "\\textbf{\\mname} (ours)"}
CAP = ("\\caption{\\textbf{Downstream world-model fine-tuning: controllability and fidelity.} Each pretrained world model from "
       "Table~\\ref{tab:main_video_quality} is fine-tuned on a small labelled set of a new domain. \\textbf{(a) Controllability} after 800 updates. "
       "Follow$\\uparrow$: accuracy of an inverse-dynamics classifier at recognising the commanded action in the generated frame (chance 0.333; "
       "\\emph{Real frames}: the same classifier on real frames); on nuScenes, Spearman correlation between commanded and generated forward motion "
       "(chance 0). Control$\\uparrow$: PSNR under the commanded action minus under other actions (dB; 0 = action ignored). "
       "\\textbf{(b) Fidelity}: PSNR$\\uparrow$ (dB) after 0 (before fine-tuning), 200 and 800 updates. Markers as in Table~\\ref{tab:main_video_quality}. "
       "$\\Delta$: ours minus the best baseline; grey when the 95\\% paired bootstrap interval contains 0. \\best{Bold}: best; \\second{underline}: second best.}")


def render(src):
    V, C, CEIL = src["values"], src["contrasts"], src["follow_ceiling_real_frames"]

    def panel(cols):
        out, dl = {a: [] for a in ARMS}, []
        for ds, m, s in cols:
            key = f"{ds}/{m}/step{s}"
            vals = {a: V[a][key] for a in ARMS}
            dec = 3 if m == "follow" else 2
            shown = sorted({round(v, dec) for v in vals.values()}, reverse=True)  # ties at the printed precision share a mark
            for a in ARMS:
                t = f"{vals[a]:.{dec}f}".replace("-", "$-$")
                r = shown.index(round(vals[a], dec))
                out[a].append(f"\\best{{{t}}}" if r == 0 else (f"\\second{{{t}}}" if r == 1 else t))
            opp = max(ARMS[:2], key=lambda a: vals[a])
            ci, d = C[key]["ci95"], vals["justact"] - vals[opp]
            col = "gray" if ci[0] <= 0 <= ci[1] else ("deltapos" if d >= 0 else "deltaneg")
            sign = "" if round(abs(d), dec) == 0 else ("+" if d > 0 else "$-$")
            dl.append(f"\\footnotesize \\textcolor{{{col}}}{{{sign}{abs(d):.{dec}f}}}")
        return out, dl

    oa, da = panel([(ds, m, 800) for ds in DS for m in ("follow", "margin")])
    ob, db = panel([(ds, "psnr", s) for ds in DS for s in STEPS])
    span_a = lambda xs: " & ".join(x if i % 2 == 0 else f"\\multicolumn{{2}}{{c}}{{{x}}}" for i, x in enumerate(xs))
    gray = lambda x: f"\\textcolor{{gray}}{{{x}}}"
    t = ["\\begin{table*}[t]", "\\centering", CAP, "\\label{tab:downstream_finetune}", "\\setlength{\\tabcolsep}{4pt}",
         "\\renewcommand{\\arraystretch}{1.12}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}l" + "ccc@{\\hspace{8pt}}" * 3 + "ccc@{}}",
         "\\toprule", " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{NAMES[ds]}}}" for ds in DS) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(4)),
         "\\multicolumn{13}{@{}l}{\\textbf{(a) Controllability} after 800 updates} \\\\",
         "\\textbf{Action representation} & " + span_a(["{\\small Follow$\\uparrow$}", "{\\small Control$\\uparrow$}"] * 4) + " \\\\", "\\midrule"]
    t.append(gray("Real frames") + " & " + span_a(sum(([gray(f"{CEIL[ds]:.3f}") if ds in CEIL else gray("--"), gray("--")] for ds in DS), [])) + " \\\\")
    for a in ARMS[:2]:
        t.append(LAB[a] + " & " + span_a(oa[a]) + " \\\\")
    t += ["\\midrule", "\\rowcolor{oursblue}", LAB["justact"] + " & " + span_a(oa["justact"]) + " \\\\",
          "\\multicolumn{1}{@{}r}{\\footnotesize $\\Delta$} & " + span_a(da) + " \\\\", "\\midrule\\midrule",
          "\\multicolumn{13}{@{}l}{\\textbf{(b) Fidelity}, PSNR$\\uparrow$ (dB) after 0 / 200 / 800 updates} \\\\",
          "\\textbf{Action representation} & " + " & ".join(["{\\small 0}", "{\\small 200}", "{\\small 800}"] * 4) + " \\\\", "\\midrule"]
    t += [LAB[a] + " & " + " & ".join(ob[a]) + " \\\\" for a in ARMS[:2]]
    t += ["\\midrule", "\\rowcolor{oursblue}", LAB["justact"] + " & " + " & ".join(ob["justact"]) + " \\\\",
          "\\multicolumn{1}{@{}r}{\\footnotesize $\\Delta$} & " + " & ".join(db) + " \\\\", "\\bottomrule"]
    t += ["\\end{tabular}}", "\\end{table*}"]
    return "\n".join(t) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="table3_source.json from `python -m downstream.table`")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tex = render(json.load(open(a.source)))
    if a.out:
        open(a.out, "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()

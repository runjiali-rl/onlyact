"""Table 4 (scene leakage) and its appendix table (per-dataset SSIM and L1) from the decoder result files.

Reads every <results_root>/**/result.json written by leakage/train.py (one per dataset x representation x seed) and writes
    <out_dir>/table4_leakage.tex            Table 4: per-dataset held-out LPIPS, then LPIPS / SSIM / L1 averaged over the 13 datasets
    <out_dir>/table4_leakage_appendix.tex   appendix table: per-dataset held-out SSIM and L1
    <out_dir>/table4_leakage_source.json    the seed means behind both tables

Each cell is the mean over decoder seeds of the last epoch's validation value. A seed counts only if the run finished all of
its epochs and held out the same scenes as the V-JEPA 2 (frozen) run of the same dataset and seed. "Average" is the
unweighted mean of the 13 per-dataset means. Higher LPIPS and L1 and lower SSIM mean that less of the scene can be recovered
from the representation. Bold / underline and Delta (ours minus the best other row) are computed on unrounded values.
The LaTeX uses the paper's macros \\best, \\second, \\mname and the colours oursblue, deltapos and deltaneg.

    python tables/table4_leakage.py --results_root runs/table4 --out_dir runs/table4/tables
"""
import argparse
import json
import os

SEEN = {"libero", "droid", "spatialvid_nav"}  # datasets used to train JustAct (marked with a circle)
GROUPS = [("Robot manipulation", ["libero", "droid", "rlbench", "robocasa", "droid_iprl", "metaworld", "utaustin_mutex", "vp2_robosuite", "vp2_robodesk"]),
          ("Camera navigation", ["spatialvid_nav", "arkitscenes", "realestate10k"]), ("Game", ["procgen_plan"])]
NAME = {"libero": "LIBERO", "droid": "DROID", "rlbench": "RLBench", "robocasa": "RoboCasa", "droid_iprl": "DROID-IPRL", "metaworld": "MetaWorld",
        "utaustin_mutex": "MUTEX", "vp2_robosuite": "Robosuite", "vp2_robodesk": "RoboDesk", "spatialvid_nav": "SpatialVID",
        "arkitscenes": "ARKitScenes", "realestate10k": "RE10K", "procgen_plan": "Procgen"}
METHODS = [("adaworld_lam_retrained", "AdaWorld$^\\dagger$~\\cite{gao2025adaworld}"),
           ("dreamdojo_lam_retrained", "DreamDojo$^\\dagger$~\\cite{gao2026dreamdojo}"),
           ("vjepa2_frozen", "V-JEPA~2~\\cite{vjepa2}"), ("justact", "\\textbf{\\mname} (ours)")]
OURS, REFERENCE = "justact", "vjepa2_frozen"  # REFERENCE: run whose held-out scenes every other row must share
OTHERS = [m for m, _ in METHODS if m != OURS]
DS = [c for _, cs in GROUPS for c in cs]
SIGN = {"lpips": 1, "ssim": -1, "l1": 1}  # +1: higher = less scene leakage


def load_results(root):
    res = {}
    for dirpath, _, files in os.walk(root):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json")) as f:
                r = json.load(f)
            key = (r["dataset"], r["representation"], int(r["seed"]))
            if key in res:
                raise SystemExit(f"two result files for {key}")
            res[key] = r
    return res


def complete(r):
    return r is not None and r["epochs_completed"] >= r["epochs"]


def seed_means(res, seeds):
    """{dataset: {representation: {lpips, ssim, l1 (means over the valid seeds), n (number of valid seeds)}}}"""
    L, missing = {}, []
    for c in DS:
        L[c] = {}
        for k, _ in METHODS:
            per = []
            for s in seeds:
                r, z = res.get((c, k, s)), res.get((c, REFERENCE, s))
                if not (complete(r) and complete(z)) or r["split_md5"] != z["split_md5"]:
                    continue
                per.append(r["final_val"])
            if not per:
                missing.append(f"{c}/{k}")
                continue
            m = {q: sum(p[q] for p in per) / len(per) for q in ("lpips", "ssim", "l1")}
            m["n"] = len(per)
            if m["n"] < len(seeds):
                print(f"warning: {c} / {k}: {m['n']} of {len(seeds)} seeds usable")
            L[c][k] = m
    if missing:
        raise SystemExit("no complete run with the reference split for: " + ", ".join(missing))
    return L


def val(L, col, k):
    c, q = col
    return sum(L[d][k][q] for d in DS) / len(DS) if c == "avg" else L[c][k][q]


def fmt(L, col, k):
    order = sorted([m for m, _ in METHODS], key=lambda m: -SIGN[col[1]] * val(L, col, m))
    s = f"{val(L, col, k):.3f}"
    return f"\\best{{{s}}}" if k == order[0] else f"\\second{{{s}}}" if k == order[1] else s


def delta(L, col):
    o, best = val(L, col, OURS), max((val(L, col, m) for m in OTHERS), key=lambda v: SIGN[col[1]] * v)
    d = o - best
    good = SIGN[col[1]] * d >= 0
    return f"\\textcolor{{{'deltapos' if good else 'deltaneg'}}}{{{'+' if d >= 0 else '$-$'}{abs(d):.3f}}}"


def head(c):
    return NAME[c] + ("$^{\\circ}$" if c in SEEN else "")


def group_header(t, extra):
    """Group row + cmidrules; `extra` = (title, number of columns) of a trailing group or None."""
    n = [len(cs) for _, cs in GROUPS]
    cells = [f"\\multicolumn{{{k}}}{{c}}{{\\textit{{{g}}}}}" for (g, _), k in zip(GROUPS, n)]
    if extra:
        cells.append(f"\\multicolumn{{{extra[1]}}}{{c}}{{\\textit{{{extra[0]}}}}}")
        n = n + [extra[1]]
    t.append(" & " + " & ".join(cells) + " \\\\")
    pos, rules = 2, []
    for k in n:
        rules.append(f"\\cmidrule(lr){{{pos}-{pos + k - 1}}}")
        pos += k
    t.append(" ".join(rules))


def colspec(n):
    return "@{}l" + "".join("c" * k + ("@{\\hspace{9pt}}" if i < len(n) - 1 else "") for i, k in enumerate(n)) + "@{}"


def method_rows(L, t, cols):
    for k, label in METHODS:
        pre = "\\rowcolor{oursblue}\n" if k == OURS else ""
        t.append(pre + label + " & " + " & ".join(fmt(L, col, k) for col in cols) + " \\\\")
        if k == "dreamdojo_lam_retrained":
            t.append("\\midrule")
    t.append("\\multicolumn{1}{@{}r}{\\footnotesize $\\Delta$} & " + " & ".join(f"\\footnotesize {delta(L, col)}" for col in cols) + " \\\\")


CAPTION = ("\\caption{\\textbf{Scene leakage.} A decoder with the same architecture for every representation (trained with L1 + 0.5$\\,\\times\\,$LPIPS, "
           "10 epochs, $128{\\times}128$) reconstructs the post-action frame from the representation and is evaluated on scenes held out from its "
           "training; values are averaged over 4 decoder seeds. Per-dataset columns: held-out LPIPS$\\uparrow$ (higher = less scene information "
           "recoverable). Last three columns: LPIPS$\\uparrow$, SSIM$\\downarrow$ and L1$\\uparrow$ averaged over the 13 datasets; per-dataset SSIM and L1 "
           "are in the appendix. A representation that carries no information at all also leaks nothing, so read this table together with "
           "Table~\\ref{tab:action_probing}. Markers and baselines as in Table~\\ref{tab:action_probing}. $\\Delta$: ours minus the best other method "
           "(green: ours leaks less). \\best{Bold}: least leakage; \\second{underline}: second least.}")

CAPTION_APPENDIX = ("\\caption{\\textbf{Scene leakage: per-dataset SSIM and L1.} Same decoders, held-out scenes and 4 decoder seeds as "
                    "Table~\\ref{tab:semantic_leakage}; the Average column repeats its last two columns. SSIM$\\downarrow$: lower = less "
                    "scene information recoverable; L1$\\uparrow$: higher = less. Markers and baselines as in Table~\\ref{tab:action_probing}. "
                    "$\\Delta$: ours minus the best other method (green: ours leaks less). \\best{Bold}: least leakage; "
                    "\\second{underline}: second least.}")


def main_table(L):
    n = [len(cs) for _, cs in GROUPS] + [3]
    cols = [(c, "lpips") for c in DS] + [("avg", "lpips"), ("avg", "ssim"), ("avg", "l1")]
    t = ["\\begin{table*}[t]", "\\centering", CAPTION, "\\label{tab:semantic_leakage}", "\\setlength{\\tabcolsep}{4.5pt}",
         "\\renewcommand{\\arraystretch}{1.12}", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec(n)}}}", "\\toprule"]
    group_header(t, ("Average", 3))
    t.append("\\textbf{Representation} & " + " & ".join(head(c) for c in DS) + " & LPIPS$\\uparrow$ & SSIM$\\downarrow$ & L1$\\uparrow$ \\\\")
    t.append("\\midrule")
    method_rows(L, t, cols)
    t += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(t) + "\n"


def appendix_table(L):
    n = [len(cs) for _, cs in GROUPS] + [1]
    ncol = 1 + sum(n)
    t = ["\\begin{table*}[t]", "\\centering", CAPTION_APPENDIX, "\\label{tab:semantic_leakage_ssim_l1}", "\\setlength{\\tabcolsep}{4.5pt}",
         "\\renewcommand{\\arraystretch}{1.12}", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec(n)}}}", "\\toprule"]
    group_header(t, None)
    t.append("\\textbf{Representation} & " + " & ".join(head(c) for c in DS) + " & Average \\\\")
    for q, title in (("ssim", "SSIM$\\downarrow$"), ("l1", "L1$\\uparrow$")):
        t.append("\\midrule")
        t.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        t.append("\\midrule")
        method_rows(L, t, [(c, q) for c in DS] + [("avg", q)])
    t += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(t) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", required=True, help="folder searched recursively for result.json files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    L = seed_means(load_results(args.results_root), args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "table4_leakage_source.json"), "w") as f:
        json.dump(L, f, indent=1)
    with open(os.path.join(args.out_dir, "table4_leakage.tex"), "w") as f:
        f.write(main_table(L))
    with open(os.path.join(args.out_dir, "table4_leakage_appendix.tex"), "w") as f:
        f.write(appendix_table(L))
    for k, _ in METHODS:
        print(f"{k:24s}", {q: round(val(L, ("avg", q), k), 3) for q in ("lpips", "ssim", "l1")})
    print("wrote", os.path.join(args.out_dir, "table4_leakage.tex"), "and table4_leakage_appendix.tex")


if __name__ == "__main__":
    main()

"""Collect the four per-domain score files into the numbers of paper Table 3 (no inference).

Panel (a), after 800 updates: Follow and Control per domain; "Real frames" = the classifier on real transitions.
Panel (b): PSNR after 0, 200 and 800 updates. Delta = OnlyAct minus the better baseline in that column; it is marked as not
significant (grey in the paper) when the paired bootstrap 95 % interval of that difference contains 0.

  python -m downstream.table --scores runs/table3/scores --out runs/table3/table3_source.json
"""
from __future__ import annotations

import argparse
import json
import os

from .common import ARMS, BASELINES, OURS, write_json

DOMAINS = ["dmlab", "minecraft", "habitat", "nuscenes"]
STEPS = [0, 200, 800]
KEYS = {"psnr": ("psnr", "psnr"), "margin": ("m3_margin", "m3"), "follow": ("m1_action_following", "m1")}
NUSCENES_KEYS = {"psnr": ("psnr", "psnr"), "margin": ("mc4_psnr", "mc4_psnr"), "follow": ("follow", "follow")}


def build(scores):
    keys = lambda ds: NUSCENES_KEYS if ds == "nuscenes" else KEYS
    value = lambda ds, arm, m, s: scores[ds]["cells"][f"{arm}/step{s}"][keys(ds)[m][0]]
    V = {a: {} for a in ARMS}
    for a in ARMS:
        for ds in DOMAINS:
            V[a][(ds, "follow", 800)] = value(ds, a, "follow", 800); V[a][(ds, "margin", 800)] = value(ds, a, "margin", 800)
            for s in STEPS: V[a][(ds, "psnr", s)] = value(ds, a, "psnr", s)
    src = {"values": {a: {f"{ds}/{m}/step{s}": v for (ds, m, s), v in V[a].items()} for a in ARMS},
           "follow_ceiling_real_frames": {ds: scores[ds]["ceiling"]["idm_real_test_accuracy"] for ds in DOMAINS if "ceiling" in scores[ds]},
           "contrasts": {}}
    cols = [(ds, m, 800) for ds in DOMAINS for m in ("follow", "margin")] + [(ds, "psnr", s) for ds in DOMAINS for s in STEPS]
    for ds, m, s in cols:
        opp = max(BASELINES, key=lambda a: V[a][(ds, m, s)])
        c = scores[ds]["contrasts"][f"step{s}/{OURS}-{opp}"][keys(ds)[m][1]]
        src["contrasts"][f"{ds}/{m}/step{s}"] = {"vs": opp, "mean": c["mean"], "ci95": c["ci95"],
                                                 "significant": not (c["ci95"][0] <= 0 <= c["ci95"][1])}
    return src


def show(src):
    names = {"adaworld_lam_retrained": "AdaWorld-LAM (retrained)", "dreamdojo_lam_retrained": "DreamDojo-LAM (retrained)", "onlyact": "OnlyAct"}
    cols_a = [(ds, m) for ds in DOMAINS for m in ("follow", "margin")]
    print("(a) after 800 updates: " + " | ".join(f"{ds} {m}" for ds, m in cols_a))
    print(f"{'Real frames':28s}" + "".join(f"{src['follow_ceiling_real_frames'].get(ds, float('nan')):9.3f}{'--':>9s}" for ds in DOMAINS))
    for a in ARMS:
        print(f"{names[a]:28s}" + "".join(f"{src['values'][a][f'{ds}/{m}/step800']:9.3f}" for ds, m in cols_a))
    print("(b) PSNR after 0 / 200 / 800 updates")
    for a in ARMS:
        print(f"{names[a]:28s}" + "".join(f"{src['values'][a][f'{ds}/psnr/step{s}']:8.2f}" for ds in DOMAINS for s in STEPS))
    for k, c in src["contrasts"].items():
        print(f"Delta {k:26s} vs {c['vs']:24s} {c['mean']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{'' if c['significant'] else '  (interval contains 0)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="folder with dmlab.json, minecraft.json, habitat.json, nuscenes.json from score.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scores = {ds: json.load(open(os.path.join(a.scores, f"{ds}.json"))) for ds in DOMAINS}
    src = build(scores); write_json(a.out, src); show(src); print("wrote", a.out)


if __name__ == "__main__":
    main()

"""nuScenes Control metric (Table 3): commanded-action PSNR minus the PSNR under four random training-clip actions.

Each test case draws four indices into the bank of the 100 training clips' actions (the draw is fixed by the case seed,
see generate.py). Every draw keeps its weight: a draw equal to the factual action reuses the factual prediction (margin
0), and repeated draws reuse one prediction. Control = mean over the four draws of (factual PSNR - draw PSNR).
"""
from __future__ import annotations

import math
import statistics

from .common import sampling_seed


def _action(value):
    assert type(value) in (tuple, list) and len(value) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in value)
    return tuple(value)


def make_case(task, bank, indices):
    """Describe the four draws of one case and the distinct new actions that must be generated (no random numbers here)."""
    assert len(indices) == 4 and len(bank) > 0
    factual = _action(task["actions_ego_xy"][5])
    unique, draws = [], []
    for i, bank_index in enumerate(indices):
        alternative = _action(bank[bank_index]["raw_action"])
        reuse_factual = alternative == factual
        prediction_index = None
        if not reuse_factual:
            if alternative not in unique:
                unique.append(alternative)
            prediction_index = unique.index(alternative)
        draws.append({"draw_index": i, "bank_index": bank_index, "raw_action": list(alternative),
                      "reuse_factual_correct": reuse_factual, "unique_prediction_index": prediction_index})
    return {"task_id": task["task_id"], "log_token": task["log_token"], "target_index": 6, "action_index": 5,
            "factual_action": list(factual), "sampling_seed": sampling_seed(task["task_id"]), "draws": draws,
            "unique_new_actions": [list(x) for x in unique]}


def empirical_control(case, correct, alternatives):
    """Four-draw margins of one case; `alternatives` are the scores of the distinct new actions in make_case order."""
    truth = {k: correct[k] for k in ("psnr", "lpips")}
    assert len(case["draws"]) == 4 and len(alternatives) == len(case["unique_new_actions"])
    checked = []
    for i, (expected, row) in enumerate(zip(case["unique_new_actions"], alternatives)):
        assert row["unique_prediction_index"] == i and _action(row["raw_action"]) == _action(expected)
        checked.append({k: row["metrics"][k] for k in ("psnr", "lpips")})
    draw_metrics = [truth if d["reuse_factual_correct"] else checked[d["unique_prediction_index"]] for d in case["draws"]]
    psnr_margin = [truth["psnr"] - d["psnr"] for d in draw_metrics]
    lpips_margin = [d["lpips"] - truth["lpips"] for d in draw_metrics]
    win = lambda xs: statistics.mean(.5 if x == 0 else float(x > 0) for x in xs)
    return {"correct_PSNR": truth["psnr"], "correct_LPIPS": truth["lpips"],
            "empirical_delta1_psnr_mc4": statistics.mean(psnr_margin), "empirical_delta1_lpips_mc4": statistics.mean(lpips_margin),
            "mean_random_draw_PSNR": statistics.mean(d["psnr"] for d in draw_metrics),
            "mean_random_draw_LPIPS": statistics.mean(d["lpips"] for d in draw_metrics),
            "minimum_PSNR_margin_among_four_draws_including_factual_mass": min(psnr_margin),
            "minimum_LPIPS_margin_among_four_draws_including_factual_mass": min(lpips_margin),
            "pairwise_PSNR_correct_over_random_ties_half": win(psnr_margin),
            "pairwise_LPIPS_correct_over_random_ties_half": win(lpips_margin)}

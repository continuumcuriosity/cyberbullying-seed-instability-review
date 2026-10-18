#!/usr/bin/env python3
"""Recompute rank-based partial correlations for the seed-instability paper.

Run from anywhere:
    python analysis_repair/partial_spearman.py
or supply --repo-root PATH.

For each model family and test set, the script:
1. computes each run's harmful recall and number flagged from its saved probabilities;
2. rank-transforms macro-F1, recall, and flag count (average ranks for ties);
3. linearly residualizes the first two rank vectors against ranked flag count;
4. correlates the residuals and tests with t(df=n-3);
5. applies Holm correction to the 12 partial-correlation p-values.

The source repository is read only. Results are written beside this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t


ARMS = ("main", "soft_uw1", "soft_off", "no_contrastive")
SEEDS = (0, 1, 2)
DATASETS = ("youtube", "wiki_person", "cad_person", "cad_identity", "stormfront", "dynahate")
FAMILIES = ("bert", "roberta")


def residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(control)), control))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Pearson correlation of rank residuals; two-sided t test with n-3 df."""
    xr, yr, zr = (rankdata(v, method="average") for v in (x, y, z))
    rx = residualize(xr, zr)
    ry = residualize(yr, zr)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df = len(x) - 3
    statistic = abs(rho) * np.sqrt(df / max(1e-15, 1.0 - rho * rho))
    return rho, float(2.0 * t.sf(statistic, df))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def read_ledger(repo: Path, family: str) -> dict[tuple[str, int], float]:
    path = repo / "data" / ("ledger12.csv" if family == "bert" else "ledger_roberta.csv")
    ledger = pd.read_csv(path)
    return {(row.arm, int(row.seed)): float(row.test_macro_f1)
            for row in ledger.itertuples(index=False)}


def read_run(repo: Path, family: str, dataset: str, arm: str, seed: int) -> tuple[float, float]:
    folder = "probs" if family == "bert" else "probs_roberta"
    path = repo / "external" / folder / dataset / f"{arm}_{seed}.csv"
    frame = pd.read_csv(path)
    required = {"row_id", "gold", "p_hate", "p_offensive", "p_normal"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    labels = frame["gold"].to_numpy(dtype=int)
    flags = frame[["p_hate", "p_offensive", "p_normal"]].to_numpy().argmax(axis=1) != 2
    positive = labels == 1
    return float(flags.sum()), float(flags[positive].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    repo = args.repo_root.resolve()

    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        ledger = read_ledger(repo, family)
        for dataset in DATASETS:
            x, y, z = [], [], []
            for arm in ARMS:
                for seed in SEEDS:
                    flags, recall = read_run(repo, family, dataset, arm, seed)
                    x.append(ledger[(arm, seed)])
                    y.append(recall)
                    z.append(flags)
            rho, p = partial_spearman(np.asarray(x), np.asarray(y), np.asarray(z))
            raw = spearmanr(x, y)
            rows.append({
                "family": family,
                "dataset": dataset,
                "nruns": len(x),
                "flags_min": int(min(z)),
                "flags_max": int(max(z)),
                "recall_min": min(y),
                "recall_max": max(y),
                "rho_raw_spearman": float(raw.statistic),
                "p_raw_spearman": float(raw.pvalue),
                "rho_partial_spearman": rho,
                "p_partial_two_sided": p,
            })

    adjusted = holm_adjust([float(row["p_partial_two_sided"]) for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_partial_holm_12"] = p_adjusted

    output = Path(__file__).resolve().parent / "partial_spearman_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.6f")
    print(f"Wrote {output}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()


"""
Plotting utilities for oracle_bsuite/train.py results.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "ppo"                   : "#9E9E9E",
    "ppo_oracle"            : "#FF9800",
    "ppo_privileged"        : "#2196F3",
    "ppo_privileged_oracle" : "#4CAF50",
}

LABELS = {
    "ppo"                   : "PPO (baseline)",
    "ppo_oracle"            : "PPO + Oracle (credit fix)",
    "ppo_privileged"        : "PPO + Privileged (memory fix)",
    "ppo_privileged_oracle" : "PPO + Privileged + Oracle (ceiling)",
}


def plot_results(results_dir: Path, out: Optional[str] = None) -> None:
    results_dir = Path(results_dir)
    data: Dict[str, List] = {}

    for jf in sorted((results_dir / "results").glob("*.json")):
        with open(jf) as f:
            r = json.load(f)
        mode = r.get("mode", "?")
        data.setdefault(mode, []).append(r)

    if not data:
        print(f"[plot] No result JSONs in {results_dir / 'results'}")
        return

    modes = [m for m in LABELS if m in data]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"oracle_bsuite — DiscountingChain Credit Assignment Isolation",
        fontsize=13, fontweight="bold",
    )

    # ── Left: learning curves ─────────────────────────────────────────────────
    ax = axes[0]
    for mode in modes:
        runs   = data[mode]
        all_ts = [np.array(r["timesteps"])    for r in runs if r["timesteps"]]
        all_rw = [np.array(r["mean_rewards"]) for r in runs if r["timesteps"]]
        if not all_ts:
            continue
        max_ts = max(ts.max() for ts in all_ts)
        x      = np.linspace(0, max_ts, 300)
        interped = [np.interp(x, ts, rw) for ts, rw in zip(all_ts, all_rw)]
        arr    = np.array(interped)
        mean   = arr.mean(axis=0)
        se     = arr.std(axis=0) / max(np.sqrt(len(interped)), 1)
        c      = COLORS.get(mode, "black")
        ax.plot(x, mean, color=c, lw=2, label=LABELS.get(mode, mode))
        ax.fill_between(x, mean - se, mean + se, alpha=0.15, color=c)

    ax.axhline(0.5, color="black",   ls="--", lw=1.2, alpha=0.5, label="Chance (0.5)")
    ax.axhline(0.8, color="crimson", ls="--", lw=1.5, alpha=0.7, label="Gate (0.8)")
    ax.set_xlabel("Environment steps", fontsize=11)
    ax.set_ylabel("Mean eval reward",  fontsize=11)
    ax.set_title("Learning Curves", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # ── Right: final performance bar chart ────────────────────────────────────
    ax2   = axes[1]
    x_pos = np.arange(len(modes))
    means, stds = [], []
    for mode in modes:
        finals = [r["final_mean"] for r in data[mode]]
        means.append(np.mean(finals))
        stds.append(np.std(finals))

    bars = ax2.bar(
        x_pos, means, yerr=stds,
        color=[COLORS.get(m, "grey") for m in modes],
        capsize=5, alpha=0.85,
    )
    ax2.axhline(0.5, color="black",   ls="--", lw=1.2, alpha=0.5)
    ax2.axhline(0.8, color="crimson", ls="--", lw=1.5, alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([LABELS.get(m, m) for m in modes], rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Final mean eval reward", fontsize=11)
    ax2.set_title("Final Performance (mean ± std)", fontsize=11)
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = out or str(results_dir / "results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close()

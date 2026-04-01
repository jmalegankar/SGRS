"""
Plotting utilities for Milestone 0 results.

Usage:
  python train.py --plot --results_dir results/milestone0
  python -m oracle.plot results/milestone0
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ppo"         : "#2196F3",   # blue
    "rppo"        : "#FF9800",   # orange
    "ppo_oracle"  : "#4CAF50",   # green
    "rppo_oracle" : "#E91E63",   # pink
}
LABELS = {
    "ppo"         : "PPO (baseline)",
    "rppo"        : "RecurrentPPO / LSTM (baseline)",
    "ppo_oracle"  : "PPO + Oracle shaping",
    "rppo_oracle" : "RecurrentPPO + Oracle shaping",
}
MODES_ORDER = ["ppo", "rppo", "ppo_oracle", "rppo_oracle"]


def plot_results(results_dir: Path, out: Optional[str] = None) -> None:
    results_dir = Path(results_dir)
    per_mode: Dict[str, List[Dict]] = defaultdict(list)

    for jf in sorted((results_dir / "results").glob("*.json")):
        parts = jf.stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        mode = parts[0]
        with open(jf) as f:
            per_mode[mode].append(json.load(f))

    if not per_mode:
        print(f"[plot] No result JSONs found in {results_dir / 'results'}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        "Milestone 0: Oracle Credit Assignment Test\nMiniGrid-MemoryS7-v0",
        fontsize=13, fontweight="bold",
    )

    # ── learning curves ───────────────────────────────────────────────────────
    ax = axes[0]
    for mode in MODES_ORDER:
        runs = per_mode.get(mode, [])
        if not runs:
            continue
        all_ts = [np.array(r["timesteps"]) for r in runs if r["timesteps"]]
        if not all_ts:
            continue
        max_ts = max(ts.max() for ts in all_ts)
        x = np.linspace(0, max_ts, 300)
        interped = []
        for r in runs:
            if not r["timesteps"]:
                continue
            ts = np.array(r["timesteps"])
            rw = np.array(r["mean_rewards"])
            interped.append(np.interp(x, ts, rw))
        arr  = np.array(interped)
        mean = arr.mean(axis=0)
        se   = arr.std(axis=0) / max(np.sqrt(len(interped)), 1)
        c    = COLORS.get(mode, "gray")
        ax.plot(x, mean, color=c, label=LABELS.get(mode, mode), linewidth=2)
        ax.fill_between(x, mean - se, mean + se, alpha=0.18, color=c)

    ax.axhline(0.50, color="black",   ls="--", lw=1.2, alpha=0.6, label="Chance (0.50)")
    ax.axhline(0.80, color="crimson", ls="--", lw=1.5, alpha=0.8, label="Gate (0.80)")
    ax.set_xlabel("Environment steps", fontsize=11)
    ax.set_ylabel("Mean episode reward  (eval env, no oracle)", fontsize=11)
    ax.set_title("Learning Curves  (mean ± SE across seeds)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.05, 1.10)
    ax.grid(True, alpha=0.3)

    # ── final performance bar chart ───────────────────────────────────────────
    ax2 = axes[1]
    bar_means, bar_stds, bar_colors, bar_labels = [], [], [], []
    for mode in MODES_ORDER:
        runs = per_mode.get(mode, [])
        if not runs:
            continue
        finals = [r["final_mean"] for r in runs]
        bar_means.append(np.mean(finals))
        bar_stds.append(np.std(finals))
        bar_colors.append(COLORS.get(mode, "gray"))
        bar_labels.append(LABELS.get(mode, mode))

    xp   = np.arange(len(bar_means))
    bars = ax2.bar(
        xp, bar_means, yerr=bar_stds,
        color=bar_colors, alpha=0.82,
        capsize=6, edgecolor="black", linewidth=0.8,
    )
    for bar, m in zip(bars, bar_means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{m:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    ax2.axhline(0.50, color="black",   ls="--", lw=1.2, alpha=0.6)
    ax2.axhline(0.80, color="crimson", ls="--", lw=1.5, alpha=0.8)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(bar_labels, rotation=12, ha="right", fontsize=9)
    ax2.set_ylabel("Final mean reward (last eval checkpoint)", fontsize=11)
    ax2.set_title("Final Performance  (mean ± std across seeds)", fontsize=11)
    ax2.set_ylim(-0.1, 1.25)
    ax2.grid(True, alpha=0.3, axis="y")

    for i, mode in enumerate(["ppo_oracle", "rppo_oracle"]):
        if mode not in per_mode:
            continue
        finals = [r["final_mean"] for r in per_mode[mode]]
        passed = np.mean(finals) >= 0.8
        ax2.text(
            0.98, 0.04 + i * 0.06,
            f"{'PASS' if passed else 'FAIL'} gate ({mode})",
            transform=ax2.transAxes, ha="right", fontsize=10,
            color="green" if passed else "red", fontweight="bold",
        )

    plt.tight_layout()
    out_path = out or str(results_dir / "milestone0_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_results(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/milestone0"))

#!/usr/bin/env python3
"""
oracle_bsuite — Horizon Scaling Experiment
==========================================
Tests the core SGRS thesis prediction on DiscountingChain:

  At short delays (≤3 steps), memory fix alone (privileged) reaches ceiling —
  the 15-step corridor in MiniGrid is analogous here.

  At longer delays (≥10 steps), credit assignment becomes the binding bottleneck:
  ppo_privileged degrades while ppo_privileged_oracle stays at ceiling.

  The crossover point is where SGRS's value proposition begins.

CONDITIONS (3 — sufficient for the scaling figure)
  ppo                    No fixes (chance floor)
  ppo_privileged         Memory fix only — isolates credit assignment bottleneck
  ppo_privileged_oracle  Both fixes — ceiling reference

CHAIN LENGTHS (mapping_seed 0-4 → reward delays [1, 3, 10, 30, 100] steps)
  D1   mapping_seed=0  delay=1    γ^1  ≈ 0.99
  D3   mapping_seed=1  delay=3    γ^3  ≈ 0.97
  D10  mapping_seed=2  delay=10   γ^10 ≈ 0.90
  D30  mapping_seed=3  delay=30   γ^30 ≈ 0.74
  D100 mapping_seed=4  delay=100  γ^100≈ 0.37

USAGE
  # Full scaling experiment (3 conditions × 5 delays × 5 seeds × 500k steps)
  python scale_experiment.py

  # Single condition for smoke test
  python scale_experiment.py --mode ppo_privileged --seeds 5 --steps 1000000

  # Plot after runs complete
  python scale_experiment.py --plot --results_dir results/bsuite_scaling
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    sys.exit("stable-baselines3 not found.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs import make_env_fn, make_eval_env_fn, REWARD_DELAYS

# ══════════════════════════════════════════════════════════════════════════════
# Registry and config
# ══════════════════════════════════════════════════════════════════════════════

CHAIN_REGISTRY = {
    # key    mapping_seed  delay   γ^delay
    "D1"  : (0,            1,      0.99**1),
    "D3"  : (1,            3,      0.99**3),
    "D10" : (2,            10,     0.99**10),
    "D30" : (3,            30,     0.99**30),
    "D100": (4,            100,    0.99**100),
}

SCALE_CONDITIONS = ["ppo", "ppo_privileged", "ppo_privileged_oracle"]

CONDITION_CFG = {
    # mode                      use_oracle  use_privileged
    "ppo"                     : (False,     False),
    "ppo_privileged"          : (False,     True),
    "ppo_privileged_oracle"   : (True,      True),
}

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.05,
    clip_range    = 0.2,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 0,
)

ORACLE_ALPHA     = 2.0
ORACLE_SHAPING_C = 1.0

COLORS = {
    "ppo"                   : "#9E9E9E",
    "ppo_privileged"        : "#2196F3",
    "ppo_privileged_oracle" : "#4CAF50",
}
LABELS = {
    "ppo"                   : "PPO (no fixes)",
    "ppo_privileged"        : "PPO + Privileged memory",
    "ppo_privileged_oracle" : "PPO + Privileged + Oracle (ceiling)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Metrics callback
# ══════════════════════════════════════════════════════════════════════════════

class ScaleCallback(BaseCallback):
    def __init__(self, log_every: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every  = log_every
        self._last_log  = 0
        self._cur_bonus : float       = 0.0
        self._ep_env    : List[float] = []
        self._correct   : List[bool]  = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._cur_bonus += info.get("oracle_bonus", 0.0)
            if "oracle_correct" in info:
                self._correct.append(info["oracle_correct"])
            if "episode" in info:
                self._ep_env.append(info["episode"]["r"] - self._cur_bonus)
                self._cur_bonus = 0.0

        if self.num_timesteps - self._last_log >= self.log_every:
            if self._correct:
                self.logger.record("oracle/correct_frac",
                                   float(np.mean(self._correct)))
            if self._ep_env:
                self.logger.record("oracle/ep_env_reward_mean",
                                   float(np.mean(self._ep_env)))
                cfrac = (f"  correct={np.mean(self._correct):.2f}"
                         if self._correct else "")
                print(f"  t={self.num_timesteps:>8,}"
                      f"  env_rew={np.mean(self._ep_env):.3f}"
                      f"{cfrac}  n_eps={len(self._ep_env)}")
            self._ep_env  = []
            self._correct = []
            self._last_log = self.num_timesteps
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Single run
# ══════════════════════════════════════════════════════════════════════════════

def run_single(
    mode        : str,
    chain_key   : str,
    seed        : int,
    total_steps : int,
    results_dir : Path,
    n_envs      : int = 4,
    eval_freq   : int = 20_000,
    n_eval_eps  : int = 100,
) -> Optional[Dict]:
    mapping_seed, delay, gamma_t = CHAIN_REGISTRY[chain_key]
    use_oracle, use_privileged   = CONDITION_CFG[mode]

    run_id = f"{mode}__{chain_key}_seed{seed}"
    print(f"\n{'─'*62}")
    print(f"  {run_id}  delay={delay}  γ^delay≈{gamma_t:.2f}  steps={total_steps:,}")
    print(f"{'─'*62}")

    train_vec = make_vec_env(
        make_env_fn(mapping_seed, use_oracle, use_privileged,
                    ORACLE_ALPHA, ORACLE_SHAPING_C, seed),
        n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv,
    )
    eval_vec = make_vec_env(
        make_eval_env_fn(mapping_seed, use_privileged, seed + 10_000),
        n_envs=1, seed=seed + 10_000, vec_env_cls=DummyVecEnv,
    )

    tb_log     = str(results_dir / "tensorboard" / run_id)
    model_save = str(results_dir / "models"      / run_id)
    eval_log   = str(results_dir / "eval_logs"   / run_id)

    model = PPO("MlpPolicy", train_vec, **PPO_KWARGS,
                seed=seed, tensorboard_log=tb_log)

    eval_cb = EvalCallback(
        eval_vec,
        n_eval_episodes      = n_eval_eps,
        eval_freq            = max(eval_freq // n_envs, 1),
        best_model_save_path = model_save,
        log_path             = eval_log,
        deterministic        = True,
        verbose              = 0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps = total_steps,
        callback        = [eval_cb, ScaleCallback(log_every=20_000)],
        progress_bar    = False,
    )
    elapsed = time.time() - t0

    result: Dict = {
        "mode": mode, "chain_key": chain_key, "mapping_seed": mapping_seed,
        "delay": delay, "gamma_t": gamma_t, "seed": seed,
        "steps": total_steps, "elapsed": elapsed,
        "timesteps": [], "mean_rewards": [], "std_rewards": [], "final_mean": 0.0,
    }
    eval_npz = Path(eval_log) / "evaluations.npz"
    if eval_npz.exists():
        data = np.load(str(eval_npz))
        result["timesteps"]    = data["timesteps"].tolist()
        result["mean_rewards"] = data["results"].mean(axis=1).tolist()
        result["std_rewards"]  = data["results"].std(axis=1).tolist()
        result["final_mean"]   = float(data["results"][-1].mean())

    out_dir = results_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  ✓ final_eval={result['final_mean']:.3f}  "
          f"elapsed={elapsed/60:.1f}min  → {out_path}")
    train_vec.close()
    eval_vec.close()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_scaling(results_dir: Path, out: Optional[str] = None) -> None:
    results_dir = Path(results_dir)
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    for jf in sorted((results_dir / "results").glob("*.json")):
        with open(jf) as f:
            r = json.load(f)
        mode      = r.get("mode")
        chain_key = r.get("chain_key")
        if mode and chain_key:
            data[mode][chain_key].append(r)

    if not data:
        print(f"[plot] No result JSONs in {results_dir / 'results'}")
        return

    chain_keys = [k for k in CHAIN_REGISTRY if any(k in data[m] for m in data)]
    chain_keys.sort(key=lambda k: CHAIN_REGISTRY[k][1])
    delays     = [CHAIN_REGISTRY[k][1]   for k in chain_keys]
    gamma_ts   = [CHAIN_REGISTRY[k][2]   for k in chain_keys]
    x_labels   = [f"{k}\n(delay={d})" for k, d in zip(chain_keys, delays)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "DiscountingChain Scaling: Credit Assignment Bottleneck\n"
        "Reward delay: 1 → 3 → 10 → 30 → 100 steps",
        fontsize=13, fontweight="bold",
    )

    # ── Left: final performance vs delay ─────────────────────────────────────
    ax    = axes[0]
    x_pos = np.arange(len(chain_keys))
    for mode in SCALE_CONDITIONS:
        if mode not in data:
            continue
        means, stds = [], []
        for ck in chain_keys:
            runs = data[mode].get(ck, [])
            if runs:
                finals = [r["final_mean"] for r in runs]
                means.append(np.mean(finals))
                stds.append(np.std(finals))
            else:
                means.append(np.nan)
                stds.append(0.0)
        c = COLORS.get(mode, "black")
        ax.plot(x_pos, means, color=c, marker="o", lw=2.5, ms=8,
                label=LABELS.get(mode, mode))
        ax.fill_between(x_pos,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=c)

    for i, (ck, gamt) in enumerate(zip(chain_keys, gamma_ts)):
        ax.annotate(f"γ^T={gamt:.2f}",
                    xy=(i, -0.06), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8, color="#888888")

    ax.axhline(0.5, color="black",   ls="--", lw=1.2, alpha=0.5, label="Chance (0.5)")
    ax.axhline(0.8, color="crimson", ls="--", lw=1.5, alpha=0.7, label="Gate (0.8)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Final mean eval reward", fontsize=11)
    ax.set_title("Final Performance vs Reward Delay", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Right: learning curves D1 vs D100 ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Learning Curves: D1 (solid) vs D100 (dashed)", fontsize=11)
    target = {"D1": "-", "D100": "--"}
    for mode in SCALE_CONDITIONS:
        if mode not in data:
            continue
        for ck, ls in target.items():
            runs = data[mode].get(ck, [])
            if not runs:
                continue
            all_ts = [np.array(r["timesteps"])    for r in runs if r["timesteps"]]
            all_rw = [np.array(r["mean_rewards"]) for r in runs if r["timesteps"]]
            if not all_ts:
                continue
            max_ts = max(ts.max() for ts in all_ts)
            x      = np.linspace(0, max_ts, 300)
            interped = [np.interp(x, ts, rw) for ts, rw in zip(all_ts, all_rw)]
            arr  = np.array(interped)
            mean = arr.mean(axis=0)
            se   = arr.std(axis=0) / max(np.sqrt(len(interped)), 1)
            c    = COLORS.get(mode, "black")
            lbl  = f"{LABELS.get(mode, mode)} ({ck})" if ls == "--" else None
            ax2.plot(x, mean, color=c, ls=ls, lw=2, label=lbl)
            ax2.fill_between(x, mean - se, mean + se, alpha=0.10, color=c)

    ax2.axhline(0.5, color="black",   ls=":", lw=1.0, alpha=0.5)
    ax2.axhline(0.8, color="crimson", ls=":", lw=1.2, alpha=0.7)
    ax2.set_xlabel("Environment steps", fontsize=11)
    ax2.set_ylabel("Mean eval reward",  fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_ylim(-0.05, 1.10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out or str(results_dir / "bsuite_scaling.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DiscountingChain horizon scaling experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",  default="all",
        choices=["all"] + SCALE_CONDITIONS)
    parser.add_argument("--chain", default="all",
        choices=["all"] + list(CHAIN_REGISTRY.keys()),
        help="Which chain length(s) to run.")
    parser.add_argument("--seeds",      type=int, default=5)
    parser.add_argument("--steps",      type=int, default=500_000)
    parser.add_argument("--n_envs",     type=int, default=4)
    parser.add_argument("--eval_freq",  type=int, default=20_000)
    parser.add_argument("--n_eval_eps", type=int, default=100)
    parser.add_argument("--results_dir", default="results/bsuite_scaling")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--plot_out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        plot_scaling(results_dir, args.plot_out)
        return

    modes      = SCALE_CONDITIONS if args.mode == "all" else [args.mode]
    chain_keys = list(CHAIN_REGISTRY.keys()) if args.chain == "all" else [args.chain]
    seeds      = list(range(args.seeds))
    total      = len(modes) * len(chain_keys) * len(seeds)

    print(f"\n{'='*62}")
    print(f"  DiscountingChain Scaling Experiment")
    print(f"  Modes   : {modes}")
    print(f"  Chains  : {chain_keys}")
    print(f"  Seeds   : {seeds}")
    print(f"  Steps   : {args.steps:,}/run")
    print(f"  Total   : {total} runs  (≈{total * args.steps / 1e6:.1f}M steps)")
    print(f"  Results : {results_dir}")
    print(f"{'='*62}")

    common = dict(
        total_steps = args.steps,
        results_dir = results_dir,
        n_envs      = args.n_envs,
        eval_freq   = args.eval_freq,
        n_eval_eps  = args.n_eval_eps,
    )

    all_results = defaultdict(lambda: defaultdict(list))
    for mode in modes:
        for ck in chain_keys:
            for seed in seeds:
                r = run_single(mode=mode, chain_key=ck, seed=seed, **common)
                if r:
                    all_results[mode][ck].append(r)

    print(f"\n{'='*62}")
    print("  SCALING RESULTS SUMMARY")
    print(f"  {'Condition':<30}  " + "  ".join(f"{k:>6}" for k in chain_keys))
    print(f"  {'─'*30}  " + "  ".join("──────" for _ in chain_keys))
    for mode in modes:
        row = f"  {LABELS.get(mode, mode):<30}  "
        for ck in chain_keys:
            runs = all_results[mode].get(ck, [])
            if runs:
                m    = np.mean([r["final_mean"] for r in runs])
                row += f"  {m:>6.3f}"
            else:
                row += f"  {'N/A':>6}"
        print(row)
    print(f"{'='*62}\n")

    plot_scaling(results_dir, args.plot_out)


if __name__ == "__main__":
    main()

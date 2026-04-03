#!/usr/bin/env python3
"""
Milestone 3 (early): Horizon Scaling Experiment  S7 → S9 → S11 → S13
=====================================================================
Tests the core SGRS thesis prediction:

  At short horizons (≤15 steps), memory is the binding bottleneck —
  privileged PPO (memory fix, no credit fix) reaches near-ceiling performance.

  At longer horizons (≥30 steps), credit assignment becomes co-equal —
  privileged PPO degrades while privileged+oracle (both fixed) holds the ceiling.

  The crossover point is where SGRS's value proposition begins.

CONDITIONS (3 — sufficient for the scaling figure)
  ppo                    No fixes          — floor / chance baseline
  ppo_privileged         Memory fix only   — isolates credit assignment bottleneck
  ppo_privileged_oracle  Both fixes        — ceiling reference

ENVIRONMENTS
  MiniGrid-MemoryS7-v0   ~15 step corridor   γ^15 ≈ 0.86
  MiniGrid-MemoryS9-v0   ~25 step corridor   γ^25 ≈ 0.78
  MiniGrid-MemoryS11-v0  ~40 step corridor   γ^40 ≈ 0.67
  MiniGrid-MemoryS13-v0  ~55 step corridor   γ^55 ≈ 0.58

USAGE
  # Full scaling experiment (3 conditions × 4 envs × 5 seeds × 500k steps)
  python scale_experiment.py

  # Single condition + env for smoke test
  python scale_experiment.py --mode ppo_privileged --env S9 --seed 0 --steps 100000

  # Plot after all runs complete
  python scale_experiment.py --plot --results_dir results/scaling

EXPECTED RESULT
  ppo_privileged stays near ceiling at S7, degrades at S9, S11, S13.
  ppo_privileged_oracle stays near ceiling across all sizes.
  The gap between them is the empirical magnitude of the credit assignment bottleneck
  as a function of horizon length — the main figure of the SGRS paper.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.wrappers import FlatObsWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

from wrapper import OracleCreditWrapper
from privileged import PrivilegedSignalWrapper

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Environment registry
# ══════════════════════════════════════════════════════════════════════════════

ENV_REGISTRY = {
    # key    env_id                        approx_corridor_steps  gamma^T (γ=0.99)
    "S7" : ("MiniGrid-MemoryS7-v0",   15,  0.86),
    "S9" : ("MiniGrid-MemoryS9-v0",   25,  0.78),
    "S11": ("MiniGrid-MemoryS11-v0",  40,  0.67),
    "S13": ("MiniGrid-MemoryS13-v0",  55,  0.58),
}

# Conditions to run in the scaling experiment
SCALE_CONDITIONS = ["ppo", "ppo_privileged", "ppo_privileged_oracle"]

CONDITION_CFG = {
    # mode                     use_oracle  use_privileged
    "ppo"                   : (False,      False),
    "ppo_privileged"        : (False,      True),
    "ppo_privileged_oracle" : (True,       True),
}

# Plot config
COLORS = {
    "ppo"                   : "#9E9E9E",   # grey  — chance floor
    "ppo_privileged"        : "#2196F3",   # blue  — memory fix only
    "ppo_privileged_oracle" : "#4CAF50",   # green — both fixes (ceiling)
}
LABELS = {
    "ppo"                   : "PPO (no fixes)",
    "ppo_privileged"        : "PPO + Privileged memory (no credit fix)",
    "ppo_privileged_oracle" : "PPO + Privileged + Oracle (ceiling)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 1024,
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


# ══════════════════════════════════════════════════════════════════════════════
# Env factories
# ══════════════════════════════════════════════════════════════════════════════

def make_train_env(env_id: str, use_oracle: bool, use_privileged: bool):
    def _fn():
        env = gym.make(env_id)
        if use_oracle:
            env = OracleCreditWrapper(
                env, alpha=ORACLE_ALPHA, shaping_c=ORACLE_SHAPING_C
            )
        if use_privileged:
            env = PrivilegedSignalWrapper(env)
        else:
            env = FlatObsWrapper(env)
        return Monitor(env)
    return _fn


def make_eval_env(env_id: str, use_privileged: bool):
    """Eval is always plain (no oracle). Privileged matches training condition."""
    def _fn():
        env = gym.make(env_id)
        if use_privileged:
            env = PrivilegedSignalWrapper(env)
        else:
            env = FlatObsWrapper(env)
        return Monitor(env)
    return _fn


def _verify_env(env_id: str) -> bool:
    try:
        e = gym.make(env_id)
        e.close()
        return True
    except Exception:
        print(f"[WARN] {env_id!r} not found — skipping")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Lightweight metrics callback (no oracle dependency)
# ══════════════════════════════════════════════════════════════════════════════

class ScaleCallback(BaseCallback):
    """Logs oracle metrics and prints progress. Works with or without oracle."""

    def __init__(self, log_every: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every  = log_every
        self._last_log  = 0
        self._cur_bonus : float       = 0.0
        self._ep_env    : List[float] = []
        self._ep_bonus  : List[float] = []
        self._junc_ok   : List[bool]  = []
        self._junc_hit  : List[bool]  = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._cur_bonus += info.get("oracle_bonus", 0.0)
            if "oracle_junction_correct" in info:
                self._junc_ok.append(info["oracle_junction_correct"])
                self._junc_hit.append(True)
            if "episode" in info:
                shaped = info["episode"]["r"]
                self._ep_env.append(shaped - self._cur_bonus)
                self._ep_bonus.append(self._cur_bonus)
                if not (self._junc_hit and self._junc_hit[-1]):
                    self._junc_hit.append(False)
                self._cur_bonus = 0.0

        if self.num_timesteps - self._last_log >= self.log_every:
            if self._junc_ok:
                self.logger.record("oracle/junction_correct_frac",
                                   float(np.mean(self._junc_ok)))
            if self._junc_hit:
                self.logger.record("oracle/junction_reached_frac",
                                   float(np.mean(self._junc_hit)))
            if self._ep_env:
                self.logger.record("oracle/ep_env_reward_mean",
                                   float(np.mean(self._ep_env)))
                n = len(self._ep_env)
                jstr = (f"  junc={np.mean(self._junc_ok):.2f}"
                        if self._junc_ok else "")
                print(f"  t={self.num_timesteps:>8,}"
                      f"  env_rew={np.mean(self._ep_env):.3f}"
                      f"{jstr}  n_eps={n}")
            # reset
            self._ep_env   = []
            self._ep_bonus = []
            self._junc_ok  = []
            self._junc_hit = []
            self._last_log = self.num_timesteps
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Single run
# ══════════════════════════════════════════════════════════════════════════════

def run_single(
    mode       : str,
    env_key    : str,
    seed       : int,
    total_steps: int,
    results_dir: Path,
    n_envs     : int = 4,
    eval_freq  : int = 20_000,
    n_eval_eps : int = 100,
) -> Optional[Dict]:

    env_id, corridor_steps, gamma_t = ENV_REGISTRY[env_key]
    if not _verify_env(env_id):
        return None

    use_oracle, use_privileged = CONDITION_CFG[mode]

    run_id = f"{mode}__{env_key}_seed{seed}"
    print(f"\n{'─'*62}")
    print(f"  {run_id}  steps={total_steps:,}")
    print(f"  env={env_id}  corridor≈{corridor_steps}  γ^T≈{gamma_t:.2f}")
    print(f"{'─'*62}")

    train_vec = make_vec_env(
        make_train_env(env_id, use_oracle, use_privileged),
        n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv,
    )
    eval_vec = make_vec_env(
        make_eval_env(env_id, use_privileged),
        n_envs=1, seed=seed + 10_000, vec_env_cls=DummyVecEnv,
    )

    tb_log     = str(results_dir / "tensorboard" / run_id)
    model_save = str(results_dir / "models"      / run_id)
    eval_log   = str(results_dir / "eval_logs"   / run_id)

    model = PPO(
        "MlpPolicy", train_vec,
        **PPO_KWARGS,
        seed=seed, tensorboard_log=tb_log,
    )

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
        "mode": mode, "env_key": env_key,
        "env_id": env_id, "corridor_steps": corridor_steps,
        "gamma_t": gamma_t, "seed": seed, "steps": total_steps,
        "elapsed": elapsed, "timesteps": [], "mean_rewards": [],
        "std_rewards": [], "final_mean": 0.0,
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
    # {mode: {env_key: [result, ...]}}
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    for jf in sorted((results_dir / "results").glob("*.json")):
        with open(jf) as f:
            r = json.load(f)
        mode    = r.get("mode")
        env_key = r.get("env_key")
        if mode and env_key:
            data[mode][env_key].append(r)

    if not data:
        print(f"[plot] No result JSONs in {results_dir / 'results'}")
        return

    env_keys = [k for k in ENV_REGISTRY if any(k in data[m] for m in data)]
    env_keys.sort(key=lambda k: ENV_REGISTRY[k][1])  # sort by corridor length
    x_labels  = [f"Memory{k}\n(~{ENV_REGISTRY[k][1]} steps)" for k in env_keys]
    x_gamma   = [ENV_REGISTRY[k][2] for k in env_keys]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Horizon Scaling: Credit Assignment Bottleneck\n"
        "MiniGrid Memory S7 → S13",
        fontsize=13, fontweight="bold",
    )

    # ── left: final performance vs horizon ───────────────────────────────────
    ax = axes[0]
    x_pos = np.arange(len(env_keys))

    for mode in SCALE_CONDITIONS:
        if mode not in data:
            continue
        means, stds = [], []
        for ek in env_keys:
            runs = data[mode].get(ek, [])
            if runs:
                finals = [r["final_mean"] for r in runs]
                means.append(np.mean(finals))
                stds.append(np.std(finals))
            else:
                means.append(np.nan)
                stds.append(0.0)

        c = COLORS.get(mode, "black")
        ax.plot(x_pos, means, color=c, marker="o", linewidth=2.5,
                markersize=8, label=LABELS.get(mode, mode))
        ax.fill_between(
            x_pos,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.15, color=c,
        )

    # Annotate γ^T at each env
    for i, (ek, gamt) in enumerate(zip(env_keys, x_gamma)):
        ax.annotate(f"γ^T={gamt:.2f}",
                    xy=(i, -0.04), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8, color="#888888")

    ax.axhline(0.50, color="black",   ls="--", lw=1.2, alpha=0.5, label="Chance (0.50)")
    ax.axhline(0.80, color="crimson", ls="--", lw=1.5, alpha=0.7, label="Gate (0.80)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Final mean eval reward", fontsize=11)
    ax.set_title("Final Performance vs Horizon", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")

    # ── right: learning curves for each env (subpanels) ──────────────────────
    # Show S7 and S13 side-by-side for the sharpest contrast
    ax2 = axes[1]
    ax2.set_title("Learning Curves: S7 (solid) vs S13 (dashed)", fontsize=11)
    target_envs = {"S7": "-", "S13": "--"}

    for mode in SCALE_CONDITIONS:
        if mode not in data:
            continue
        for ek, ls in target_envs.items():
            runs = data[mode].get(ek, [])
            if not runs:
                continue
            all_ts = [np.array(r["timesteps"]) for r in runs if r["timesteps"]]
            if not all_ts:
                continue
            max_ts = max(ts.max() for ts in all_ts)
            x = np.linspace(0, max_ts, 300)
            interped = [
                np.interp(x, np.array(r["timesteps"]), np.array(r["mean_rewards"]))
                for r in runs if r["timesteps"]
            ]
            arr  = np.array(interped)
            mean = arr.mean(axis=0)
            se   = arr.std(axis=0) / max(np.sqrt(len(interped)), 1)
            c    = COLORS.get(mode, "black")
            lbl  = f"{LABELS.get(mode, mode)} ({ek})" if ls == "--" else None
            ax2.plot(x, mean, color=c, ls=ls, lw=2, label=lbl)
            ax2.fill_between(x, mean-se, mean+se, alpha=0.10, color=c)

    ax2.axhline(0.50, color="black",   ls=":", lw=1.0, alpha=0.5)
    ax2.axhline(0.80, color="crimson", ls=":", lw=1.2, alpha=0.7)
    ax2.set_xlabel("Environment steps", fontsize=11)
    ax2.set_ylabel("Mean eval reward", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_ylim(-0.05, 1.10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out or str(results_dir / "scaling_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Horizon Scaling Experiment S7→S13",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",  default="all",
        choices=["all"] + SCALE_CONDITIONS)
    parser.add_argument("--env",   default="all",
        choices=["all"] + list(ENV_REGISTRY.keys()))
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n_envs",     type=int, default=4)
    parser.add_argument("--eval_freq",  type=int, default=20_000)
    parser.add_argument("--n_eval_eps", type=int, default=100)
    parser.add_argument("--results_dir", default="results/scaling")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--plot_out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        plot_scaling(results_dir, args.plot_out)
        return

    modes    = SCALE_CONDITIONS if args.mode == "all" else [args.mode]
    env_keys = list(ENV_REGISTRY.keys()) if args.env == "all" else [args.env]
    seeds    = list(range(args.seeds))
    total    = len(modes) * len(env_keys) * len(seeds)

    print(f"\n{'='*62}")
    print(f"  Horizon Scaling Experiment")
    print(f"  Modes    : {modes}")
    print(f"  Envs     : {env_keys}")
    print(f"  Seeds    : {seeds}")
    print(f"  Steps    : {args.steps:,}/run")
    print(f"  Total    : {total} runs  "
          f"(≈{total * args.steps / 1e6:.1f}M env steps)")
    print(f"  Results  : {results_dir}")
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
        for env_key in env_keys:
            for seed in seeds:
                r = run_single(mode=mode, env_key=env_key, seed=seed, **common)
                if r:
                    all_results[mode][env_key].append(r)

    # Summary table
    print(f"\n{'='*62}")
    print("  SCALING RESULTS SUMMARY")
    print(f"  {'Condition':<28}  " + "  ".join(f"{k:>6}" for k in env_keys))
    print(f"  {'─'*28}  " + "  ".join("──────" for _ in env_keys))
    for mode in modes:
        row = f"  {LABELS.get(mode, mode):<28}  "
        for ek in env_keys:
            runs = all_results[mode].get(ek, [])
            if runs:
                m = np.mean([r["final_mean"] for r in runs])
                row += f"  {m:>6.3f}"
            else:
                row += f"  {'N/A':>6}"
        print(row)
    print(f"{'='*62}\n")

    plot_scaling(results_dir, args.plot_out)


if __name__ == "__main__":
    main()
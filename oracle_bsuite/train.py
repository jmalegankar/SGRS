#!/usr/bin/env python3
"""
oracle_bsuite — Credit Assignment Isolation on DiscountingChain
===============================================================
Research question: Is credit assignment (not memory) the bottleneck on
bsuite discounting_chain?

HYPOTHESIS
  PPO plateau near chance (0.5) on long-delay chains. An oracle that gives
  immediate reward at arm-selection (t=0) should break the bottleneck.
  Separately, privileged signal (free memory) controls for memory.

CONDITIONS
  ppo                    No fixes (baseline)
  ppo_oracle             Oracle only (credit fix)
  ppo_privileged         Privileged only (memory fix)
  ppo_privileged_oracle  Both fixes (ceiling)

USAGE
  # Inspect env first:
  python inspect_env.py

  # Smoke test (1 seed, 200k steps, delay=10):
  python train.py --mode ppo_privileged_oracle --mapping_seed 2 --seed 0 --steps 200000

  # Full 2x2 table (5 seeds, 500k steps, all conditions on delay=30):
  python train.py --mode all --mapping_seed 3 --seeds 5 --steps 500000

  # Plot after training:
  python train.py --plot --results_dir results/bsuite_d30
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    sys.exit("stable-baselines3 not found. Run: pip install stable-baselines3")

from envs   import make_env_fn, make_eval_env_fn, REWARD_DELAYS
from plot   import LABELS, plot_results

# ══════════════════════════════════════════════════════════════════════════════
# Experiment configuration
# ══════════════════════════════════════════════════════════════════════════════

CONDITION_CFG = {
    # mode                      use_oracle  use_privileged
    "ppo"                     : (False,     False),
    "ppo_oracle"              : (True,      False),
    "ppo_privileged"          : (False,     True),
    "ppo_privileged_oracle"   : (True,      True),
}

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,   # covers ~20 full 100-step episodes per rollout
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.05,   # keep exploration alive across 5 arms
    clip_range    = 0.2,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 0,
)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics callback
# ══════════════════════════════════════════════════════════════════════════════

class OracleMetricsCallback(BaseCallback):
    """
    Logs oracle-specific metrics to TensorBoard and stdout.

    TB tags:
      oracle/correct_frac        — P(correct arm chosen at t=0) per logging window
      oracle/ep_env_reward_mean  — raw env reward per episode
      oracle/ep_bonus_mean       — oracle bonus per episode
      oracle/ep_shaped_reward_mean — shaped reward per episode
    """

    def __init__(self, log_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._last_log = 0
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._correct       : List[bool]  = []
        self._ep_bonuses    : List[float] = []
        self._ep_env_rews   : List[float] = []
        self._ep_shaped_rews: List[float] = []
        self._cur_bonus     : float       = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._cur_bonus += info.get("oracle_bonus", 0.0)

            if "oracle_correct" in info:
                self._correct.append(info["oracle_correct"])

            if "episode" in info:
                ep_shaped = info["episode"]["r"]
                ep_env    = ep_shaped - self._cur_bonus
                self._ep_bonuses.append(self._cur_bonus)
                self._ep_env_rews.append(ep_env)
                self._ep_shaped_rews.append(ep_shaped)
                self._cur_bonus = 0.0

        if self.num_timesteps - self._last_log >= self.log_every:
            if self._correct:
                self.logger.record("oracle/correct_frac",
                                   float(np.mean(self._correct)))
            if self._ep_env_rews:
                n = len(self._ep_env_rews)
                self.logger.record("oracle/ep_env_reward_mean",
                                   float(np.mean(self._ep_env_rews)))
                self.logger.record("oracle/ep_bonus_mean",
                                   float(np.mean(self._ep_bonuses)))
                self.logger.record("oracle/ep_shaped_reward_mean",
                                   float(np.mean(self._ep_shaped_rews)))
                cfrac = (f"  correct={np.mean(self._correct):.2f}"
                         if self._correct else "")
                print(
                    f"  t={self.num_timesteps:>8,}"
                    f"  env_rew={np.mean(self._ep_env_rews):.3f}"
                    f"{cfrac}  n_eps={n}"
                )
            self._reset_accumulators()
            self._last_log = self.num_timesteps
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Single training run
# ══════════════════════════════════════════════════════════════════════════════

def run_single(
    mode        : str,
    mapping_seed: int,
    seed        : int,
    total_steps : int,
    results_dir : Path,
    alpha       : float = 2.0,
    shaping_c   : float = 1.0,
    n_envs      : int   = 4,
    eval_freq   : int   = 10_000,
    n_eval_eps  : int   = 100,
) -> Dict:
    use_oracle, use_privileged = CONDITION_CFG[mode]
    delay = REWARD_DELAYS[mapping_seed % 5]

    print(f"\n{'─'*60}")
    print(f"  mode={mode}  mapping_seed={mapping_seed}  "
          f"delay={delay}  seed={seed}  steps={total_steps:,}")
    print(f"{'─'*60}")

    train_vec = make_vec_env(
        make_env_fn(mapping_seed, use_oracle, use_privileged, alpha, shaping_c, seed),
        n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv,
    )
    eval_vec = make_vec_env(
        make_eval_env_fn(mapping_seed, use_privileged, seed + 10_000),
        n_envs=1, seed=seed + 10_000, vec_env_cls=DummyVecEnv,
    )

    run_id     = f"{mode}_d{delay}_seed{seed}"
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
        callback        = [eval_cb, OracleMetricsCallback(log_every=10_000)],
        progress_bar    = True,
    )
    elapsed = time.time() - t0

    result: Dict = {
        "mode": mode, "mapping_seed": mapping_seed, "delay": delay,
        "seed": seed, "steps": total_steps, "elapsed": elapsed,
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

    print(f"  final_eval={result['final_mean']:.3f}  "
          f"elapsed={elapsed/60:.1f}min  → {out_path}")
    train_vec.close()
    eval_vec.close()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="oracle_bsuite: credit assignment isolation on DiscountingChain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="all",
        choices=["all"] + list(CONDITION_CFG.keys()))
    parser.add_argument("--mapping_seed", type=int, default=3,
        help="0-4 → correct arm 0-4, reward delay [1,3,10,30,100].")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--seeds",  type=int, default=5)
    parser.add_argument("--steps",  type=int, default=500_000)
    parser.add_argument("--alpha",      type=float, default=2.0)
    parser.add_argument("--shaping_c",  type=float, default=1.0)
    parser.add_argument("--n_envs",     type=int,   default=4)
    parser.add_argument("--eval_freq",  type=int,   default=10_000)
    parser.add_argument("--n_eval_eps", type=int,   default=100)
    parser.add_argument("--results_dir", default="results/bsuite_oracle")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--plot",    action="store_true")
    parser.add_argument("--plot_out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        import subprocess, sys as _sys
        subprocess.run([_sys.executable, "inspect_env.py"])
        return

    if args.plot:
        plot_results(results_dir, args.plot_out)
        return

    common = dict(
        mapping_seed = args.mapping_seed,
        total_steps  = args.steps,
        results_dir  = results_dir,
        alpha        = args.alpha,
        shaping_c    = args.shaping_c,
        n_envs       = args.n_envs,
        eval_freq    = args.eval_freq,
        n_eval_eps   = args.n_eval_eps,
    )

    modes = list(CONDITION_CFG.keys()) if args.mode == "all" else [args.mode]
    seeds = list(range(args.seeds))
    delay = REWARD_DELAYS[args.mapping_seed % 5]

    print(f"\n{'='*62}")
    print(f"  oracle_bsuite — DiscountingChain  delay={delay} steps")
    print(f"  Conditions : {modes}")
    print(f"  Seeds      : {seeds}")
    print(f"  Steps/run  : {args.steps:,}")
    print(f"  Results    : {results_dir}")
    print(f"{'='*62}")

    all_results: Dict = defaultdict(list)
    for mode in modes:
        for seed in seeds:
            r = run_single(mode=mode, seed=seed, **common)
            all_results[mode].append(r)

    print(f"\n{'='*62}")
    print("  RESULTS SUMMARY")
    print(f"  {'Condition':<30}  {'Mean':>8}  {'Std':>6}  {'Gate':>6}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*6}  {'─'*6}")
    for mode in modes:
        finals = [r["final_mean"] for r in all_results[mode]]
        m, s   = np.mean(finals), np.std(finals)
        gate   = "PASS" if (m >= 0.8 and "oracle" in mode) else ("FAIL" if "oracle" in mode else "  —")
        print(f"  {LABELS.get(mode, mode):<30}  {m:>8.3f}  {s:>6.3f}  {gate:>6}")
    print(f"{'='*62}\n")

    plot_results(results_dir, args.plot_out)


if __name__ == "__main__":
    main()

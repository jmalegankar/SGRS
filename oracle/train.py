#!/usr/bin/env python3
"""
Milestone 0: Oracle Credit Assignment Test
==========================================
Research question: Is credit assignment (not representation or memory) the
bottleneck on MiniGrid-MemoryS7-v0?

HYPOTHESIS
  PPO/RecurrentPPO plateau at ~0.5 (random T-junction guessing). If an agent
  given hand-crafted *oracle* shaped reward at the two known decision points
  achieves >0.8, credit assignment IS the bottleneck -> SGRS thesis valid.

CONDITIONS
  ppo          Vanilla PPO                      (baseline)
  rppo         RecurrentPPO w/ LSTM             (baseline)
  ppo_oracle   PPO  + oracle credit shaping
  rppo_oracle  RecurrentPPO + oracle shaping

USAGE
  # 1) Inspect environment first — verifies oracle detection is working:
  python train.py --inspect

  # 2) Quick smoke-test (1 seed, 50k steps, oracle only):
  python train.py --mode ppo_oracle --seed 0 --steps 50000

  # 3) Full experiment (all 4 conditions x 5 seeds x 500k steps):
  python train.py --mode all --seeds 5 --steps 500000

  # 4) Plot after training:
  python train.py --plot --results_dir results/milestone0

SUCCESS GATE
  oracle-shaped agent sustains mean_reward > 0.8
  PASS -> credit assignment IS the bottleneck -> proceed with SGRS
  FAIL -> pivot: investigate representation / value-function capacity instead
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── dependency checks ─────────────────────────────────────────────────────────
try:
    import minigrid  # noqa: F401
except ImportError:
    sys.exit("minigrid not found. Run:  pip install minigrid")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    sys.exit("stable-baselines3 not found. Run:  pip install stable-baselines3")

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    sys.exit("sb3-contrib not found. Run:  pip install sb3-contrib")

from envs import _verify_env_id, make_env_fn, make_eval_env_fn
from inspect_env import inspect_env
from plot import LABELS, plot_results

# ══════════════════════════════════════════════════════════════════════════════
# Experiment configuration
# ══════════════════════════════════════════════════════════════════════════════

CONDITION_CFG = {
    # mode           use_lstm  use_oracle
    "ppo"          : (False,   False),
    "rppo"         : (True,    False),
    "ppo_oracle"   : (False,   True),
    "rppo_oracle"  : (True,    True),
}

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 1024,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.01,
    clip_range    = 0.2,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 0,
)


# ══════════════════════════════════════════════════════════════════════════════
# Logging callback
# ══════════════════════════════════════════════════════════════════════════════

class ProgressCallback(BaseCallback):
    """Prints a brief log line every `log_every` timesteps."""

    def __init__(self, log_every: int = 25_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_every:
            ep_rewards = [
                info["episode"]["r"]
                for info in self.locals.get("infos", [])
                if "episode" in info
            ]
            if ep_rewards:
                print(
                    f"  steps={self.num_timesteps:>8,}  "
                    f"ep_reward(train)={np.mean(ep_rewards):.3f}  "
                    f"n_eps={len(ep_rewards)}"
                )
            self._last_log = self.num_timesteps
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Single training run
# ══════════════════════════════════════════════════════════════════════════════

def run_single(
    mode       : str,
    seed       : int,
    total_steps: int,
    results_dir: Path,
    alpha      : float = 1.0,
    shaping_c  : float = 1.0,
    n_envs     : int   = 4,
    eval_freq  : int   = 10_000,
    n_eval_eps : int   = 100,
) -> Dict:
    """
    Train one agent for one seed. Returns a results dict and writes a JSON file.
    Eval is always on the plain env (no oracle) to measure true policy quality.
    """
    use_lstm, use_oracle = CONDITION_CFG[mode]

    print(f"\n{'─'*58}")
    print(f"  mode={mode}  seed={seed}  steps={total_steps:,}  "
          f"oracle_alpha={alpha if use_oracle else 'N/A'}")
    print(f"{'─'*58}")

    train_vec = make_vec_env(
        make_env_fn(use_oracle, alpha, shaping_c, seed),
        n_envs      = n_envs,
        seed        = seed,
        vec_env_cls = DummyVecEnv,
    )
    eval_vec = make_vec_env(
        make_eval_env_fn(seed + 10_000),
        n_envs      = 1,
        seed        = seed + 10_000,
        vec_env_cls = DummyVecEnv,
    )

    tb_log     = str(results_dir / "tensorboard" / f"{mode}_seed{seed}")
    model_save = str(results_dir / "models"      / f"{mode}_seed{seed}")
    eval_log   = str(results_dir / "eval_logs"   / f"{mode}_seed{seed}")

    if use_lstm:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_vec,
            **PPO_KWARGS,
            seed            = seed,
            tensorboard_log = tb_log,
            policy_kwargs   = dict(lstm_hidden_size=256, n_lstm_layers=1),
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_vec,
            **PPO_KWARGS,
            seed            = seed,
            tensorboard_log = tb_log,
        )

    eval_callback = EvalCallback(
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
        callback        = [eval_callback, ProgressCallback(log_every=10_000)],
        progress_bar    = True,
    )
    elapsed = time.time() - t0

    result: Dict = {
        "mode"        : mode,
        "seed"        : seed,
        "steps"       : total_steps,
        "elapsed"     : elapsed,
        "timesteps"   : [],
        "mean_rewards": [],
        "std_rewards" : [],
        "final_mean"  : 0.0,
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
    out_path = out_dir / f"{mode}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"  final_eval_reward={result['final_mean']:.3f}  "
        f"elapsed={elapsed/60:.1f}min  saved->{out_path}"
    )

    train_vec.close()
    eval_vec.close()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Milestone 0: Oracle Credit Assignment Test for SGRS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="all",
        choices=["all"] + list(CONDITION_CFG.keys()),
        help="Which condition(s) to run.  'all' runs all 4.")
    parser.add_argument("--seed",      type=int,   default=0,
        help="Seed when mode is a single condition.")
    parser.add_argument("--seeds",     type=int,   default=5,
        help="Number of seeds (0..seeds-1) when --mode all.")
    parser.add_argument("--steps",     type=int,   default=500_000,
        help="Total training steps per run.")
    parser.add_argument("--alpha",     type=float, default=1.0,
        help="Oracle shaping coefficient alpha.")
    parser.add_argument("--shaping_c", type=float, default=1.0,
        help="Raw shaping magnitude c (before alpha).")
    parser.add_argument("--n_envs",    type=int,   default=4,
        help="Parallel envs per training run.")
    parser.add_argument("--eval_freq", type=int,   default=10_000,
        help="Eval every N env steps (across all n_envs).")
    parser.add_argument("--n_eval_eps",type=int,   default=100,
        help="Episodes per eval checkpoint.")
    parser.add_argument("--results_dir", default="results/milestone0",
        help="Root directory for all outputs.")
    parser.add_argument("--inspect",  action="store_true",
        help="Inspect env layout + test oracle wrapper (no training).")
    parser.add_argument("--plot",     action="store_true",
        help="Plot existing results (no training).")
    parser.add_argument("--plot_out", default=None,
        help="Custom output path for the plot PNG.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _verify_env_id()

    if args.inspect:
        inspect_env()
        return

    if args.plot:
        plot_results(results_dir, args.plot_out)
        return

    common = dict(
        total_steps = args.steps,
        results_dir = results_dir,
        alpha       = args.alpha,
        shaping_c   = args.shaping_c,
        n_envs      = args.n_envs,
        eval_freq   = args.eval_freq,
        n_eval_eps  = args.n_eval_eps,
    )

    if args.mode == "all":
        modes = list(CONDITION_CFG.keys())
        seeds = list(range(args.seeds))
        total_runs = len(modes) * len(seeds)
        print(f"\n{'='*62}")
        print(f"  Milestone 0 — Full Experiment")
        print(f"  Conditions : {modes}")
        print(f"  Seeds      : {seeds}")
        print(f"  Steps/run  : {args.steps:,}")
        print(f"  Total runs : {total_runs}  (~{total_runs * args.steps / 1e6:.1f}M env steps)")
        print(f"  Results    : {results_dir}")
        print(f"{'='*62}")

        all_results: Dict[str, List] = defaultdict(list)
        for mode in modes:
            for seed in seeds:
                r = run_single(mode=mode, seed=seed, **common)
                all_results[mode].append(r)

        print(f"\n{'='*62}")
        print("  RESULTS SUMMARY")
        print(f"  {'Condition':<22}  {'Mean reward':>12}  {'Std':>8}  {'Gate':>10}")
        print(f"  {'─'*22}  {'─'*12}  {'─'*8}  {'─'*10}")
        for mode in modes:
            runs   = all_results[mode]
            finals = [r["final_mean"] for r in runs]
            m, s   = np.mean(finals), np.std(finals)
            gate   = (
                "PASS" if (m >= 0.8 and "oracle" in mode)
                else ("FAIL" if "oracle" in mode else "  —")
            )
            print(f"  {LABELS.get(mode, mode):<22}  {m:>12.3f}  {s:>8.3f}  {gate:>10}")
        print(f"{'='*62}\n")

        plot_results(results_dir, args.plot_out)

    else:
        run_single(mode=args.mode, seed=args.seed, **common)


if __name__ == "__main__":
    main()

"""
Environment factories for Milestone 0.

Provides:
  ENV_ID          — canonical MiniGrid Memory env name
  _verify_env_id  — sanity-check registration at startup
  make_env_fn     — factory for train envs (optionally oracle-wrapped)
  make_eval_env_fn— factory for plain eval envs (never oracle-shaped)
"""

import sys

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.monitor import Monitor

from wrapper import OracleCreditWrapper

ENV_ID = "MiniGrid-MemoryS7-v0"


def _verify_env_id() -> None:
    """Check the env is registered; suggest alternatives if not."""
    try:
        e = gym.make(ENV_ID)
        e.close()
    except gym.error.NameNotFound:
        import minigrid.envs  # noqa: F401 — triggers registration
        try:
            e = gym.make(ENV_ID)
            e.close()
        except Exception:
            available = [k for k in gym.envs.registry.keys() if "Memory" in k]
            print(f"[ERROR] {ENV_ID!r} not found. Available Memory envs:")
            for a in sorted(available)[:20]:
                print(f"  {a}")
            sys.exit(1)


def make_env_fn(use_oracle: bool, alpha: float, shaping_c: float, seed: int = 0):
    """
    Returns a callable that creates a single (optionally oracle-wrapped) env.
    Suitable for use with make_vec_env / DummyVecEnv.
    """
    def _fn():
        env = gym.make(ENV_ID)
        if use_oracle:
            env = OracleCreditWrapper(env, alpha=alpha, shaping_c=shaping_c)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env
    return _fn


def make_eval_env_fn(seed: int = 0):
    """
    Returns a callable for the plain evaluation env (never oracle-shaped).
    Eval always runs on the plain env so we measure true policy quality.
    """
    def _fn():
        env = gym.make(ENV_ID)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env
    return _fn

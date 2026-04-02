"""
Environment factories for Milestone 0.

Provides:
  ENV_ID          — canonical MiniGrid Memory env name
  _verify_env_id  — sanity-check registration at startup
  make_env_fn     — factory for train envs (optionally oracle/privileged-wrapped)
  make_eval_env_fn— factory for eval envs (never oracle-shaped; privileged matches train)
"""

import sys

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.monitor import Monitor

from wrapper import OracleCreditWrapper
from privileged import PrivilegedSignalWrapper

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


def make_env_fn(
    use_oracle: bool,
    alpha: float,
    shaping_c: float,
    use_privileged: bool = False,
    seed: int = 0,
):
    """
    Returns a callable that creates a single training env.
    Stacking order: OracleCreditWrapper → PrivilegedSignalWrapper/FlatObsWrapper → Monitor.
    PrivilegedSignalWrapper outputs a flat Box, so FlatObsWrapper is skipped when used.
    """
    def _fn():
        env = gym.make(ENV_ID)
        if use_oracle:
            env = OracleCreditWrapper(env, alpha=alpha, shaping_c=shaping_c)
        if use_privileged:
            env = PrivilegedSignalWrapper(env)
        else:
            env = FlatObsWrapper(env)
        env = Monitor(env)
        return env
    return _fn


def make_eval_env_fn(use_privileged: bool = False, seed: int = 0):
    """
    Returns a callable for the eval env (never oracle-shaped).
    use_privileged must match the training condition so obs spaces align.
    """
    def _fn():
        env = gym.make(ENV_ID)
        if use_privileged:
            env = PrivilegedSignalWrapper(env)
        else:
            env = FlatObsWrapper(env)
        env = Monitor(env)
        return env
    return _fn

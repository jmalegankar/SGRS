"""
PrivilegedSignalWrapper
=======================
Appends a 2-element one-hot [is_key, is_ball] to the flattened observation
at EVERY timestep. This gives the policy perfect, free memory of the signal
it saw at t=0 — equivalent to a perfect external memory register.

PURPOSE
  Disambiguate memory bottleneck from credit assignment bottleneck.

EXPERIMENTAL LOGIC
  PPO (non-recurrent) + oracle + privileged → >0.8:
    Memory IS the bottleneck. LSTM cannot maintain signal through corridor.
    Credit assignment is a separate bottleneck (oracle was still needed).
    → SGRS + RSSM addresses both: belief state encodes signal, KL
      redistribution handles credit.

  PPO + oracle + privileged STILL fails:
    Something else is wrong (obs encoding, policy capacity, env setup).
    → Debug with --inspect before proceeding.

STACKING ORDER
  env = gym.make(ENV_ID)
  env = OracleCreditWrapper(env, alpha=2.0, shaping_c=1.0)
  env = PrivilegedSignalWrapper(env)   ← insert here, before FlatObsWrapper
  env = Monitor(env)
  (FlatObsWrapper not needed — PrivilegedSignalWrapper outputs a flat Box.)
"""

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.monitor import Monitor

from wrapper import OracleCreditWrapper
ENV_ID = "MiniGrid-MemoryS7-v0"


class PrivilegedSignalWrapper(gym.ObservationWrapper):
    """
    Appends [is_key, is_ball] one-hot to the observation at every step.

    Must be applied BEFORE FlatObsWrapper (outputs a flat Box itself).
    Must be applied AFTER OracleCreditWrapper (so unwrapped env is accessible).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._signal_vec: np.ndarray = np.zeros(2, dtype=np.float32)
        self._signal_type: Optional[str] = None

        inner_space = env.observation_space

        if isinstance(inner_space, spaces.Dict):
            # Standard MiniGrid dict obs: {'image': Box(7,7,3), ...}
            img_space = inner_space["image"]
            flat_dim  = int(np.prod(img_space.shape)) + 2
            self.observation_space = spaces.Box(
                low   = 0.0,
                high  = 255.0,
                shape = (flat_dim,),
                dtype = np.float32,
            )
            self._obs_type = "dict"
        elif isinstance(inner_space, spaces.Box):
            # Already flat
            flat_dim = int(np.prod(inner_space.shape)) + 2
            self.observation_space = spaces.Box(
                low   = float(inner_space.low.min()),
                high  = float(inner_space.high.max()),
                shape = (flat_dim,),
                dtype = np.float32,
            )
            self._obs_type = "flat"
        else:
            raise ValueError(f"Unsupported obs space type: {type(inner_space)}")

    def _read_signal(self) -> None:
        """Read signal type from unwrapped env grid and set one-hot vector."""
        inner = self.env.unwrapped
        grid  = inner.grid
        W, H  = grid.width, grid.height
        self._signal_vec[:] = 0.0
        for x in range(min(3, W)):
            for y in range(H):
                cell = grid.get(x, y)
                if cell is not None and hasattr(cell, "type"):
                    if cell.type == "key":
                        self._signal_vec[0] = 1.0
                        self._signal_type   = "key"
                        return
                    elif cell.type == "ball":
                        self._signal_vec[1] = 1.0
                        self._signal_type   = "ball"
                        return

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._read_signal()
        return self._augment(obs), info

    def observation(self, obs):
        return self._augment(obs)

    def _augment(self, obs) -> np.ndarray:
        if self._obs_type == "dict":
            img_flat = obs["image"].astype(np.float32).flatten()
        else:
            img_flat = obs.astype(np.float32).flatten()
        return np.concatenate([img_flat, self._signal_vec])


# ── Env factories ─────────────────────────────────────────────────────────────

def make_privileged_env_fn(alpha: float, shaping_c: float, seed: int = 0):
    """
    Train env: OracleCreditWrapper → PrivilegedSignalWrapper → Monitor.
    No FlatObsWrapper needed — PrivilegedSignalWrapper outputs a flat Box.
    """
    def _fn():
        env = gym.make(ENV_ID)
        env = OracleCreditWrapper(env, alpha=alpha, shaping_c=shaping_c)
        env = PrivilegedSignalWrapper(env)
        env = Monitor(env)
        return env
    return _fn


def make_privileged_eval_env_fn(seed: int = 0):
    """
    Eval env: privileged signal, NO oracle shaping.
    Measures: can PPO solve the task when memory is free?
    """
    def _fn():
        env = gym.make(ENV_ID)
        env = PrivilegedSignalWrapper(env)
        env = Monitor(env)
        return env
    return _fn

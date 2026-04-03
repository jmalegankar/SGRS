"""
dm_env → gymnasium adapter for bsuite DiscountingChain.

DiscountingChain mechanics:
  - 5 discrete actions (arms), each maps to a reward delay: [1, 3, 10, 30, 100] steps
  - At t=0 the agent picks an arm (context becomes that action for the episode)
  - Reward fires once at step reward_delay[context]: 1.0 for all arms, 1.1 for correct arm
  - episode_len = 100, obs = (context, time_fraction), shape (2,)
  - mapping_seed % 5 determines which arm is the "correct" (bonus) arm — FIXED per instance

The correct_arm and _timestep are exposed as public attributes for wrappers.
"""

import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

try:
    from bsuite.environments.discounting_chain import DiscountingChain
except ImportError:
    sys.exit("bsuite not found. Run: pip install bsuite")

REWARD_DELAYS = [1, 3, 10, 30, 100]
NUM_ARMS      = 5


class DiscountingChainGymEnv(gym.Env):
    """
    Gymnasium wrapper around bsuite DiscountingChain.

    Key public attributes (read by wrappers via env.unwrapped):
      correct_arm : int  — the arm that yields +0.1 bonus reward (fixed per instance)
      _timestep   : int  — current step within episode (0 before first step)
    """

    metadata = {}

    def __init__(self, mapping_seed: int = 0):
        super().__init__()
        self._mapping_seed  = mapping_seed
        self._bsuite_env    = DiscountingChain(mapping_seed=mapping_seed)
        self.correct_arm    = mapping_seed % NUM_ARMS   # fixed for this instance
        self.reward_delay   = REWARD_DELAYS[self.correct_arm]
        self._timestep      = 0

        # obs = [context, time_fraction]
        # context: -1 at t=0, then arm index (0-4) for remaining steps
        # time_fraction: _timestep / 100 in [0, 1]
        self.observation_space = spaces.Box(
            low   = np.array([-1.0, 0.0], dtype=np.float32),
            high  = np.array([ 4.0, 1.0], dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ARMS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._bsuite_env.reset()
        self._timestep = 0
        obs = ts.observation.flatten().astype(np.float32)
        return obs, {}

    def step(self, action):
        assert not self._bsuite_env._reset_next_step, (
            "step() called on a terminated env — call reset() first"
        )
        ts     = self._bsuite_env.step(int(action))
        obs    = ts.observation.flatten().astype(np.float32)
        reward = float(ts.reward) if ts.reward is not None else 0.0

        import dm_env
        terminated = ts.step_type == dm_env.StepType.LAST
        truncated  = False

        self._timestep = self._bsuite_env._timestep
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass


# ── Env factories ─────────────────────────────────────────────────────────────

def make_env_fn(
    mapping_seed   : int,
    use_oracle     : bool,
    use_privileged : bool,
    alpha          : float = 2.0,
    shaping_c      : float = 1.0,
    seed           : int   = 0,
):
    """
    Factory for training envs.
    Stacking order: DiscountingChainGymEnv → OracleCreditWrapper?
                    → PrivilegedSignalWrapper/passthrough → Monitor
    """
    from wrapper   import OracleCreditWrapper
    from privileged import PrivilegedSignalWrapper

    def _fn():
        env = DiscountingChainGymEnv(mapping_seed=mapping_seed)
        if use_oracle:
            env = OracleCreditWrapper(env, alpha=alpha, shaping_c=shaping_c)
        if use_privileged:
            env = PrivilegedSignalWrapper(env)
        env = Monitor(env)
        return env
    return _fn


def make_eval_env_fn(
    mapping_seed   : int,
    use_privileged : bool,
    seed           : int = 0,
):
    """
    Factory for eval envs — never oracle-shaped.
    use_privileged must match training condition so obs spaces align.
    """
    from privileged import PrivilegedSignalWrapper

    def _fn():
        env = DiscountingChainGymEnv(mapping_seed=mapping_seed)
        if use_privileged:
            env = PrivilegedSignalWrapper(env)
        env = Monitor(env)
        return env
    return _fn

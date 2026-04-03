"""
PrivilegedSignalWrapper for bsuite DiscountingChain.

Appends a 5-element one-hot [arm_0, arm_1, arm_2, arm_3, arm_4] to the obs
at EVERY timestep, encoding which arm is the correct (bonus) arm.

Since mapping_seed is fixed per env instance, correct_arm never changes across
episodes — the one-hot is constant. This is intentional: we are giving the policy
perfect free memory of the ground-truth, so it never needs to infer or remember
which arm is best. Memory is solved; only credit assignment remains.

Output observation space: (7,) = (context, time_frac, one_hot_5)
Bounds:
  obs[0] : context       in [-1, 4]
  obs[1] : time_frac     in [ 0, 1]
  obs[2:] : one-hot      in [ 0, 1]

Apply AFTER OracleCreditWrapper (does not change obs space visible to oracle).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PrivilegedSignalWrapper(gym.ObservationWrapper):
    """
    Appends correct-arm one-hot to the observation at every step.
    Must be applied AFTER OracleCreditWrapper, as the final obs wrapper.
    """

    NUM_ARMS = 5

    def __init__(self, env: gym.Env):
        super().__init__(env)

        base = env.observation_space  # Box(2,) from DiscountingChainGymEnv
        self._one_hot = np.zeros(self.NUM_ARMS, dtype=np.float32)

        self.observation_space = spaces.Box(
            low   = np.concatenate([base.low,  np.zeros(self.NUM_ARMS, np.float32)]),
            high  = np.concatenate([base.high, np.ones(self.NUM_ARMS,  np.float32)]),
            dtype = np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Recompute one-hot on reset (correct_arm is fixed but protocol-correct)
        self._one_hot[:] = 0.0
        self._one_hot[self.env.unwrapped.correct_arm] = 1.0
        return self.observation(obs), info

    def observation(self, obs):
        return np.concatenate([obs, self._one_hot])

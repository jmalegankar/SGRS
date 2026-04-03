"""
Oracle reward shaping wrapper for bsuite DiscountingChain.

The only decision that matters is arm selection at t=0 (the first step).
The natural env reward for that choice arrives reward_delay steps later
(1, 3, 10, 30, or 100 steps). At gamma=0.99 this means discounts of
0.99, 0.97, 0.90, 0.74, 0.37 — severe for long delays.

The oracle eliminates this credit gap by injecting a bonus immediately at t=0
if the correct arm was chosen:

  bonus = alpha * shaping_c   if action == correct_arm at t=0
        = 0                   otherwise (asymmetric — no punishment)

Shaped return landscape (alpha=2, shaping_c=1, correct arm = max reward 1.1):
  Correct arm: gamma^0 * 2.0 + gamma^delay * 1.1  ← oracle bonus + natural reward
  Wrong arm:   gamma^0 * 0   + gamma^delay * 1.0  ← no bonus, lower natural reward
  At delay=100: correct ≈ 2.0 + 0.37, wrong ≈ 0 + 0.37 × (wrong arm reward 1.0/1.1)
  The oracle makes the correct arm unambiguously better regardless of delay.

Info keys set by this wrapper:
  oracle_correct    : bool  — True if correct arm chosen (at t=0 steps only)
  oracle_bonus      : float — alpha * shaping_c (at t=0 steps, correct arm only)
  oracle_ep_bonus   : float — total oracle bonus for the episode (on terminal step)
"""

import gymnasium as gym


class OracleCreditWrapper(gym.Wrapper):
    """
    Asymmetric oracle shaping: bonus for correct arm at t=0, nothing for wrong arm.
    Apply BEFORE PrivilegedSignalWrapper.
    """

    def __init__(
        self,
        env      : gym.Env,
        alpha    : float = 2.0,
        shaping_c: float = 1.0,
        verbose  : bool  = False,
    ):
        super().__init__(env)
        self.alpha     = alpha
        self.shaping_c = shaping_c
        self.verbose   = verbose
        self._ep_bonus = 0.0

    def reset(self, **kwargs):
        obs, info      = self.env.reset(**kwargs)
        self._ep_bonus = 0.0
        return obs, info

    def step(self, action):
        inner  = self.env.unwrapped
        is_t0  = (inner._timestep == 0)   # read BEFORE inner step increments it

        obs, reward, terminated, truncated, info = self.env.step(action)

        bonus = 0.0
        if is_t0:
            correct = (int(action) == inner.correct_arm)
            info["oracle_correct"] = correct
            if correct:
                bonus = self.alpha * self.shaping_c
                info["oracle_bonus"] = bonus
            if self.verbose:
                arm_str = "CORRECT" if correct else "WRONG  "
                print(
                    f"[Oracle] t=0  action={action}  correct_arm={inner.correct_arm}"
                    f"  {arm_str}  bonus={bonus:+.2f}"
                )

        self._ep_bonus += bonus
        if terminated or truncated:
            info["oracle_ep_bonus"] = self._ep_bonus
            self._ep_bonus = 0.0

        return obs, reward + bonus, terminated, truncated, info

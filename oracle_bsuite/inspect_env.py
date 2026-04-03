#!/usr/bin/env python3
"""
Inspect and verify oracle_bsuite environment stack.

Checks:
  1. Base env: obs shape, action space, reward fires at correct delay
  2. OracleCreditWrapper: bonus fires at t=0 for correct arm only
  3. PrivilegedSignalWrapper: one-hot appended correctly
  4. Wrong arm: no oracle bonus, oracle_correct=False
  5. Auto-reset safety: terminated env requires reset() before step()

Run from oracle_bsuite/:
  python inspect_env.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from envs       import DiscountingChainGymEnv, REWARD_DELAYS
from wrapper    import OracleCreditWrapper
from privileged import PrivilegedSignalWrapper

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(cond, msg):
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {msg}")
    return cond


def inspect_base(mapping_seed=2):
    print(f"\n── Base env (mapping_seed={mapping_seed}) ─────────────────────────")
    env = DiscountingChainGymEnv(mapping_seed=mapping_seed)
    check(env.observation_space.shape == (2,), f"obs shape == (2,): {env.observation_space.shape}")
    check(env.action_space.n == 5,             f"action_space.n == 5: {env.action_space.n}")
    check(env.correct_arm == mapping_seed % 5, f"correct_arm == {mapping_seed % 5}: {env.correct_arm}")

    delay = REWARD_DELAYS[env.correct_arm]
    print(f"  correct_arm={env.correct_arm}  reward_delay={delay}")

    obs, _ = env.reset()
    check(float(obs[0]) == -1.0, f"obs[0]==-1 at t=0: {obs[0]}")
    check(float(obs[1]) == 0.0,  f"obs[1]==0 at t=0: {obs[1]}")

    # Step through full episode with correct arm, verify reward fires at delay
    obs, _ = env.reset()
    total_reward = 0.0
    reward_at_step = {}
    for t in range(1, 101):
        action = env.correct_arm if t == 1 else 0
        obs, r, terminated, truncated, info = env.step(action)
        if r != 0.0:
            reward_at_step[t] = r
        total_reward += r
        if terminated:
            break

    check(delay in reward_at_step, f"Reward fires at step {delay}: {reward_at_step}")
    check(abs(total_reward - 1.1) < 1e-6, f"Total reward ≈ 1.1 (correct arm): {total_reward:.4f}")
    env.close()


def inspect_oracle(mapping_seed=2):
    print(f"\n── OracleCreditWrapper (mapping_seed={mapping_seed}) ───────────────")
    base = DiscountingChainGymEnv(mapping_seed=mapping_seed)
    env  = OracleCreditWrapper(base, alpha=2.0, shaping_c=1.0, verbose=True)

    correct_arm  = base.correct_arm
    wrong_arm    = (correct_arm + 1) % 5

    # Correct arm: should get bonus at t=0
    obs, _ = env.reset()
    obs, r, terminated, truncated, info = env.step(correct_arm)
    check("oracle_correct" in info,        "oracle_correct in info at t=0")
    check(info.get("oracle_correct") is True, f"oracle_correct==True for correct arm")
    check("oracle_bonus" in info,          "oracle_bonus in info for correct arm")
    check(abs(info.get("oracle_bonus", 0) - 2.0) < 1e-6,
          f"oracle_bonus==2.0 (alpha=2, shaping_c=1): {info.get('oracle_bonus')}")
    # No more bonus after t=0
    prev_bonus = info.get("oracle_bonus", 0.0)
    for _ in range(5):
        obs, r, terminated, truncated, info = env.step(0)
        check("oracle_bonus" not in info, "No oracle_bonus on subsequent steps")

    # Wrong arm: no bonus
    obs, _ = env.reset()
    obs, r, terminated, truncated, info = env.step(wrong_arm)
    check(info.get("oracle_correct") is False, "oracle_correct==False for wrong arm")
    check("oracle_bonus" not in info,           "No oracle_bonus for wrong arm")

    # oracle_ep_bonus on terminal step
    obs, _ = env.reset()
    obs, r, terminated, truncated, info = env.step(correct_arm)
    while not terminated:
        obs, r, terminated, truncated, info = env.step(0)
    check("oracle_ep_bonus" in info, f"oracle_ep_bonus on terminal step: {info.get('oracle_ep_bonus')}")
    check(abs(info["oracle_ep_bonus"] - 2.0) < 1e-6,
          f"oracle_ep_bonus==2.0 (correct arm, one bonus): {info['oracle_ep_bonus']}")
    env.close()


def inspect_privileged(mapping_seed=3):
    print(f"\n── PrivilegedSignalWrapper (mapping_seed={mapping_seed}) ────────────")
    base = DiscountingChainGymEnv(mapping_seed=mapping_seed)
    env  = PrivilegedSignalWrapper(base)

    check(env.observation_space.shape == (7,),
          f"obs space shape == (7,): {env.observation_space.shape}")

    obs, _ = env.reset()
    check(obs.shape == (7,), f"obs shape == (7,): {obs.shape}")

    one_hot = obs[2:]
    correct = base.correct_arm
    check(one_hot[correct] == 1.0,      f"one_hot[{correct}]==1.0: {one_hot}")
    check(one_hot.sum() == 1.0,         f"one_hot sums to 1: {one_hot.sum()}")
    check(float(obs[0]) == -1.0,        f"context==-1 at t=0: {obs[0]}")

    # One-hot persists through steps
    obs, r, terminated, truncated, info = env.step(correct)
    one_hot2 = obs[2:]
    check((one_hot2 == one_hot).all(),  f"one-hot unchanged after step: {one_hot2}")
    env.close()


def inspect_stacked(mapping_seed=4):
    print(f"\n── Full stack: Oracle + Privileged (mapping_seed={mapping_seed}) ───")
    from stable_baselines3.common.monitor import Monitor
    base = DiscountingChainGymEnv(mapping_seed=mapping_seed)
    env  = OracleCreditWrapper(base, alpha=2.0, shaping_c=1.0)
    env  = PrivilegedSignalWrapper(env)
    env  = Monitor(env)

    check(env.observation_space.shape == (7,),
          f"stacked obs shape == (7,): {env.observation_space.shape}")

    obs, _ = env.reset()
    correct_arm = base.correct_arm
    obs, r, terminated, truncated, info = env.step(correct_arm)
    check(r == 2.0 + 0.0,  # oracle bonus only (delay=100, natural reward not yet)
          f"shaped reward at t=0 (correct arm, delay=100) == 2.0: {r:.4f}")
    check(obs.shape == (7,), f"stacked obs shape (7,) after step: {obs.shape}")
    print(f"  delay for mapping_seed={mapping_seed}: {REWARD_DELAYS[correct_arm]} steps")
    env.close()


def main():
    print("=" * 60)
    print("  oracle_bsuite environment inspection")
    print("=" * 60)
    inspect_base(mapping_seed=2)
    inspect_oracle(mapping_seed=2)
    inspect_privileged(mapping_seed=3)
    inspect_stacked(mapping_seed=4)
    print("\n" + "=" * 60)
    print("  Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

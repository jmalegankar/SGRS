"""
Environment inspection for Milestone 0.

Usage:
  python -m oracle.inspect          # uses default seed 42
  python train.py --inspect
"""

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from minigrid.core.world_object import Ball, Box, Goal, Key, Wall

from oracle.envs import ENV_ID
from oracle.wrapper import OracleCreditWrapper


def inspect_env(seed: int = 42) -> None:
    """
    Print the grid layout and walk a short random episode to verify the oracle
    wrapper fires at the correct timesteps.  Run this before any training.
    """
    print(f"\n{'='*62}")
    print(f"  MiniGrid-MemoryS7-v0  —  Environment Inspection")
    print(f"{'='*62}\n")

    # ── raw grid ──────────────────────────────────────────────────────────────
    raw_env = gym.make(ENV_ID)
    raw_env.reset(seed=seed)
    inner = raw_env.unwrapped

    W, H = inner.grid.width, inner.grid.height
    print(f"  Grid : {W} × {H}")
    print(f"  Agent: pos={tuple(inner.agent_pos)}  dir={inner.agent_dir}")
    print(f"  Target color: {getattr(inner, 'target_color', 'N/A')}\n")

    print("  Legend:  @ agent   # wall   . floor  K/B key/ball  G goal")
    print()
    for y in range(H):
        row = f"  y={y:02d} |"
        for x in range(W):
            if (x, y) == tuple(inner.agent_pos):
                row += " @ "
            else:
                cell = inner.grid.get(x, y)
                if cell is None:
                    row += " . "
                elif isinstance(cell, Wall):
                    row += " # "
                elif isinstance(cell, (Key, Ball, Box)):
                    row += f"{cell.type[0].upper()}{cell.color[0].upper()} "
                elif isinstance(cell, Goal):
                    row += " G "
                else:
                    row += " ? "
        print(row)
    raw_env.close()

    # ── oracle wrapper validation ─────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Oracle Wrapper Validation  (verbose=True, running 60 random steps)")
    print(f"{'─'*62}\n")

    oracle_env = OracleCreditWrapper(
        gym.make(ENV_ID),
        alpha=1.0,
        shaping_c=1.0,
        verbose=True,
    )
    flat_env = FlatObsWrapper(oracle_env)
    flat_env.reset(seed=seed)

    print(f"  Wrapper state after reset:")
    print(f"    corridor_y     = {oracle_env._corridor_y}")
    print(f"    junction_x ≥   = {oracle_env._junction_x}")
    print(f"    correct_pos    = {oracle_env._correct_pos}")
    print(f"    wrong_pos      = {oracle_env._wrong_pos}")
    print(f"    target_color   = {oracle_env._target_color}\n")

    total_bonus = 0.0
    for t in range(60):
        action = flat_env.action_space.sample()
        _, reward, terminated, truncated, info = flat_env.step(action)
        bonus = info.get("oracle_bonus", 0.0)
        if bonus != 0:
            total_bonus += bonus
            print(
                f"  step={t:3d}  reward={reward:.3f}  oracle_bonus={bonus:+.3f}  "
                f"info={info}"
            )
        if terminated or truncated:
            print(
                f"\n  Episode ended at step={t}  terminated={terminated}  "
                f"truncated={truncated}  total_oracle_bonus={total_bonus:.2f}"
            )
            break

    print(f"\n  Events log: {oracle_env.events}")
    flat_env.close()

    print(f"\n{'─'*62}")
    print("  Observation space:", flat_env.observation_space)
    print("  Action space     :", flat_env.action_space)
    print(f"{'─'*62}\n")
    print("  Verify:")
    print("    • 'SIGNAL' fires at step 0")
    print("    • 'JUNCTION' fires once the agent enters a non-corridor row")
    print("      in the right zone (x >= junction_x)")
    print("    • CORRECT/WRONG matches the terminal episode outcome")
    print()


if __name__ == "__main__":
    inspect_env()

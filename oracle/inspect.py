"""
Environment inspection for Milestone 0.

Verifies:
  1. SIGNAL fires at step 0
  2. JUNCTION fires exactly once when agent enters the arm (not at random times)
  3. CORRECT/WRONG matches the terminal episode outcome

Also renders key frames to PNG for visual verification.

Usage:
  python -m oracle.inspect           # seed 42, renders to oracle/renders/
  python train.py --inspect
"""

import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from minigrid.core.world_object import Ball, Box, Goal, Key, Wall
from minigrid.wrappers import FlatObsWrapper

from oracle.envs import ENV_ID
from oracle.wrapper import OracleCreditWrapper


# ── ASCII grid printer ────────────────────────────────────────────────────────

def _print_grid(inner, label: str = "") -> None:
    grid = inner.grid
    W, H = grid.width, grid.height
    if label:
        print(f"\n  {label}")
    print(f"  Grid: {W}×{H}   agent_pos={tuple(inner.agent_pos)}   "
          f"agent_dir={inner.agent_dir}   "
          f"signal_type={_signal_type(inner)}")
    print()
    print(f"  Legend:  @ agent   # wall   K key   B ball   G goal   . floor")
    print()
    for y in range(H):
        row = f"  y={y} |"
        for x in range(W):
            if (x, y) == tuple(inner.agent_pos):
                DIRS = {0: "→", 1: "↓", 2: "←", 3: "↑"}
                row += f" {DIRS.get(inner.agent_dir, '@')} "
            else:
                cell = grid.get(x, y)
                if cell is None:
                    row += " . "
                elif isinstance(cell, Wall):
                    row += " # "
                elif isinstance(cell, (Key, Ball, Box)):
                    mark = f"[{cell.type[0].upper()}]"
                    # Mark the success/failure positions
                    sp = getattr(inner, "success_pos", None)
                    fp = getattr(inner, "failure_pos", None)
                    # The junction arms are one step from the objects
                    row += f"{cell.type[0].upper()}{cell.color[0].upper()} "
                elif isinstance(cell, Goal):
                    row += " G "
                else:
                    row += " ? "
        # Annotate oracle positions on the right
        ann = ""
        sp = getattr(inner, "success_pos", None)
        fp = getattr(inner, "failure_pos", None)
        corridor_y = inner.agent_pos[1]  # approx
        if sp is not None and sp[1] == y:
            ann = "  ← SUCCESS arm"
        if fp is not None and fp[1] == y:
            ann = "  ← FAILURE arm"
        print(row + ann)
    print()
    if hasattr(inner, "success_pos"):
        print(f"  success_pos={tuple(inner.success_pos)}   "
              f"failure_pos={tuple(inner.failure_pos)}")
    print()


def _signal_type(inner) -> str:
    """Return the type of the signal object (leftmost Key/Ball in the grid)."""
    grid = inner.grid
    for x in range(min(3, grid.width)):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and hasattr(cell, "type") and cell.type in ("key", "ball"):
                return f"{cell.type} ({cell.color})"
    return "unknown"


# ── Rendering ────────────────────────────────────────────────────────────────

def _render_frame(env_rgb, ax, title: str, agent_pos, step: int) -> None:
    ax.imshow(env_rgb)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def render_episode(
    seed: int = 42,
    go_correct: bool = True,
    out_dir: Path = Path("oracle/renders"),
) -> None:
    """
    Run a scripted episode that visits BOTH oracle events, capture key frames,
    save as a PNG strip and individual PNGs.

    go_correct=True  → turn toward success arm  (junction F=+1, env reward > 0)
    go_correct=False → turn toward failure arm  (junction F=-1, env reward = 0)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # MiniGrid dir: 0=right 1=down 2=left 3=up
    # Actions:      0=turn_left 1=turn_right 2=forward
    FORWARD    = 2
    TURN_LEFT  = 0
    TURN_RIGHT = 1

    oracle_env = OracleCreditWrapper(
        gym.make(ENV_ID, render_mode="rgb_array"),
        alpha=1.0,
        shaping_c=1.0,
        verbose=True,
    )
    oracle_env.reset(seed=seed)

    inner        = oracle_env.unwrapped
    corridor_y   = int(inner.agent_pos[1])
    success_y    = int(inner.success_pos[1])
    failure_y    = int(inner.failure_pos[1])
    target_x     = int(inner.success_pos[0])   # junction objects' x (= hallway_end + 1)
    signal_label = _signal_type(inner)

    # Which arm are we steering toward?
    arm_y        = success_y if go_correct else failure_y
    arm_label    = "SUCCESS" if go_correct else "FAILURE"

    # To reach arm_y from corridor_y:
    #   arm_y > corridor_y → need to go down → turn right  (dir 0→1)
    #   arm_y < corridor_y → need to go up   → turn left   (dir 0→3)
    go_down = arm_y > corridor_y

    frames: list = []
    titles: list = []

    def capture(title: str) -> None:
        rgb = oracle_env.unwrapped.render()
        frames.append(rgb.copy())
        titles.append(title)

    # ── t=0: initial frame before any action ─────────────────────────────────
    capture(f"t=0  Start\nsignal={signal_label}")

    # ── Walk right to junction center ─────────────────────────────────────────
    step = 0
    bonus_total = 0.0
    terminated = truncated = False

    while int(inner.agent_pos[0]) < target_x and not (terminated or truncated):
        _, reward, terminated, truncated, info = oracle_env.step(FORWARD)
        step += 1
        b = info.get("oracle_bonus", 0.0)
        bonus_total += b
        if b != 0.0:
            capture(f"t={step}  SIGNAL fires\nF={b:+.2f}")

    capture(f"t={step}  Corridor end\n(x={inner.agent_pos[0]}, y={inner.agent_pos[1]})")

    # ── Turn toward chosen arm ────────────────────────────────────────────────
    oracle_env.step(TURN_RIGHT if go_down else TURN_LEFT)
    step += 1
    capture(f"t={step}  Turned toward\n{arm_label} arm (y={arm_y})")

    # ── Step into the arm — junction fires here ───────────────────────────────
    _, reward, terminated, truncated, info = oracle_env.step(FORWARD)
    step += 1
    b = info.get("oracle_bonus", 0.0)
    bonus_total += b
    capture(
        f"t={step}  JUNCTION fires\nagent_y={inner.agent_pos[1]}  F={b:+.2f}"
    )

    # ── Walk to episode end ───────────────────────────────────────────────────
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = oracle_env.step(FORWARD)
        step += 1

    outcome_label = "SUCCESS  r>0" if reward > 0 else "FAILURE  r=0"
    capture(f"t={step}  Episode ends\n{outcome_label}")

    print(f"\n[render:{arm_label}] {step} steps  "
          f"total_oracle_bonus={bonus_total:.2f}  final_env_reward={reward:.3f}")
    print(f"[render:{arm_label}] Events: {oracle_env.events}")

    # ── Save PNG strip ────────────────────────────────────────────────────────
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    title_color = "#1a7a1a" if go_correct else "#cc0000"
    for ax, frame, title in zip(axes, frames, titles):
        ax.imshow(frame)
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis("off")

    fig.suptitle(
        f"MiniGrid-MemoryS7-v0  |  Oracle Inspection ({arm_label} path)  |  seed={seed}\n"
        f"signal={signal_label}   success_arm_y={success_y}   failure_arm_y={failure_y}   "
        f"total_oracle_bonus={bonus_total:.2f}",
        fontsize=10, fontweight="bold", color=title_color,
    )
    plt.tight_layout()
    tag       = "success" if go_correct else "failure"
    strip_path = out_dir / f"oracle_{tag}_seed{seed}.png"
    plt.savefig(strip_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[render:{arm_label}] Saved strip → {strip_path}")

    for i, (frame, title) in enumerate(zip(frames, titles)):
        Image.fromarray(frame).save(out_dir / f"frame_{tag}_{i:02d}_seed{seed}.png")

    oracle_env.close()


# ── Full oracle validation walk ───────────────────────────────────────────────

def _run_oracle_validation(seed: int, n_steps: int = 200) -> None:
    """
    Run a random-policy walk for n_steps with verbose oracle.
    Separately also run a scripted walk that is guaranteed to reach the junction.
    """
    print(f"\n{'─'*62}")
    print(f"  Random-policy walk ({n_steps} steps) — may or may not reach junction")
    print(f"{'─'*62}\n")

    oracle_env = OracleCreditWrapper(
        gym.make(ENV_ID),
        alpha=1.0,
        shaping_c=1.0,
        verbose=True,
    )
    flat_env = FlatObsWrapper(oracle_env)
    flat_env.reset(seed=seed)

    total_bonus = 0.0
    for t in range(n_steps):
        action = flat_env.action_space.sample()
        _, reward, terminated, truncated, info = flat_env.step(action)
        bonus = info.get("oracle_bonus", 0.0)
        if bonus != 0.0:
            total_bonus += bonus
            print(
                f"  step={t:4d}  reward={reward:.3f}  "
                f"oracle_bonus={bonus:+.3f}  info_keys={list(info.keys())}"
            )
        if terminated or truncated:
            print(f"\n  Episode ended: t={t}  total_oracle_bonus={total_bonus:.2f}")
            break

    print(f"\n  Oracle events log: {oracle_env.events}")
    flat_env.close()

    # ── Scripted walk to guarantee junction fires ─────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Scripted walk — guaranteed to reach junction (correct arm)")
    print(f"{'─'*62}\n")

    FORWARD, TURN_LEFT, TURN_RIGHT = 2, 0, 1

    oracle_env2 = OracleCreditWrapper(
        gym.make(ENV_ID),
        alpha=1.0,
        shaping_c=1.0,
        verbose=True,
    )
    flat_env2 = FlatObsWrapper(oracle_env2)
    flat_env2.reset(seed=seed)

    inner      = oracle_env2.unwrapped
    target_x   = int(inner.success_pos[0])   # junction center x (objects' x)
    success_y  = int(inner.success_pos[1])
    corridor_y = int(inner.agent_pos[1])
    go_down    = success_y > corridor_y

    total_bonus2 = 0.0
    step = 0

    # Walk right to junction center (target_x = hallway_end + 1)
    while int(inner.agent_pos[0]) < target_x:
        _, reward, terminated, truncated, info = flat_env2.step(FORWARD)
        total_bonus2 += info.get("oracle_bonus", 0.0)
        step += 1
        if terminated or truncated:
            break

    # Turn toward success arm
    flat_env2.step(TURN_RIGHT if go_down else TURN_LEFT)
    step += 1

    # Step into arm
    _, reward, terminated, truncated, info = flat_env2.step(FORWARD)
    total_bonus2 += info.get("oracle_bonus", 0.0)
    step += 1

    # Walk to end
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = flat_env2.step(FORWARD)
        step += 1
        total_bonus2 += info.get("oracle_bonus", 0.0)

    outcome = "SUCCESS" if reward > 0 else "FAILURE"
    print(f"\n  Scripted episode: {step} steps  outcome={outcome}  "
          f"total_oracle_bonus={total_bonus2:.2f}")
    print(f"  Oracle events: {oracle_env2.events}")

    # Verify events
    events = oracle_env2.events
    signal_events   = [e for e in events if e["event"] == "signal"]
    junction_events = [e for e in events if e["event"] == "junction"]

    print(f"\n{'─'*62}")
    print("  VERIFICATION CHECKS")
    print(f"{'─'*62}")
    _check("SIGNAL fired exactly once",     len(signal_events) == 1)
    _check("SIGNAL fired at step 0",         signal_events and signal_events[0]["step"] == 0)
    _check("JUNCTION fired exactly once",   len(junction_events) == 1)
    if junction_events:
        junc = junction_events[0]
        _check("JUNCTION fired in junction zone (x >= junction_x after step 0)",
               junc["step"] > 0)
        _check("JUNCTION marked CORRECT (scripted → success arm)",
               junc["correct"] is True)
        _check("JUNCTION outcome matches episode outcome (SUCCESS)",
               outcome == "SUCCESS")
    flat_env2.close()


def _check(label: str, passed: bool) -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}]  {label}")


# ── Main inspection ───────────────────────────────────────────────────────────

def inspect_env(seed: int = 42, render: bool = True) -> None:
    print(f"\n{'='*62}")
    print(f"  MiniGrid-MemoryS7-v0  —  Environment Inspection")
    print(f"{'='*62}\n")

    # ── raw grid ──────────────────────────────────────────────────────────────
    raw_env = gym.make(ENV_ID)
    raw_env.reset(seed=seed)
    inner = raw_env.unwrapped
    _print_grid(inner, label="Initial grid state")
    raw_env.close()

    # ── obs / action spaces ───────────────────────────────────────────────────
    tmp = FlatObsWrapper(gym.make(ENV_ID))
    tmp.reset(seed=seed)
    print(f"  Observation space: {tmp.observation_space}")
    print(f"  Action space     : {tmp.action_space}")
    tmp.close()

    # ── oracle wrapper state ─────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Oracle Wrapper State  (verbose=True, after reset)")
    print(f"{'─'*62}\n")

    oracle_env = OracleCreditWrapper(gym.make(ENV_ID), verbose=True)
    FlatObsWrapper(oracle_env).reset(seed=seed)
    print(f"  _corridor_y    = {oracle_env._corridor_y}")
    print(f"  _junction_x    = {oracle_env._junction_x}")
    print(f"  _signal_type   = {oracle_env._signal_type}")
    print(f"  _success_pos   = {oracle_env._success_pos}")
    print(f"  _failure_pos   = {oracle_env._failure_pos}")
    print(f"  _correct_arm_y = {oracle_env._correct_arm_y}")
    print(f"  _wrong_arm_y   = {oracle_env._wrong_arm_y}")
    oracle_env.env.close()

    # ── oracle event validation ───────────────────────────────────────────────
    _run_oracle_validation(seed)

    # ── rendering ─────────────────────────────────────────────────────────────
    if render:
        print(f"\n{'─'*62}")
        print("  Rendering: SUCCESS path (go_correct=True)")
        print(f"{'─'*62}\n")
        render_episode(seed=seed, go_correct=True)

        print(f"\n{'─'*62}")
        print("  Rendering: FAILURE path (go_correct=False)  — expect F=-1")
        print(f"{'─'*62}\n")
        render_episode(seed=seed, go_correct=False)

    print(f"\n{'='*62}")
    print("  Inspection complete.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    inspect_env(seed=seed)

"""
Oracle reward shaping wrapper for MiniGrid Memory environments.

Injects dense credit at the two known decision points:
  1. Signal step  (t=0): agent acts on the signal object it starts in front of.
  2. Junction step: first time agent enters a non-corridor row in the T-zone.
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym

from minigrid.core.world_object import Ball, Box, Key


class OracleCreditWrapper(gym.Wrapper):
    """
    Oracle reward shaping for MiniGrid Memory environments.

    At reset() we scan the grid to locate:
      • corridor_y   — the row the agent starts on (= the horizontal corridor)
      • junction_x   — x threshold past which the T-junction arms begin
      • correct_pos  — (x,y) of the object matching target_color
      • wrong_pos    — (x,y) of the other choice object

    Then at each step() we inject F_t at the two oracle events.

    NOTE: Apply this wrapper BEFORE FlatObsWrapper so we can still
    call env.unwrapped to inspect grid positions.
    """

    CHOICE_TYPES = (Key, Ball, Box)

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 1.0,
        shaping_c: float = 1.0,
        junction_x_frac: float = 0.60,
        verbose: bool = False,
    ):
        super().__init__(env)
        self.alpha            = alpha
        self.shaping_c        = shaping_c
        self.junction_x_frac  = junction_x_frac
        self.verbose          = verbose

        self._step          : int                    = 0
        self._corridor_y    : Optional[int]          = None
        self._junction_x    : Optional[int]          = None
        self._target_color  : Optional[str]          = None
        self._correct_pos   : Optional[Tuple[int,int]] = None
        self._wrong_pos     : Optional[Tuple[int,int]] = None
        self._junction_done : bool                   = False

        self.events         : List[Dict]             = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_inner(self) -> gym.Env:
        return self.env.unwrapped

    def _scan_grid(self) -> bool:
        """Scan the grid at reset. Returns True if correct_pos was found."""
        inner = self._get_inner()
        grid  = inner.grid
        W, H  = grid.width, grid.height

        self._corridor_y   = int(inner.agent_pos[1])
        self._junction_x   = int(W * self.junction_x_frac)
        self._target_color = getattr(inner, "target_color", None)

        if self._target_color is None:
            self._target_color = self._infer_target_color(inner, W, H)

        self._correct_pos = None
        self._wrong_pos   = None

        for x in range(W):
            for y in range(H):
                cell = grid.get(x, y)
                if cell is None or not isinstance(cell, self.CHOICE_TYPES):
                    continue
                if x > W // 2:
                    if cell.color == self._target_color:
                        self._correct_pos = (x, y)
                    else:
                        self._wrong_pos = (x, y)

        ok = self._correct_pos is not None
        if self.verbose:
            status = "OK" if ok else "WARNING: correct_pos not found"
            print(
                f"[Oracle] reset scan {status} | "
                f"target={self._target_color} | "
                f"correct={self._correct_pos} | "
                f"wrong={self._wrong_pos} | "
                f"corridor_y={self._corridor_y} | "
                f"junction_x≥{self._junction_x}"
            )
        return ok

    def _infer_target_color(self, inner, W: int, H: int) -> Optional[str]:
        """Fallback: infer target color from the closest choice object to the agent."""
        grid    = inner.grid
        agent_x = int(inner.agent_pos[0])
        best_cell, best_dist = None, float("inf")
        for x in range(W):
            for y in range(H):
                cell = grid.get(x, y)
                if isinstance(cell, self.CHOICE_TYPES):
                    dist = abs(x - agent_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = cell
        return best_cell.color if best_cell is not None else None

    def _is_correct_direction(self, agent_y: int) -> bool:
        """Did the agent move toward the correct choice object at the junction?"""
        if self._correct_pos is None:
            return True  # can't determine; give benefit of the doubt

        correct_y = self._correct_pos[1]
        wrong_y   = (
            self._wrong_pos[1]
            if self._wrong_pos is not None
            else (self._corridor_y + 1 if self._corridor_y is not None else correct_y + 2)
        )
        return abs(agent_y - correct_y) <= abs(agent_y - wrong_y)

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info        = self.env.reset(**kwargs)
        self._step       = 0
        self._junction_done = False
        self.events      = []
        self._scan_grid()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        inner     = self._get_inner()
        agent_pos = (int(inner.agent_pos[0]), int(inner.agent_pos[1]))
        bonus     = 0.0

        # Event 1: Signal Observation — fire on first action (step 0)
        if self._step == 0:
            bonus += self.shaping_c
            self.events.append({"step": self._step, "event": "signal", "bonus": self.shaping_c})
            if self.verbose:
                print(f"[Oracle] t={self._step:4d}  SIGNAL   F={self.shaping_c:+.2f}")

        # Event 2: T-Junction Commitment — first non-corridor row inside junction zone
        if (
            not self._junction_done
            and self._junction_x is not None
            and agent_pos[0] >= self._junction_x
            and agent_pos[1] != self._corridor_y
        ):
            correct    = self._is_correct_direction(agent_pos[1])
            junc_bonus = self.shaping_c if correct else -self.shaping_c
            bonus     += junc_bonus
            self._junction_done = True
            self.events.append({
                "step"   : self._step,
                "event"  : "junction",
                "correct": correct,
                "bonus"  : junc_bonus,
                "agent_y": agent_pos[1],
            })
            info["oracle_junction_correct"] = correct
            info["oracle_junction_step"]    = self._step
            if self.verbose:
                direction = "UP" if agent_pos[1] < (self._corridor_y or 0) else "DOWN"
                print(
                    f"[Oracle] t={self._step:4d}  JUNCTION {direction:4s} "
                    f"{'CORRECT' if correct else 'WRONG  '}  F={junc_bonus:+.2f}"
                )

        self._step   += 1
        shaped_reward = reward + self.alpha * bonus
        if bonus != 0.0:
            info["oracle_bonus"] = self.alpha * bonus
        return obs, shaped_reward, terminated, truncated, info

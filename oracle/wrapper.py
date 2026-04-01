"""
Oracle reward shaping wrapper for MiniGrid Memory environments.

KEY INSIGHT (from env source inspection):
  All objects in MemoryS7 are GREEN. The distinguishing feature is object TYPE
  (Key vs Ball). The env exposes success_pos / failure_pos directly, which we
  use as ground-truth for the junction detection rather than inferring from
  color or type scanning.

Two dense bonuses are injected:
  1. Signal step  (t=0): agent's first action — the moment the policy must encode
                         what it saw. F = +shaping_c always.
  2. Junction step: first time agent enters a junction arm (y != corridor_y,
                    x >= junction_x). F = +shaping_c if correct arm, else -shaping_c.

Total shaped reward: r̃_t = r_t_env + alpha * F_t
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym


class OracleCreditWrapper(gym.Wrapper):
    """
    Oracle reward shaping for MiniGrid Memory environments.

    At reset() we read success_pos and failure_pos directly from the unwrapped
    env (set by MiniGrid's _gen_grid) to identify the two junction arms.

    Apply BEFORE FlatObsWrapper so env.unwrapped is still accessible.
    """

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 1.0,
        shaping_c: float = 1.0,
        junction_x_frac: float = 0.60,   # junction zone: x >= int(W * frac)
        verbose: bool = False,
    ):
        super().__init__(env)
        self.alpha           = alpha
        self.shaping_c       = shaping_c
        self.junction_x_frac = junction_x_frac
        self.verbose         = verbose

        # Set in reset():
        self._step           : int                       = 0
        self._corridor_y     : Optional[int]             = None
        self._junction_x     : Optional[int]             = None
        self._correct_arm_y  : Optional[int]             = None   # success_pos[1]
        self._wrong_arm_y    : Optional[int]             = None   # failure_pos[1]
        self._signal_type    : Optional[str]             = None   # "key" or "ball"
        self._junction_done  : bool                      = False

        # Kept for external inspection / diagnostics
        self._success_pos    : Optional[Tuple[int, int]] = None
        self._failure_pos    : Optional[Tuple[int, int]] = None
        self.events          : List[Dict]                = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_inner(self) -> gym.Env:
        return self.env.unwrapped

    def _scan_grid(self) -> bool:
        """
        Read oracle ground-truth from the env after reset.
        Returns True when success_pos was found (detection healthy).
        """
        inner = self._get_inner()
        grid  = inner.grid
        W, H  = grid.width, grid.height

        self._corridor_y = int(inner.agent_pos[1])
        self._junction_x = int(W * self.junction_x_frac)

        # Ground-truth arm positions, set by MiniGrid's _gen_grid
        if hasattr(inner, "success_pos") and hasattr(inner, "failure_pos"):
            self._success_pos   = (int(inner.success_pos[0]), int(inner.success_pos[1]))
            self._failure_pos   = (int(inner.failure_pos[0]), int(inner.failure_pos[1]))
            self._correct_arm_y = self._success_pos[1]
            self._wrong_arm_y   = self._failure_pos[1]
        else:
            # Fallback: shouldn't happen for MiniGrid MemoryEnv
            self._success_pos   = None
            self._failure_pos   = None
            self._correct_arm_y = None
            self._wrong_arm_y   = None

        # Find signal type (object nearest to agent in the start room, x <= 2)
        self._signal_type = None
        agent_x = int(inner.agent_pos[0])
        for x in range(min(3, W)):
            for y in range(H):
                cell = grid.get(x, y)
                if cell is not None and hasattr(cell, "type") and cell.type in ("key", "ball"):
                    self._signal_type = cell.type

        ok = self._correct_arm_y is not None
        if self.verbose:
            status = "OK" if ok else "WARNING: success_pos not found"
            print(
                f"[Oracle] reset scan {status} | "
                f"signal={self._signal_type} | "
                f"success_pos={self._success_pos} | "
                f"failure_pos={self._failure_pos} | "
                f"corridor_y={self._corridor_y} | "
                f"junction_x≥{self._junction_x}"
            )
        return ok

    def _is_correct_direction(self, agent_y: int) -> bool:
        """Did the agent move toward the correct (success) junction arm?"""
        if self._correct_arm_y is None:
            return True  # can't determine; give benefit of the doubt
        return agent_y == self._correct_arm_y

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

        # ── Event 1: Signal Observation ── fires on first action (step 0)
        if self._step == 0:
            bonus += self.shaping_c
            self.events.append({"step": self._step, "event": "signal", "bonus": self.shaping_c})
            if self.verbose:
                print(f"[Oracle] t={self._step:4d}  SIGNAL({self._signal_type})  F={self.shaping_c:+.2f}")

        # ── Event 2: T-Junction Commitment ──
        # First time agent enters a non-corridor row inside the junction zone.
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
                arm = "SUCCESS" if correct else "FAILURE"
                print(
                    f"[Oracle] t={self._step:4d}  JUNCTION→{arm:7s}  "
                    f"agent_y={agent_pos[1]}  "
                    f"correct_arm_y={self._correct_arm_y}  "
                    f"F={junc_bonus:+.2f}"
                )

        self._step    += 1
        shaped_reward  = reward + self.alpha * bonus
        if bonus != 0.0:
            info["oracle_bonus"] = self.alpha * bonus
        return obs, shaped_reward, terminated, truncated, info

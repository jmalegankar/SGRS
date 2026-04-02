"""
Oracle reward shaping wrapper for MiniGrid Memory environments — v2.

KEY INSIGHT (from env source inspection):
  All objects in MemoryS7 are GREEN. The distinguishing feature is object TYPE
  (Key vs Ball). The env exposes success_pos / failure_pos directly, which we
  use as ground-truth for the junction detection rather than inferring from
  color or type scanning.

Junction bonus is ASYMMETRIC: +shaping_c for correct arm, 0 for wrong arm.
The previous -shaping_c on wrong arm caused value function collapse at alpha=5
because shaped returns swung from +6 to -5 (explained_variance ≈ 0 throughout).

Shaped return landscape (default alpha=2, shaping_c=1):
  Correct arm:   0 + 2*1 + 1 = +3.0   ← clear winner
  Wrong arm:     0 + 2*0 + 0 =  0.0   ← same as timeout (no punishment)
  Timeout:       0 + 0   + 0 =  0.0

Value function tracks returns in [0, 3] instead of [-5, +6].
explained_variance should lift above 0.3 within 50k steps.

No step-0 signal bonus (removed in v1, kept removed).

`penalty_wrong=True` re-enables the symmetric -shaping_c penalty for ablation.

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
        alpha: float = 2.0,
        shaping_c: float = 1.0,
        junction_x_frac: float = 0.60,   # junction zone: x >= int(W * frac)
        penalty_wrong: bool = False,      # if True, wrong arm gets -shaping_c (ablation only)
        verbose: bool = False,
    ):
        super().__init__(env)
        self.alpha           = alpha
        self.shaping_c       = shaping_c
        self.junction_x_frac = junction_x_frac
        self.penalty_wrong   = penalty_wrong
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
        for x in range(min(3, W)):
            for y in range(H):
                cell = grid.get(x, y)
                if cell is not None and hasattr(cell, "type") and cell.type in ("key", "ball"):
                    self._signal_type = cell.type

        ok = self._correct_arm_y is not None
        if self.verbose:
            status = "OK" if ok else "WARNING: success_pos not found"
            print(
                f"[Oracle] reset {status} | "
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
        obs, info           = self.env.reset(**kwargs)
        self._step          = 0
        self._junction_done = False
        self.events         = []
        self._scan_grid()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        inner     = self._get_inner()
        agent_pos = (int(inner.agent_pos[0]), int(inner.agent_pos[1]))
        bonus     = 0.0

        # ── Event 1: Signal (t=0) — logged only, NO bonus ─────────────────────
        # Unconditional bonus here created a "safe-wander" local optimum where
        # timing out returned +1 shaped reward (safer than risking the junction).
        if self._step == 0:
            self.events.append({
                "step" : self._step,
                "event": "signal",
                "type" : self._signal_type,
                "bonus": 0.0,
            })
            if self.verbose:
                print(f"[Oracle] t={self._step:4d}  SIGNAL({self._signal_type})  F=0 (no bonus)")

        # ── Event 2: T-Junction Commitment ────────────────────────────────────
        # Asymmetric: correct arm → +shaping_c, wrong arm → 0 (no penalty).
        # Keeps value function range bounded: returns in [0, alpha+1] not [-5, +6].
        if (
            not self._junction_done
            and self._junction_x is not None
            and agent_pos[0] >= self._junction_x
            and agent_pos[1] != self._corridor_y
        ):
            correct = self._is_correct_direction(agent_pos[1])

            if correct:
                junc_bonus = self.shaping_c
            elif self.penalty_wrong:
                junc_bonus = -self.shaping_c   # ablation only
            else:
                junc_bonus = 0.0               # default: no punishment

            bonus += junc_bonus
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
                arm = "CORRECT" if correct else "WRONG  "
                print(
                    f"[Oracle] t={self._step:4d}  JUNCTION {arm}  "
                    f"agent_y={agent_pos[1]}  "
                    f"correct_arm_y={self._correct_arm_y}  "
                    f"F={junc_bonus:+.2f}"
                )

        self._step    += 1
        shaped_reward  = reward + self.alpha * bonus
        if bonus != 0.0:
            info["oracle_bonus"] = self.alpha * bonus
        return obs, shaped_reward, terminated, truncated, info

"""
Audit-Driven Risk Scaling — backtest-audit integration.

Connects live fill history to the backtest-audit overfitting detection library
and adjusts position sizing in real time based on overfitting risk.

Design
------
  Soft integration (not a hard kill):
    position_scale = 1.0 - overfitting_risk          # [0, 1]

  Hard kill only when risk > KILL_THRESHOLD AND we have enough data:
    if risk > 0.9 and n_fills >= MIN_FILLS_FOR_KILL:
        halt trading

  overfitting_risk is computed from:
    - DSR verdict:     PASS=0, WARN=0.3, FAIL=0.7
    - MC p-value:      p < 0.05 -> 0, p < 0.1 -> 0.2, else 0.5
    - OOS hit rate:    walk-forward (needs >= 120 fills to split)
    - Regime failure:  any regime FAIL -> +0.2

  All components are clipped to [0, 1] and blended.

Usage
-----
    from mm_live.risk.audit_risk import AuditRiskController

    controller = AuditRiskController()
    controller.record_fill(fill)
    scale = controller.position_scale()   # multiply your quote sizes by this
    if controller.should_halt():
        # stop quoting
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Optional import — gracefully degrade if backtest-audit not installed
try:
    from backtest_audit import BacktestAuditor
    from backtest_audit.walk_forward import walk_forward_validation
    _AUDIT_AVAILABLE = True
except ImportError:
    _AUDIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FILLS_TO_AUDIT = 30       # need at least 30 fills to run any audit
MIN_FILLS_FOR_KILL = 100      # kill switch only engages after 100 fills
KILL_THRESHOLD = 0.90         # overfitting_risk > 90% -> halt
AUDIT_EVERY_N = 20            # re-run audit every N new fills
WINDOW_FILLS = 500            # rolling window size for returns


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditRiskState:
    n_fills: int = 0
    overfitting_risk: float = 0.0       # [0, 1]  0=clean, 1=pure noise
    position_scale: float = 1.0         # [0, 1]  how much to scale down
    halt: bool = False
    last_dsr: float | None = None
    last_mc_pvalue: float | None = None
    last_oos_hit_rate: float | None = None
    regime_fail: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "n_fills": self.n_fills,
            "overfitting_risk": round(self.overfitting_risk, 4),
            "position_scale": round(self.position_scale, 4),
            "halt": self.halt,
            "last_dsr": self.last_dsr,
            "last_mc_pvalue": self.last_mc_pvalue,
            "last_oos_hit_rate": self.last_oos_hit_rate,
            "regime_fail": self.regime_fail,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AuditRiskController:
    """
    Live overfitting-risk monitor that scales position size.

    Parameters
    ----------
    n_trials : int
        Number of parameter combinations tried when building this strategy.
        Higher n_trials -> stricter DSR benchmark (more multiple-testing correction).
    min_fills_to_audit : int
        Minimum fill count before any audit is run.
    min_fills_for_kill : int
        Minimum fill count before the hard kill switch can engage.
    kill_threshold : float
        overfitting_risk level that triggers halt (if >= min_fills_for_kill).
    audit_every_n : int
        Re-run audit every N new fills.
    """

    def __init__(
        self,
        n_trials: int = 1,
        min_fills_to_audit: int = MIN_FILLS_TO_AUDIT,
        min_fills_for_kill: int = MIN_FILLS_FOR_KILL,
        kill_threshold: float = KILL_THRESHOLD,
        audit_every_n: int = AUDIT_EVERY_N,
    ) -> None:
        self._n_trials = n_trials
        self._min_audit = min_fills_to_audit
        self._min_kill = min_fills_for_kill
        self._kill_thr = kill_threshold
        self._audit_n = audit_every_n

        self._pnl_history: deque[float] = deque(maxlen=WINDOW_FILLS)
        self._fills_since_audit = 0
        self._state = AuditRiskState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_fill(self, pnl: float) -> None:
        """Record a fill's P&L. Triggers re-audit every AUDIT_EVERY_N fills."""
        self._pnl_history.append(pnl)
        self._state.n_fills += 1
        self._fills_since_audit += 1

        if (
            len(self._pnl_history) >= self._min_audit
            and self._fills_since_audit >= self._audit_n
        ):
            self._run_audit()
            self._fills_since_audit = 0

    def position_scale(self) -> float:
        """Return the current position scale factor [0, 1]."""
        return self._state.position_scale

    def should_halt(self) -> bool:
        """Return True if trading should be halted."""
        return self._state.halt

    def state(self) -> AuditRiskState:
        """Return a copy of the current risk state."""
        return self._state

    def reset_halt(self) -> None:
        """Manually clear the halt flag (after human review)."""
        self._state.halt = False
        self._state.notes = "Halt manually cleared."

    # ------------------------------------------------------------------
    # Audit engine
    # ------------------------------------------------------------------

    def _run_audit(self) -> None:
        if not _AUDIT_AVAILABLE:
            self._state.notes = "backtest-audit not installed — risk=0 (no audit)"
            return

        returns = pd.Series(list(self._pnl_history))
        n = len(returns)

        risk_components: list[float] = []
        notes: list[str] = []

        # ── DSR component ─────────────────────────────────────────────
        try:
            auditor = BacktestAuditor(returns, n_trials=self._n_trials)
            dsr_res = auditor.run_dsr()
            dsr_val = dsr_res.get("dsr", 0.0)
            dsr_verdict = dsr_res.get("verdict", "FAIL")
            self._state.last_dsr = round(float(dsr_val), 4)

            dsr_risk = {"PASS": 0.0, "WARN": 0.3, "FAIL": 0.7}.get(dsr_verdict, 0.7)
            risk_components.append(dsr_risk)
            notes.append(f"DSR={dsr_val:.3f}[{dsr_verdict}]")
        except Exception as e:
            risk_components.append(0.5)
            notes.append(f"DSR_err:{e!s:.20s}")

        # ── MC p-value component ──────────────────────────────────────
        try:
            mc_res = auditor.run_monte_carlo(n_permutations=500)
            mc_p = mc_res.get("pvalue", 1.0)
            self._state.last_mc_pvalue = round(float(mc_p), 4)

            if mc_p < 0.05:
                mc_risk = 0.0
            elif mc_p < 0.1:
                mc_risk = 0.2
            elif mc_p < 0.2:
                mc_risk = 0.4
            else:
                mc_risk = 0.6
            risk_components.append(mc_risk)
            notes.append(f"MC_p={mc_p:.3f}")
        except Exception as e:
            risk_components.append(0.3)
            notes.append(f"MC_err:{e!s:.20s}")

        # ── Walk-forward OOS component (needs enough data) ────────────
        if n >= 120:
            try:
                wf = walk_forward_validation(returns, n_splits=4)
                hit = wf.oos_hit_rate
                self._state.last_oos_hit_rate = round(float(hit), 4)

                # OOS hit rate: 0.75+ = clean, 0.5-0.75 = marginal, <0.5 = bad
                if hit >= 0.75:
                    oos_risk = 0.0
                elif hit >= 0.5:
                    oos_risk = 0.3
                else:
                    oos_risk = 0.7
                risk_components.append(oos_risk)
                notes.append(f"OOS_hit={hit:.0%}")
            except Exception as e:
                notes.append(f"WF_err:{e!s:.20s}")

        # ── Regime component ──────────────────────────────────────────
        if n >= 80:
            try:
                regime = auditor.run_regime_audit(n_permutations=100)
                regime_fail = regime.overall_verdict in ("BROKEN", "FAIL")
                self._state.regime_fail = regime_fail

                regime_risk = 0.3 if regime_fail else 0.0
                risk_components.append(regime_risk)
                notes.append(f"Regime={regime.overall_verdict}")
            except Exception as e:
                notes.append(f"Regime_err:{e!s:.20s}")

        # ── Blend all components ──────────────────────────────────────
        if risk_components:
            # Weighted average: DSR + MC carry more weight
            weights = [2.0] + [1.5] + [1.0] * (len(risk_components) - 2)
            weights = weights[:len(risk_components)]
            total_w = sum(weights)
            overfitting_risk = sum(r * w for r, w in zip(risk_components, weights)) / total_w
        else:
            overfitting_risk = 0.0

        overfitting_risk = float(np.clip(overfitting_risk, 0.0, 1.0))

        # ── Position scale ────────────────────────────────────────────
        # Soft: scale linearly from 1.0 (no risk) to 0.0 (full risk)
        position_scale = 1.0 - overfitting_risk

        # ── Hard kill switch ──────────────────────────────────────────
        halt = (
            overfitting_risk >= self._kill_thr
            and n >= self._min_kill
        )

        self._state.overfitting_risk = round(overfitting_risk, 4)
        self._state.position_scale   = round(position_scale, 4)
        self._state.halt             = halt
        self._state.notes            = " | ".join(notes)

    # ------------------------------------------------------------------
    # Debug / reporting
    # ------------------------------------------------------------------

    def print_state(self) -> None:
        s = self._state
        width = 60
        print(f"\n{'=' * width}")
        print("  AUDIT RISK CONTROLLER".center(width))
        print(f"{'=' * width}")
        print(f"  Fills tracked   : {s.n_fills}")
        print(f"  Overfitting risk: {s.overfitting_risk:.1%}")
        print(f"  Position scale  : {s.position_scale:.1%}  "
              f"({'FULL SIZE' if s.position_scale > 0.9 else 'REDUCED' if s.position_scale > 0.1 else 'MINIMAL'})")
        print(f"  HALT            : {'YES -- TRADING STOPPED' if s.halt else 'no'}")
        if s.last_dsr is not None:
            print(f"  Last DSR        : {s.last_dsr:.4f}")
        if s.last_mc_pvalue is not None:
            print(f"  Last MC p-value : {s.last_mc_pvalue:.4f}")
        if s.last_oos_hit_rate is not None:
            print(f"  OOS hit rate    : {s.last_oos_hit_rate:.0%}")
        if s.notes:
            print(f"  Notes           : {s.notes}")
        print(f"{'=' * width}\n")

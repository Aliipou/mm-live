"""
Audit-Driven Risk Scaling — backtest-audit integration.

Problem (honest):
  IS/OOS correlation is near zero for most strategies.
  This means a single bad audit reading causes too many false positives —
  we'd scale down or kill a real edge because of noise in the audit metrics.

Solution:
  1. CONSENSUS: scale down only when MULTIPLE independent metrics fail simultaneously.
     (DSR + MC + regime must all fail; one alone is not enough)
  2. HYSTERESIS: risk must stay elevated for N consecutive audits before scaling.
     (prevents flip-flopping on noisy short windows)
  3. SOFT first, HARD later: position_scale = 1 - overfitting_risk (continuous).
     Hard kill only when risk > 0.9 AND n_fills >= 100 AND hysteresis triggered.

Risk score formula:
  - Only scales down if >= CONSENSUS_THRESHOLD signals flag risk simultaneously
  - overfitting_risk = weighted blend of flagging signals (not all signals)
  - hysteresis: apply scaling only if risk has been elevated for >= HYSTERESIS_N audits

Usage
-----
    from mm_live.risk.audit_risk import AuditRiskController

    controller = AuditRiskController(n_trials=50)
    controller.record_fill(pnl)             # call on every fill
    scale = controller.position_scale()     # multiply bid/ask qty by this
    if controller.should_halt():
        break                               # stop quoting loop
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from backtest_audit import BacktestAuditor
    from backtest_audit.walk_forward import walk_forward_validation
    _AUDIT_AVAILABLE = True
except ImportError:
    _AUDIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FILLS_TO_AUDIT  = 30
MIN_FILLS_FOR_KILL  = 100
KILL_THRESHOLD      = 0.90
AUDIT_EVERY_N       = 20
WINDOW_FILLS        = 500

# Consensus: how many of the 4 risk signals must flag before we scale down?
# Setting to 2 means: at least 2 independent signals must fail simultaneously.
# This dramatically reduces false positives from the near-zero IS/OOS correlation.
CONSENSUS_THRESHOLD = 2

# Hysteresis: risk must be elevated for this many consecutive audits
# before we apply the scale-down. Prevents flip-flopping on noisy short windows.
HYSTERESIS_N = 2


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditRiskState:
    n_fills: int = 0
    overfitting_risk: float = 0.0
    position_scale: float = 1.0
    halt: bool = False
    n_signals_flagging: int = 0         # how many signals flagged this audit
    consecutive_elevated: int = 0       # how many consecutive audits at risk
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
            "n_signals_flagging": self.n_signals_flagging,
            "consecutive_elevated": self.consecutive_elevated,
            "last_dsr": self.last_dsr,
            "last_mc_pvalue": self.last_mc_pvalue,
            "last_oos_hit_rate": self.last_oos_hit_rate,
            "regime_fail": self.regime_fail,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Signal result
# ---------------------------------------------------------------------------

@dataclass
class _SignalResult:
    name: str
    risk: float         # [0, 1]
    flagging: bool      # True if this signal considers edge suspicious
    note: str


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AuditRiskController:
    """
    Live overfitting-risk monitor with consensus + hysteresis protection.

    Parameters
    ----------
    n_trials : int
        Strategy parameter combinations tried. Higher -> stricter DSR.
    consensus_threshold : int
        Minimum number of independent risk signals that must flag simultaneously
        before position size is reduced. Default 2 (reduces false positives).
    hysteresis_n : int
        Number of consecutive elevated-risk audits required before scaling.
        Default 2 (prevents flip-flopping on noise).
    min_fills_to_audit : int
        Minimum fills before any audit runs.
    min_fills_for_kill : int
        Minimum fills before hard kill can engage.
    kill_threshold : float
        overfitting_risk level for hard halt.
    audit_every_n : int
        Re-audit every N new fills.
    """

    def __init__(
        self,
        n_trials: int = 1,
        consensus_threshold: int = CONSENSUS_THRESHOLD,
        hysteresis_n: int = HYSTERESIS_N,
        min_fills_to_audit: int = MIN_FILLS_TO_AUDIT,
        min_fills_for_kill: int = MIN_FILLS_FOR_KILL,
        kill_threshold: float = KILL_THRESHOLD,
        audit_every_n: int = AUDIT_EVERY_N,
    ) -> None:
        self._n_trials      = n_trials
        self._consensus     = consensus_threshold
        self._hysteresis    = hysteresis_n
        self._min_audit     = min_fills_to_audit
        self._min_kill      = min_fills_for_kill
        self._kill_thr      = kill_threshold
        self._audit_n       = audit_every_n

        self._pnl_history: deque[float] = deque(maxlen=WINDOW_FILLS)
        self._fills_since_audit = 0
        self._state = AuditRiskState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_fill(self, pnl: float) -> None:
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
        return self._state.position_scale

    def should_halt(self) -> bool:
        return self._state.halt

    def state(self) -> AuditRiskState:
        return self._state

    def reset_halt(self) -> None:
        self._state.halt = False
        self._state.consecutive_elevated = 0
        self._state.notes = "Halt manually cleared."

    # ------------------------------------------------------------------
    # Audit engine
    # ------------------------------------------------------------------

    def _run_audit(self) -> None:
        if not _AUDIT_AVAILABLE:
            self._state.notes = "backtest-audit not installed"
            return

        returns = pd.Series(list(self._pnl_history))
        n = len(returns)
        signals: list[_SignalResult] = []

        try:
            auditor = BacktestAuditor(returns, n_trials=self._n_trials)
        except Exception as e:
            self._state.notes = f"audit_init_err: {e!s:.30s}"
            return

        # ── Signal 1: DSR ────────────────────────────────────────────
        try:
            dsr_res = auditor.run_dsr()
            dsr_val = float(dsr_res.get("dsr", 0.0))
            dsr_verdict = dsr_res.get("verdict", "FAIL")
            self._state.last_dsr = round(dsr_val, 4)
            risk = {"PASS": 0.0, "WARN": 0.25, "FAIL": 0.7}.get(dsr_verdict, 0.7)
            signals.append(_SignalResult("DSR", risk, risk >= 0.5, f"DSR={dsr_val:.3f}[{dsr_verdict}]"))
        except Exception as e:
            signals.append(_SignalResult("DSR", 0.3, False, f"DSR_err:{e!s:.15s}"))

        # ── Signal 2: Monte Carlo p-value ────────────────────────────
        try:
            mc_res = auditor.run_monte_carlo(n_permutations=300)
            mc_p = float(mc_res.get("pvalue", 1.0))
            self._state.last_mc_pvalue = round(mc_p, 4)
            risk = 0.0 if mc_p < 0.05 else (0.2 if mc_p < 0.15 else 0.6)
            signals.append(_SignalResult("MC", risk, risk >= 0.5, f"MC_p={mc_p:.3f}"))
        except Exception as e:
            signals.append(_SignalResult("MC", 0.2, False, f"MC_err:{e!s:.15s}"))

        # ── Signal 3: Walk-forward OOS (needs >= 120 fills) ──────────
        if n >= 120:
            try:
                wf = walk_forward_validation(returns, n_splits=3)
                hit = float(wf.oos_hit_rate)
                self._state.last_oos_hit_rate = round(hit, 4)
                risk = 0.0 if hit >= 0.67 else (0.3 if hit >= 0.5 else 0.7)
                signals.append(_SignalResult("OOS", risk, risk >= 0.5, f"OOS_hit={hit:.0%}"))
            except Exception as e:
                pass  # skip OOS signal on error — don't penalise

        # ── Signal 4: Regime consistency (needs >= 80 fills) ─────────
        if n >= 80:
            try:
                regime = auditor.run_regime_audit(n_permutations=80)
                regime_fail = regime.overall_verdict in ("BROKEN", "FAIL")
                self._state.regime_fail = regime_fail
                risk = 0.5 if regime_fail else 0.0
                signals.append(_SignalResult("Regime", risk, regime_fail, f"Regime={regime.overall_verdict}"))
            except Exception as e:
                pass  # skip regime signal on error

        # ── Consensus check ───────────────────────────────────────────
        n_flagging = sum(1 for s in signals if s.flagging)
        consensus_met = n_flagging >= self._consensus

        if consensus_met:
            # Blend only the flagging signals for the risk score
            flagging = [s for s in signals if s.flagging]
            raw_risk = sum(s.risk for s in flagging) / len(flagging)
        else:
            # Not enough independent signals agree — risk stays low
            raw_risk = 0.1 * (n_flagging / max(self._consensus, 1))

        overfitting_risk = float(np.clip(raw_risk, 0.0, 1.0))

        # ── Hysteresis ────────────────────────────────────────────────
        ELEVATED_THRESHOLD = 0.3
        if overfitting_risk >= ELEVATED_THRESHOLD:
            self._state.consecutive_elevated += 1
        else:
            self._state.consecutive_elevated = 0

        hysteresis_met = self._state.consecutive_elevated >= self._hysteresis

        # Apply scaling only if hysteresis is satisfied
        if hysteresis_met:
            position_scale = float(np.clip(1.0 - overfitting_risk, 0.0, 1.0))
        else:
            position_scale = 1.0

        # ── Hard kill: requires EVERYTHING to be confirmed ────────────
        halt = (
            overfitting_risk >= self._kill_thr
            and n >= self._min_kill
            and hysteresis_met
            and consensus_met
        )

        self._state.overfitting_risk     = round(overfitting_risk, 4)
        self._state.position_scale       = round(position_scale, 4)
        self._state.halt                 = halt
        self._state.n_signals_flagging   = n_flagging
        self._state.notes                = " | ".join(s.note for s in signals)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def print_state(self) -> None:
        s = self._state
        width = 62
        print(f"\n{'=' * width}")
        print("  AUDIT RISK CONTROLLER".center(width))
        print(f"{'=' * width}")
        print(f"  Fills tracked      : {s.n_fills}")
        print(f"  Signals flagging   : {s.n_signals_flagging}/{self._consensus} needed")
        print(f"  Consecutive high   : {s.consecutive_elevated}/{self._hysteresis} needed")
        print(f"  Overfitting risk   : {s.overfitting_risk:.1%}")
        scale_label = "FULL SIZE" if s.position_scale > 0.9 else ("REDUCED" if s.position_scale > 0.2 else "MINIMAL")
        print(f"  Position scale     : {s.position_scale:.1%}  [{scale_label}]")
        print(f"  HALT               : {'YES -- STOPPED' if s.halt else 'no'}")
        if s.last_dsr is not None:
            print(f"  Last DSR           : {s.last_dsr:.4f}")
        if s.last_mc_pvalue is not None:
            print(f"  Last MC p-value    : {s.last_mc_pvalue:.4f}")
        if s.last_oos_hit_rate is not None:
            print(f"  OOS hit rate       : {s.last_oos_hit_rate:.0%}")
        if s.notes:
            print(f"  Details            : {s.notes}")
        print(f"{'=' * width}\n")

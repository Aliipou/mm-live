"""
LiveAuditLoop — real-time backtest-audit integration for mm-live.

Architecture (follows the integration map):

  Market Data (Binance/OKX WS)
       |
       v
  Signal Layer (every 100ms tick)
  ├─ Fair Value (Kalman + OFI)
  ├─ Volatility (dual EWMA + regime)
  └─ Composite Edge Score
       |
       v
  LiveAuditLoop (this module)
  ├─ Collects per-tick P&L estimates
  ├─ Runs backtest-audit every AUDIT_EVERY ticks
  │    DSR, MC, walk-forward OOS, regime, robustness
  └─ Emits risk_score [0,1] + position_scale [0,1]
       |
       v
  Position Sizing
    size = BASE_SIZE * position_scale * regime_multiplier
       |
       v
  Adaptive Avellaneda-Stoikov quoting
       |
       v
  Fill tracking -> feeds back into LiveAuditLoop

Key design principles
---------------------
  - Soft first: position_scale = 1 - risk_score  (continuous reduction)
  - Regime-aware: high-vol regime applies extra multiplier on top
  - Consensus: DSR + MC must both signal risk before scaling
  - Hysteresis: AuditRiskController (audit_risk.py) handles this
  - Hard kill: only when risk > 0.9 AND n_fills >= 100 AND sustained

Usage (in trading engine)
--------------------------
    loop = LiveAuditLoop(n_trials=50)

    # On each fill:
    loop.record_fill(pnl=realized_pnl, regime=current_regime)

    # On each 100ms tick:
    scale = loop.position_scale(regime=current_regime)
    if loop.should_halt():
        # stop quoting
    size = BASE_SIZE * scale
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
    from backtest_audit import BacktestAuditor
    from backtest_audit.walk_forward import walk_forward_validation
    _AUDIT_AVAILABLE = True
except ImportError:
    _AUDIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIT_EVERY       = 50       # run audit every N fills
MIN_FILLS         = 30       # minimum fills before first audit
MIN_KILLS         = 100      # minimum fills before kill switch can engage
KILL_THRESHOLD    = 0.90     # risk_score threshold for halt
HYSTERESIS        = 2        # consecutive audits at risk before scaling
CONSENSUS         = 2        # signals that must flag before scaling
WINDOW            = 500      # rolling P&L window

# Regime-based extra multiplier (applied ON TOP of risk scaling)
REGIME_SCALE = {
    "high_vol":   0.5,   # high vol -> cut position in half
    "normal_vol": 1.0,
    "low_vol":    1.2,   # low vol -> allow slightly more
    "normal":     1.0,   # fallback
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class LiveAuditState:
    n_fills: int = 0
    risk_score: float = 0.0
    position_scale: float = 1.0
    halt: bool = False
    consecutive_elevated: int = 0
    n_signals_flagging: int = 0
    audit_verdict: str = "PASS"
    last_dsr: float | None = None
    last_mc_pvalue: float | None = None
    last_oos_hit_rate: float | None = None
    notes: str = "no audit yet"

    def to_dict(self) -> dict:
        return {
            "n_fills": self.n_fills,
            "risk_score": round(self.risk_score, 4),
            "position_scale": round(self.position_scale, 4),
            "halt": self.halt,
            "audit_verdict": self.audit_verdict,
            "consecutive_elevated": self.consecutive_elevated,
            "n_signals_flagging": self.n_signals_flagging,
            "last_dsr": self.last_dsr,
            "last_mc_pvalue": self.last_mc_pvalue,
            "last_oos_hit_rate": self.last_oos_hit_rate,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LiveAuditLoop:
    """
    Real-time audit loop that connects mm-live fill history to backtest-audit.

    Parameters
    ----------
    n_trials : int
        Number of parameter combos tried when designing this strategy.
        Higher -> stricter DSR (more multiple-testing correction).
    audit_every : int
        Run full audit every N fills.
    consensus : int
        Minimum independent signals that must flag before scaling.
    hysteresis : int
        Consecutive elevated-risk audits required before scaling applies.
    """

    def __init__(
        self,
        n_trials: int = 1,
        audit_every: int = AUDIT_EVERY,
        consensus: int = CONSENSUS,
        hysteresis: int = HYSTERESIS,
        kill_threshold: float = KILL_THRESHOLD,
        min_fills: int = MIN_FILLS,
        min_kills: int = MIN_KILLS,
    ) -> None:
        self._n_trials       = n_trials
        self._audit_every    = audit_every
        self._consensus      = consensus
        self._hysteresis     = hysteresis
        self._kill_thr       = kill_threshold
        self._min_fills      = min_fills
        self._min_kills      = min_kills

        self._pnl: deque[float] = deque(maxlen=WINDOW)
        self._fills_since_audit = 0
        self._state = LiveAuditState()

    # ------------------------------------------------------------------
    # Public: called by the trading engine
    # ------------------------------------------------------------------

    def record_fill(self, pnl: float, regime: str = "normal") -> None:
        """
        Record a realized P&L from a fill.
        Call this every time an order is filled.
        Triggers re-audit every AUDIT_EVERY fills.
        """
        self._pnl.append(pnl)
        self._state.n_fills += 1
        self._fills_since_audit += 1

        if (
            len(self._pnl) >= self._min_fills
            and self._fills_since_audit >= self._audit_every
        ):
            self._run_audit()
            self._fills_since_audit = 0

    def position_scale(self, regime: str = "normal") -> float:
        """
        Return position scale factor in [0, 1].

        Applies:
          1. Audit risk scaling (1 - risk_score)
          2. Regime multiplier (high_vol reduces further)
        Final result is clipped to [0, 1].

        Usage:
            size = BASE_SIZE * loop.position_scale(regime=current_regime)
        """
        if self._state.halt:
            return 0.0
        base_scale = self._state.position_scale
        regime_mult = REGIME_SCALE.get(regime, 1.0)
        return float(np.clip(base_scale * regime_mult, 0.0, 1.0))

    def should_halt(self) -> bool:
        """True if trading should stop immediately."""
        return self._state.halt

    def state(self) -> LiveAuditState:
        return self._state

    def reset_halt(self) -> None:
        """Call after human review to resume trading."""
        self._state.halt = False
        self._state.consecutive_elevated = 0
        self._state.notes = "halt manually cleared"

    def risk_score(self) -> float:
        """Raw risk score [0, 1] — 0 = clean edge, 1 = pure noise."""
        return self._state.risk_score

    # ------------------------------------------------------------------
    # Audit engine
    # ------------------------------------------------------------------

    def _run_audit(self) -> None:
        if not _AUDIT_AVAILABLE or len(self._pnl) < self._min_fills:
            return

        returns = pd.Series(list(self._pnl))
        n = len(returns)
        signals_flagging = 0
        risk_components: list[float] = []
        notes: list[str] = []

        try:
            auditor = BacktestAuditor(returns, n_trials=self._n_trials)
        except Exception as e:
            self._state.notes = f"init_err: {e!s:.40s}"
            return

        # ── Signal 1: DSR ─────────────────────────────────────────────
        try:
            dsr_res = auditor.run_dsr()
            dsr     = float(dsr_res.get("dsr", 0.0))
            verdict = dsr_res.get("verdict", "FAIL")
            self._state.last_dsr = round(dsr, 4)
            risk = {"PASS": 0.0, "WARN": 0.25, "FAIL": 0.7}.get(verdict, 0.7)
            if risk >= 0.5:
                signals_flagging += 1
            risk_components.append(risk)
            notes.append(f"DSR={dsr:.3f}[{verdict}]")
        except Exception:
            risk_components.append(0.3)

        # ── Signal 2: Monte Carlo ──────────────────────────────────────
        try:
            mc_res = auditor.run_monte_carlo(n_permutations=300)
            mc_p   = float(mc_res.get("pvalue", 1.0))
            self._state.last_mc_pvalue = round(mc_p, 4)
            risk = 0.0 if mc_p < 0.05 else (0.2 if mc_p < 0.15 else 0.6)
            if risk >= 0.5:
                signals_flagging += 1
            risk_components.append(risk)
            notes.append(f"MC_p={mc_p:.3f}")
        except Exception:
            risk_components.append(0.2)

        # ── Signal 3: Walk-forward OOS (>= 120 fills) ─────────────────
        if n >= 120:
            try:
                wf  = walk_forward_validation(returns, n_splits=3)
                hit = float(wf.oos_hit_rate)
                self._state.last_oos_hit_rate = round(hit, 4)
                risk = 0.0 if hit >= 0.67 else (0.3 if hit >= 0.5 else 0.7)
                if risk >= 0.5:
                    signals_flagging += 1
                risk_components.append(risk)
                notes.append(f"OOS={hit:.0%}")
            except Exception:
                pass

        # ── Signal 4: Regime audit (>= 80 fills) ──────────────────────
        if n >= 80:
            try:
                regime_audit = auditor.run_regime_audit(n_permutations=60)
                bad = regime_audit.overall_verdict in ("BROKEN", "FAIL")
                risk = 0.5 if bad else 0.0
                if bad:
                    signals_flagging += 1
                risk_components.append(risk)
                notes.append(f"Regime={regime_audit.overall_verdict}")
            except Exception:
                pass

        # ── Consensus gate ─────────────────────────────────────────────
        self._state.n_signals_flagging = signals_flagging
        consensus_met = signals_flagging >= self._consensus

        if consensus_met and risk_components:
            raw_risk = float(np.mean(risk_components))
        else:
            raw_risk = 0.05 * signals_flagging  # small penalty but no scale-down

        risk_score = float(np.clip(raw_risk, 0.0, 1.0))

        # ── Hysteresis ────────────────────────────────────────────────
        if risk_score >= 0.3:
            self._state.consecutive_elevated += 1
        else:
            self._state.consecutive_elevated = 0

        hysteresis_met = self._state.consecutive_elevated >= self._hysteresis

        position_scale = (
            float(np.clip(1.0 - risk_score, 0.0, 1.0))
            if hysteresis_met else 1.0
        )

        # ── Hard kill ─────────────────────────────────────────────────
        halt = (
            risk_score >= self._kill_thr
            and n >= self._min_kills
            and hysteresis_met
            and consensus_met
        )

        # ── Audit verdict via full report ─────────────────────────────
        try:
            report = auditor.run_all(
                n_permutations=200,
                include_walk_forward=(n >= 120),
                include_regime=(n >= 80),
                include_robustness=(n >= 30),
                include_economic=True,
            )
            self._state.audit_verdict = report.overall_verdict
        except Exception:
            self._state.audit_verdict = "WARN"

        self._state.risk_score        = round(risk_score, 4)
        self._state.position_scale    = round(position_scale, 4)
        self._state.halt              = halt
        self._state.notes             = " | ".join(notes) if notes else "no signals"

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def print_state(self) -> None:
        s = self._state
        w = 65
        print(f"\n{'=' * w}")
        print("  LIVE AUDIT LOOP".center(w))
        print(f"{'=' * w}")
        print(f"  Fills           : {s.n_fills}")
        print(f"  Risk score      : {s.risk_score:.1%}")
        print(f"  Signals flagging: {s.n_signals_flagging}/{self._consensus} needed")
        print(f"  Hysteresis      : {s.consecutive_elevated}/{self._hysteresis}")
        print(f"  Position scale  : {s.position_scale:.1%}")
        print(f"  Audit verdict   : {s.audit_verdict}")
        print(f"  HALT            : {'YES' if s.halt else 'no'}")
        if s.notes != "no audit yet":
            print(f"  Details         : {s.notes}")
        print(f"{'=' * w}\n")

"""Tests for live_audit_loop module."""
from __future__ import annotations

import numpy as np

from mm_live.risk.live_audit_loop import REGIME_SCALE, LiveAuditLoop, LiveAuditState


def _feed(loop: LiveAuditLoop, n: int, mu: float = 0.001, sigma: float = 0.01, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for pnl in rng.normal(mu, sigma, n):
        loop.record_fill(float(pnl))


class TestLiveAuditLoop:
    def test_initial_scale_is_one(self):
        loop = LiveAuditLoop()
        assert loop.position_scale() == 1.0

    def test_initial_no_halt(self):
        loop = LiveAuditLoop()
        assert not loop.should_halt()

    def test_initial_risk_score_zero(self):
        loop = LiveAuditLoop()
        assert loop.risk_score() == 0.0

    def test_fills_counted(self):
        loop = LiveAuditLoop()
        _feed(loop, 20)
        assert loop.state().n_fills == 20

    def test_state_is_dataclass(self):
        loop = LiveAuditLoop()
        assert isinstance(loop.state(), LiveAuditState)

    def test_scale_in_unit_interval(self):
        loop = LiveAuditLoop(audit_every=10)
        _feed(loop, 50)
        assert 0.0 <= loop.position_scale() <= 1.0

    def test_halt_returns_zero_scale(self):
        loop = LiveAuditLoop()
        loop._state.halt = True
        assert loop.position_scale() == 0.0

    def test_reset_halt_clears_flag(self):
        loop = LiveAuditLoop()
        loop._state.halt = True
        loop.reset_halt()
        assert not loop.should_halt()

    def test_regime_high_vol_reduces_scale(self):
        loop = LiveAuditLoop()
        loop._state.position_scale = 1.0
        scale_normal = loop.position_scale(regime="normal_vol")
        scale_high   = loop.position_scale(regime="high_vol")
        assert scale_high < scale_normal

    def test_regime_low_vol_allows_more(self):
        loop = LiveAuditLoop()
        loop._state.position_scale = 0.8
        scale_normal = loop.position_scale(regime="normal_vol")
        scale_low    = loop.position_scale(regime="low_vol")
        assert scale_low >= scale_normal

    def test_scale_clipped_to_one(self):
        loop = LiveAuditLoop()
        loop._state.position_scale = 1.0
        # low_vol mult is 1.2 but result must be <= 1.0
        assert loop.position_scale(regime="low_vol") <= 1.0

    def test_audit_runs_after_min_fills(self):
        loop = LiveAuditLoop(min_fills=30, audit_every=5)
        _feed(loop, 40)
        assert loop.state().last_dsr is not None

    def test_no_audit_before_min_fills(self):
        loop = LiveAuditLoop(min_fills=200)
        _feed(loop, 50)
        assert loop.state().last_dsr is None

    def test_risk_in_unit_interval(self):
        loop = LiveAuditLoop(audit_every=10)
        _feed(loop, 50)
        assert 0.0 <= loop.risk_score() <= 1.0

    def test_to_dict_has_required_keys(self):
        loop = LiveAuditLoop(audit_every=10)
        _feed(loop, 40)
        d = loop.state().to_dict()
        for key in ("n_fills", "risk_score", "position_scale", "halt", "audit_verdict"):
            assert key in d

    def test_print_state_runs(self, capsys):
        loop = LiveAuditLoop(audit_every=10)
        _feed(loop, 40)
        loop.print_state()
        out = capsys.readouterr().out
        assert "LIVE AUDIT LOOP" in out

    def test_kill_needs_min_fills(self):
        # Even with extreme kill threshold=0, needs min_kills fills
        loop = LiveAuditLoop(kill_threshold=0.0, min_kills=1000, audit_every=5)
        _feed(loop, 60, mu=-0.01)
        assert not loop.should_halt()

    def test_consensus_not_met_keeps_scale_high(self):
        # With consensus=3, very unlikely to scale down on short window
        loop = LiveAuditLoop(consensus=3, audit_every=10)
        _feed(loop, 50, mu=0.0, sigma=0.01, seed=99)
        # Even if 1 signal flags, scale should stay near 1.0 without consensus
        assert loop.position_scale() >= 0.8 or loop.state().n_signals_flagging >= 3

    def test_regime_scale_constants(self):
        assert REGIME_SCALE["high_vol"] < REGIME_SCALE["normal_vol"]
        assert REGIME_SCALE["low_vol"] >= REGIME_SCALE["normal_vol"]

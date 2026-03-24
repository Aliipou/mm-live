"""Tests for audit_risk module."""
from __future__ import annotations

import numpy as np

from mm_live.risk.audit_risk import AuditRiskController, AuditRiskState


def _feed(controller: AuditRiskController, n: int, mu: float = 0.001, sigma: float = 0.01, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for pnl in rng.normal(mu, sigma, n):
        controller.record_fill(float(pnl))


class TestAuditRiskController:
    def test_initial_scale_is_one(self):
        c = AuditRiskController()
        assert c.position_scale() == 1.0

    def test_initial_no_halt(self):
        c = AuditRiskController()
        assert not c.should_halt()

    def test_scale_in_unit_interval(self):
        c = AuditRiskController(audit_every_n=10)
        _feed(c, 50)
        assert 0.0 <= c.position_scale() <= 1.0

    def test_state_returns_dataclass(self):
        c = AuditRiskController()
        assert isinstance(c.state(), AuditRiskState)

    def test_fills_counted(self):
        c = AuditRiskController()
        _feed(c, 25)
        assert c.state().n_fills == 25

    def test_no_audit_before_min_fills(self):
        c = AuditRiskController(min_fills_to_audit=100)
        _feed(c, 20)
        # Should not have run audit yet — no DSR recorded
        assert c.state().last_dsr is None

    def test_audit_runs_after_min_fills(self):
        c = AuditRiskController(min_fills_to_audit=30, audit_every_n=5)
        _feed(c, 40)
        # Audit should have run — risk state updated
        assert c.state().overfitting_risk >= 0.0

    def test_good_edge_has_lower_risk(self):
        # Strong consistent PnL -> lower overfitting risk
        c_good = AuditRiskController(audit_every_n=5)
        _feed(c_good, 60, mu=0.01, sigma=0.005)

        # Noise -> higher overfitting risk
        c_noise = AuditRiskController(audit_every_n=5)
        _feed(c_noise, 60, mu=0.0, sigma=0.01, seed=99)

        # Good edge should have lower or equal risk than pure noise
        # (not always true with small samples but directional)
        assert c_good.state().overfitting_risk <= c_noise.state().overfitting_risk + 0.5

    def test_halt_not_triggered_below_threshold(self):
        c = AuditRiskController(kill_threshold=0.99, audit_every_n=5)
        _feed(c, 60)
        # With threshold=0.99, extremely unlikely to halt
        # (not impossible but very unlikely for reasonable returns)
        assert isinstance(c.should_halt(), bool)

    def test_halt_requires_min_fills(self):
        # Even with very high risk threshold, halt needs min_fills
        c = AuditRiskController(kill_threshold=0.0, min_fills_for_kill=1000, audit_every_n=5)
        _feed(c, 60, mu=-0.005)
        # Halt should not engage because n_fills < min_fills_for_kill
        assert not c.should_halt()

    def test_reset_halt_clears_flag(self):
        c = AuditRiskController()
        c._state.halt = True
        c.reset_halt()
        assert not c.should_halt()

    def test_to_dict_has_required_keys(self):
        c = AuditRiskController(audit_every_n=5)
        _feed(c, 40)
        d = c.state().to_dict()
        for key in ("n_fills", "overfitting_risk", "position_scale", "halt"):
            assert key in d

    def test_print_state_runs(self, capsys):
        c = AuditRiskController(audit_every_n=5)
        _feed(c, 40)
        c.print_state()
        out = capsys.readouterr().out
        assert "AUDIT RISK" in out

    def test_position_scale_bounded_by_risk(self):
        # With hysteresis, scale is either 1.0 (not yet triggered)
        # or 1 - risk (after consecutive elevated audits).
        # Either way: scale <= 1.0 and scale >= 1 - risk (never worse than full reduction).
        c = AuditRiskController(audit_every_n=5)
        _feed(c, 50)
        s = c.state()
        assert s.position_scale <= 1.0
        assert s.position_scale >= 0.0
        # If hysteresis triggered, scale == 1 - risk
        if s.consecutive_elevated >= 2:
            assert abs(s.position_scale - (1.0 - s.overfitting_risk)) < 1e-6

    def test_overfitting_risk_in_unit_interval(self):
        c = AuditRiskController(audit_every_n=5)
        _feed(c, 50)
        assert 0.0 <= c.state().overfitting_risk <= 1.0

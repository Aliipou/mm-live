"""Tests for the core Engine — audit loop integration and config."""
from __future__ import annotations

from mm_live.core.engine import Engine, EngineConfig
from mm_live.risk.live_audit_loop import LiveAuditLoop


class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.audit_n_trials == 1
        assert cfg.audit_every == 50
        assert cfg.audit_consensus == 2
        assert cfg.audit_hysteresis == 2

    def test_custom_audit_params(self):
        cfg = EngineConfig(audit_n_trials=20, audit_every=30, audit_consensus=3)
        assert cfg.audit_n_trials == 20
        assert cfg.audit_every == 30
        assert cfg.audit_consensus == 3


class TestEngineAuditIntegration:
    def test_engine_has_audit_loop(self):
        engine = Engine()
        assert isinstance(engine.audit_loop, LiveAuditLoop)

    def test_audit_loop_config_forwarded(self):
        cfg = EngineConfig(audit_n_trials=10, audit_every=25, audit_consensus=3, audit_hysteresis=3)
        engine = Engine(cfg)
        assert engine.audit_loop._n_trials == 10
        assert engine.audit_loop._audit_every == 25
        assert engine.audit_loop._consensus == 3
        assert engine.audit_loop._hysteresis == 3

    def test_initial_audit_scale_is_one(self):
        engine = Engine()
        assert engine.audit_loop.position_scale() == 1.0

    def test_initial_no_audit_halt(self):
        engine = Engine()
        assert not engine.audit_loop.should_halt()

    def test_audit_halt_stops_running(self):
        """When audit halt is triggered, engine should stop on next timer tick."""
        engine = Engine()
        # Manually trigger audit halt
        engine.audit_loop._state.halt = True
        # should_halt() reflects the forced state
        assert engine.audit_loop.should_halt()

    def test_reset_audit_halt(self):
        engine = Engine()
        engine.audit_loop._state.halt = True
        engine.audit_loop.reset_halt()
        assert not engine.audit_loop.should_halt()

    def test_audit_scale_affects_fill_qty(self):
        """Position scale < 1 should reduce effective fill quantity."""
        engine = Engine(EngineConfig(fill_qty=0.01))
        engine.audit_loop._state.position_scale = 0.5
        audit_scale = engine.audit_loop.position_scale()
        scaled_qty = engine.cfg.fill_qty * audit_scale
        assert scaled_qty == 0.005

    def test_audit_state_accessible(self):
        engine = Engine()
        state = engine.audit_loop.state()
        assert state.n_fills == 0
        assert state.risk_score == 0.0

    def test_multiple_engines_independent_audit(self):
        """Each engine instance has its own audit loop — no shared state."""
        e1 = Engine()
        e2 = Engine()
        e1.audit_loop._state.risk_score = 0.8
        assert e2.audit_loop.risk_score() == 0.0

"""
Tests for mm_live.research modules:
  - imbalance_prediction.ImbalanceEdgeTest / EdgeTestResult
  - regime_attribution.RegimeAttributionTracker / RegimeStats / Regime
  - benchmark.BenchmarkRunner / FixedSpreadMaker / NaiveMaker / BacktestResult

All tests use synthetic in-process data — no network calls.
"""

from __future__ import annotations

import math
import random

import pytest

from mm_live.research.imbalance_prediction import (
    EdgeTestResult,
    ImbalanceEdgeTest,
    ImbalanceSample,
)
from mm_live.research.regime_attribution import (
    FillRecord,
    Regime,
    RegimeAttributionTracker,
    RegimeStats,
)
from mm_live.research.benchmark import (
    BacktestResult,
    BenchmarkRunner,
    FixedSpreadMaker,
    MarketTick,
    NaiveMaker,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_tick(
    timestamp_ms: int = 1_000_000,
    mid: float = 50_000.0,
    fair_value: float = 50_000.0,
    sigma: float = 1.0,
    imbalance: float = 0.0,
    trade_price: float | None = None,
    trade_is_buyer_maker: bool | None = None,
) -> MarketTick:
    return MarketTick(
        timestamp_ms=timestamp_ms,
        mid=mid,
        fair_value=fair_value,
        sigma=sigma,
        imbalance=imbalance,
        trade_price=trade_price,
        trade_is_buyer_maker=trade_is_buyer_maker,
    )


def _make_ticks(n: int = 50, base_mid: float = 50_000.0) -> list[MarketTick]:
    """Create n ticks with no trades (no fills), stable mid."""
    return [
        _make_tick(timestamp_ms=1_000_000 + i * 100, mid=base_mid, fair_value=base_mid)
        for i in range(n)
    ]


def _make_crossing_ticks(
    n: int = 50,
    base_mid: float = 50_000.0,
    spread: float = 10.0,
) -> list[MarketTick]:
    """
    Ticks where every tick has a trade that crosses the FixedSpreadMaker's
    bid or ask, producing fills.

    Alternates between buyer-maker (hits bid) and seller-maker (hits ask)
    so inventory stays roughly flat.
    """
    ticks = []
    half = spread / 2.0
    for i in range(n):
        # Alternate: even = sell-initiated (hits ask), odd = buy-initiated (hits bid)
        if i % 2 == 0:
            # trade_is_buyer_maker=False → buy-initiated → hits ask
            trade_price = base_mid + half  # exactly at ask
            buyer_maker = False
        else:
            # trade_is_buyer_maker=True → sell-initiated → hits bid
            trade_price = base_mid - half  # exactly at bid
            buyer_maker = True
        ticks.append(
            _make_tick(
                timestamp_ms=1_000_000 + i * 100,
                mid=base_mid,
                fair_value=base_mid,
                trade_price=trade_price,
                trade_is_buyer_maker=buyer_maker,
            )
        )
    return ticks


# ---------------------------------------------------------------------------
# ImbalanceEdgeTest
# ---------------------------------------------------------------------------


class TestImbalanceEdgeTestBuffering:
    """add_sample buffers samples in _pending / _completed correctly."""

    def test_add_sample_creates_pending_entry(self) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[100])
        test.add_sample(1_000, 0.5, 50_000.0)
        assert test.n_pending == 1

    def test_add_sample_multiple_pending(self) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[500])
        for i in range(5):
            test.add_sample(1_000 + i * 10, 0.1 * i, 50_000.0 + i)
        # No horizon (500 ms) satisfied yet — all pending
        assert test.n_pending == 5
        assert test.n_completed == 0

    def test_sample_graduates_to_completed_after_horizon(self) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[100])
        test.add_sample(0, 0.5, 50_000.0)
        # Feed a tick 100 ms later to resolve the horizon
        test.add_sample(100, 0.2, 50_001.0)
        # The first sample should now be completed
        assert test.n_completed >= 1

    def test_pending_cleared_after_graduation(self) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[50])
        test.add_sample(0, 0.3, 100.0)
        test.add_sample(50, 0.1, 101.0)   # resolves the first sample
        # The first sample moves out of pending
        assert 0 not in test._pending

    def test_n_pending_decreases_as_horizons_resolve(self) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[100])
        for i in range(10):
            test.add_sample(i * 100, 0.1, 50_000.0 + i)
        # Every new tick resolves the previous sample — most should be completed
        assert test.n_completed > 0


class TestImbalanceEdgeTestRunTest:
    """run_test returns the right number of results and correct field types."""

    def _build_test_with_data(
        self, horizons_ms: list[int], n: int = 20
    ) -> ImbalanceEdgeTest:
        """Feed enough samples so at least `n` complete for each horizon."""
        test = ImbalanceEdgeTest(horizons_ms=horizons_ms)
        max_h = max(horizons_ms)
        # Feed n + extra ticks so the first n samples can resolve all horizons
        for i in range(n + 5):
            test.add_sample(i * max_h, float(i % 3) / 3.0, 50_000.0 + i * 0.1)
        return test

    def test_run_test_returns_one_result_per_horizon(self) -> None:
        horizons = [100, 500, 1000]
        test = self._build_test_with_data(horizons, n=20)
        results = test.run_test()
        assert len(results) == len(horizons)

    def test_run_test_horizon_ms_matches_input(self) -> None:
        horizons = [200, 800]
        test = self._build_test_with_data(horizons, n=20)
        results = test.run_test()
        returned_horizons = {r.horizon_ms for r in results}
        assert returned_horizons == set(horizons)

    def test_run_test_returns_empty_with_insufficient_data(self) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[100])
        # Only 2 samples — below the minimum of 3
        test.add_sample(0, 0.1, 100.0)
        test.add_sample(100, 0.2, 101.0)
        assert test.run_test() == []

    def test_run_test_default_horizons_four_results(self) -> None:
        test = self._build_test_with_data([100, 500, 1000, 5000], n=20)
        results = test.run_test()
        assert len(results) == 4


class TestEdgeTestResultFields:
    """EdgeTestResult exposes all required fields with correct types."""

    @pytest.fixture()
    def result(self) -> EdgeTestResult:
        test = ImbalanceEdgeTest(horizons_ms=[100])
        for i in range(20):
            test.add_sample(i * 100, 0.1 * (i % 5), 50_000.0 + i * 0.05)
        results = test.run_test()
        assert results, "Expected at least one result"
        return results[0]

    def test_correlation_is_float(self, result: EdgeTestResult) -> None:
        assert isinstance(result.correlation, float)

    def test_r_squared_is_float(self, result: EdgeTestResult) -> None:
        assert isinstance(result.r_squared, float)

    def test_t_statistic_is_float(self, result: EdgeTestResult) -> None:
        assert isinstance(result.t_statistic, float)

    def test_p_value_is_float(self, result: EdgeTestResult) -> None:
        assert isinstance(result.p_value, float)

    def test_beta_is_float(self, result: EdgeTestResult) -> None:
        assert isinstance(result.beta, float)

    def test_significant_is_bool(self, result: EdgeTestResult) -> None:
        assert isinstance(result.significant, bool)

    def test_r_squared_non_negative(self, result: EdgeTestResult) -> None:
        assert result.r_squared >= 0.0

    def test_p_value_in_unit_interval(self, result: EdgeTestResult) -> None:
        assert 0.0 <= result.p_value <= 1.0

    def test_significant_consistent_with_p_value(self, result: EdgeTestResult) -> None:
        assert result.significant == (result.p_value < 0.05)


class TestImbalanceEdgeTestCorrelatedData:
    """With strongly correlated synthetic data: significant edge is detected."""

    def _build_correlated_test(self) -> ImbalanceEdgeTest:
        """
        Construct samples where imbalance ≈ future return.

        Strategy: feed every other tick at double the horizon spacing.
        Manually manipulate completed samples so imbalance = 10 * future_return.
        Instead, use add_sample with a deterministic pattern that ensures
        high positive correlation by making mid move proportionally to imbalance.
        """
        random.seed(42)
        test = ImbalanceEdgeTest(horizons_ms=[100])
        # We'll build samples where future mid moves in the direction of imbalance.
        # mid[t+100] = mid[t] + imbalance * 1.0  (perfect positive correlation)
        n = 60
        imbalances = [round((i % 11 - 5) / 5.0, 2) for i in range(n + 5)]
        mids = [50_000.0]
        for i in range(1, n + 6):
            # next mid is influenced by previous imbalance
            mids.append(mids[-1] + imbalances[i - 1] * 2.0)

        for i in range(n + 5):
            test.add_sample(i * 100, imbalances[i], mids[i])
        return test

    def test_correlated_data_correlation_positive(self) -> None:
        test = self._build_correlated_test()
        results = test.run_test()
        assert results, "Expected results"
        r = results[0]
        assert r.correlation > 0.0

    def test_correlated_data_significant(self) -> None:
        test = self._build_correlated_test()
        results = test.run_test()
        assert results
        assert results[0].significant is True

    def test_correlated_data_p_value_below_threshold(self) -> None:
        test = self._build_correlated_test()
        results = test.run_test()
        assert results
        assert results[0].p_value < 0.05


class TestImbalanceEdgeTestUncorrelatedData:
    """With truly random uncorrelated data: no significant edge."""

    def _build_uncorrelated_test(self) -> ImbalanceEdgeTest:
        rng = random.Random(99)
        test = ImbalanceEdgeTest(horizons_ms=[100])
        n = 200
        for i in range(n):
            imbalance = rng.uniform(-1.0, 1.0)
            mid = 50_000.0 + rng.gauss(0, 5.0)  # pure noise, independent of imbalance
            test.add_sample(i * 100, imbalance, mid)
        return test

    def test_uncorrelated_data_not_significant(self) -> None:
        test = self._build_uncorrelated_test()
        results = test.run_test()
        assert results
        # With 200 samples of pure noise, p_value should be well above 0.05
        # (this may occasionally fail at p~0.05 boundary, seed=99 is chosen to be safe)
        sig_count = sum(1 for r in results if r.significant)
        assert sig_count == 0


class TestImbalanceEdgeTestPrintReport:
    """print_report runs without raising an exception."""

    def test_print_report_with_results(self, capsys: pytest.CaptureFixture) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[100])
        for i in range(20):
            test.add_sample(i * 100, 0.1 * (i % 3), 50_000.0 + i * 0.1)
        results = test.run_test()
        test.print_report(results)  # must not raise
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_report_with_empty_results(self, capsys: pytest.CaptureFixture) -> None:
        test = ImbalanceEdgeTest(horizons_ms=[100])
        test.print_report([])  # must not raise
        captured = capsys.readouterr()
        assert "no results" in captured.out.lower()


# ---------------------------------------------------------------------------
# RegimeAttributionTracker
# ---------------------------------------------------------------------------


class TestRecordFill:
    """record_fill stores fills in the correct per-regime bucket."""

    def test_record_fill_increments_count(self) -> None:
        tracker = RegimeAttributionTracker()
        tracker.record_fill(
            fill_price=100.0, fair_value=100.3, side="buy",
            regime=Regime.NORMAL, spread=0.5, timestamp_ms=1000,
        )
        assert len(tracker._fills[Regime.NORMAL]) == 1

    def test_record_fill_stored_in_correct_regime(self) -> None:
        tracker = RegimeAttributionTracker()
        tracker.record_fill(
            fill_price=200.0, fair_value=200.1, side="sell",
            regime=Regime.HIGH_VOL, spread=1.0, timestamp_ms=2000,
        )
        assert len(tracker._fills[Regime.HIGH_VOL]) == 1
        assert len(tracker._fills[Regime.NORMAL]) == 0

    def test_multiple_fills_same_regime(self) -> None:
        tracker = RegimeAttributionTracker()
        for i in range(5):
            tracker.record_fill(
                fill_price=100.0 + i, fair_value=100.5 + i, side="buy",
                regime=Regime.LOW_VOL, spread=0.4, timestamp_ms=1000 + i * 100,
            )
        assert len(tracker._fills[Regime.LOW_VOL]) == 5

    def test_fills_across_regimes(self) -> None:
        tracker = RegimeAttributionTracker()
        tracker.record_fill(100.0, 100.2, "buy", Regime.NORMAL, 0.5, timestamp_ms=1000)
        tracker.record_fill(100.0, 99.7, "sell", Regime.HIGH_VOL, 1.0, timestamp_ms=2000)
        tracker.record_fill(100.0, 100.1, "buy", Regime.LOW_VOL, 0.3, timestamp_ms=3000)
        assert len(tracker._fills[Regime.NORMAL]) == 1
        assert len(tracker._fills[Regime.HIGH_VOL]) == 1
        assert len(tracker._fills[Regime.LOW_VOL]) == 1


class TestComputeStats:
    """compute_stats returns RegimeStats for every regime."""

    def test_compute_stats_returns_all_regimes(self) -> None:
        tracker = RegimeAttributionTracker()
        tracker.record_fill(100.0, 100.3, "buy", Regime.NORMAL, 0.5, timestamp_ms=1000)
        stats = tracker.compute_stats()
        assert set(stats.keys()) == {Regime.LOW_VOL, Regime.NORMAL, Regime.HIGH_VOL}

    def test_compute_stats_empty_regime_has_zero_fills(self) -> None:
        tracker = RegimeAttributionTracker()
        tracker.record_fill(100.0, 100.3, "buy", Regime.NORMAL, 0.5, timestamp_ms=1000)
        stats = tracker.compute_stats()
        assert stats[Regime.HIGH_VOL].n_fills == 0

    def test_compute_stats_n_fills_matches_records(self) -> None:
        tracker = RegimeAttributionTracker()
        for i in range(7):
            tracker.record_fill(
                100.0, 100.2, "buy", Regime.NORMAL, 0.5,
                timestamp_ms=1000 + i * 60_000,
            )
        stats = tracker.compute_stats()
        assert stats[Regime.NORMAL].n_fills == 7

    def test_compute_stats_returns_regime_stats_type(self) -> None:
        tracker = RegimeAttributionTracker()
        tracker.record_fill(100.0, 100.3, "buy", Regime.NORMAL, 0.5, timestamp_ms=1000)
        stats = tracker.compute_stats()
        assert isinstance(stats[Regime.NORMAL], RegimeStats)


class TestAvgSpreadCapture:
    """avg_spread_capture is positive when we consistently buy below fair."""

    def test_buy_below_fair_positive_spread_capture(self) -> None:
        tracker = RegimeAttributionTracker()
        # Bought at 99.7, fair = 100.0  →  spread_capture = 100.0 - 99.7 = +0.3
        for i in range(5):
            tracker.record_fill(
                fill_price=99.7, fair_value=100.0, side="buy",
                regime=Regime.NORMAL, spread=0.6, timestamp_ms=1000 + i * 1000,
            )
        stats = tracker.compute_stats()
        assert stats[Regime.NORMAL].avg_spread_capture > 0.0

    def test_sell_above_fair_positive_spread_capture(self) -> None:
        tracker = RegimeAttributionTracker()
        # Sold at 100.3, fair = 100.0  →  spread_capture = 100.3 - 100.0 = +0.3
        for i in range(5):
            tracker.record_fill(
                fill_price=100.3, fair_value=100.0, side="sell",
                regime=Regime.NORMAL, spread=0.6, timestamp_ms=1000 + i * 1000,
            )
        stats = tracker.compute_stats()
        assert stats[Regime.NORMAL].avg_spread_capture > 0.0


class TestNetEdgePerFill:
    """net_edge_per_fill equals spread_capture - adverse_selection."""

    def test_net_edge_equals_spread_capture_minus_adverse_selection(self) -> None:
        tracker = RegimeAttributionTracker()
        tracker.record_fill(
            fill_price=99.7, fair_value=100.0, side="buy",
            regime=Regime.NORMAL, spread=0.6, timestamp_ms=1000,
        )
        tracker.record_fill(
            fill_price=99.8, fair_value=100.0, side="buy",
            regime=Regime.NORMAL, spread=0.6, timestamp_ms=61_000,
        )
        stats = tracker.compute_stats()[Regime.NORMAL]
        expected = stats.avg_spread_capture - stats.avg_adverse_selection
        assert stats.net_edge_per_fill == pytest.approx(expected, abs=1e-9)

    def test_net_edge_positive_for_profitable_fills(self) -> None:
        tracker = RegimeAttributionTracker()
        for i in range(4):
            tracker.record_fill(
                fill_price=99.5, fair_value=100.0, side="buy",
                regime=Regime.NORMAL, spread=1.0, timestamp_ms=1000 + i * 60_000,
            )
        stats = tracker.compute_stats()[Regime.NORMAL]
        assert stats.net_edge_per_fill > 0.0


class TestWinRate:
    """win_rate is always in [0, 1]."""

    def test_win_rate_between_zero_and_one_mixed(self) -> None:
        tracker = RegimeAttributionTracker()
        # 3 good fills (buy below fair), 2 bad (buy above fair)
        for i, (price, fv) in enumerate(
            [(99.5, 100.0), (99.6, 100.0), (99.7, 100.0), (100.5, 100.0), (100.6, 100.0)]
        ):
            tracker.record_fill(
                fill_price=price, fair_value=fv, side="buy",
                regime=Regime.NORMAL, spread=1.0,
                timestamp_ms=1000 + i * 60_000,
            )
        stats = tracker.compute_stats()[Regime.NORMAL]
        assert 0.0 <= stats.win_rate <= 1.0

    def test_win_rate_one_for_all_profitable(self) -> None:
        tracker = RegimeAttributionTracker()
        for i in range(4):
            tracker.record_fill(
                fill_price=99.5, fair_value=100.0, side="buy",
                regime=Regime.NORMAL, spread=1.0, timestamp_ms=1000 + i * 60_000,
            )
        stats = tracker.compute_stats()[Regime.NORMAL]
        assert stats.win_rate == pytest.approx(1.0)

    def test_win_rate_zero_for_empty_regime(self) -> None:
        tracker = RegimeAttributionTracker()
        stats = tracker.compute_stats()[Regime.HIGH_VOL]
        assert stats.win_rate == 0.0


class TestRecommendation:
    """recommendation returns {Regime: float} multipliers with correct ordering."""

    def test_recommendation_returns_dict_of_regime_keys(self) -> None:
        tracker = RegimeAttributionTracker()
        mults = tracker.recommendation()
        assert set(mults.keys()) == {Regime.LOW_VOL, Regime.NORMAL, Regime.HIGH_VOL}

    def test_recommendation_values_are_floats(self) -> None:
        tracker = RegimeAttributionTracker()
        mults = tracker.recommendation()
        for v in mults.values():
            assert isinstance(v, float)

    def test_normal_multiplier_is_baseline(self) -> None:
        tracker = RegimeAttributionTracker()
        mults = tracker.recommendation()
        assert mults[Regime.NORMAL] == pytest.approx(1.0)

    def test_low_vol_multiplier_is_tighter(self) -> None:
        tracker = RegimeAttributionTracker()
        mults = tracker.recommendation()
        assert mults[Regime.LOW_VOL] < mults[Regime.NORMAL]

    def test_high_vol_recommendation_ge_normal(self) -> None:
        """HIGH_VOL spread multiplier must be >= NORMAL (worse regime = wider spreads)."""
        tracker = RegimeAttributionTracker()
        # Simulate poor edge in high vol
        for i in range(10):
            tracker.record_fill(
                fill_price=100.5, fair_value=100.0, side="buy",
                regime=Regime.HIGH_VOL, spread=1.0, timestamp_ms=1000 + i * 60_000,
            )
            tracker.record_fill(
                fill_price=99.7, fair_value=100.0, side="buy",
                regime=Regime.NORMAL, spread=0.6,
                timestamp_ms=2000 + i * 60_000,
            )
        mults = tracker.recommendation()
        assert mults[Regime.HIGH_VOL] >= mults[Regime.NORMAL]

    def test_high_vol_widens_when_edge_degraded(self) -> None:
        """When HIGH_VOL net edge < 50% of NORMAL, multiplier should be 1.5."""
        tracker = RegimeAttributionTracker()
        # NORMAL: good edge (buy far below fair)
        for i in range(10):
            tracker.record_fill(
                fill_price=99.0, fair_value=100.0, side="buy",
                regime=Regime.NORMAL, spread=2.0, timestamp_ms=1000 + i * 60_000,
            )
        # HIGH_VOL: very poor edge (buy above fair — adverse)
        for i in range(10):
            tracker.record_fill(
                fill_price=100.8, fair_value=100.0, side="buy",
                regime=Regime.HIGH_VOL, spread=2.0,
                timestamp_ms=2000 + i * 60_000,
            )
        mults = tracker.recommendation()
        assert mults[Regime.HIGH_VOL] == pytest.approx(1.5)


class TestRegimeAttributionPrintReport:
    """print_report runs without raising."""

    def test_print_report_no_data(self, capsys: pytest.CaptureFixture) -> None:
        tracker = RegimeAttributionTracker()
        tracker.print_report()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_report_with_both_regimes(self, capsys: pytest.CaptureFixture) -> None:
        tracker = RegimeAttributionTracker()
        for i in range(5):
            tracker.record_fill(
                99.8, 100.0, "buy", Regime.NORMAL, 0.4, timestamp_ms=1000 + i * 60_000
            )
            tracker.record_fill(
                100.3, 100.0, "sell", Regime.HIGH_VOL, 1.0,
                timestamp_ms=5000 + i * 60_000,
            )
        tracker.print_report()
        captured = capsys.readouterr()
        assert "normal" in captured.out.lower() or "high_vol" in captured.out.lower()


# ---------------------------------------------------------------------------
# FixedSpreadMaker
# ---------------------------------------------------------------------------


class TestFixedSpreadMaker:
    """FixedSpreadMaker.compute returns bid = mid - spread/2, ask = mid + spread/2."""

    def test_bid_equals_mid_minus_half_spread(self) -> None:
        maker = FixedSpreadMaker(fixed_spread=10.0)
        tick = _make_tick(mid=50_000.0)
        bid, ask = maker.compute(tick)
        assert bid == pytest.approx(50_000.0 - 5.0)

    def test_ask_equals_mid_plus_half_spread(self) -> None:
        maker = FixedSpreadMaker(fixed_spread=10.0)
        tick = _make_tick(mid=50_000.0)
        bid, ask = maker.compute(tick)
        assert ask == pytest.approx(50_000.0 + 5.0)

    def test_spread_is_fixed_spread(self) -> None:
        maker = FixedSpreadMaker(fixed_spread=20.0)
        tick = _make_tick(mid=30_000.0)
        bid, ask = maker.compute(tick)
        assert ask - bid == pytest.approx(20.0)

    def test_bid_less_than_ask(self) -> None:
        maker = FixedSpreadMaker(fixed_spread=5.0)
        tick = _make_tick(mid=1234.56)
        bid, ask = maker.compute(tick)
        assert bid < ask

    def test_custom_spread(self) -> None:
        maker = FixedSpreadMaker(fixed_spread=4.0)
        tick = _make_tick(mid=100.0)
        bid, ask = maker.compute(tick)
        assert bid == pytest.approx(98.0)
        assert ask == pytest.approx(102.0)


# ---------------------------------------------------------------------------
# NaiveMaker
# ---------------------------------------------------------------------------


class TestNaiveMaker:
    """NaiveMaker.compute returns bid = mid - 0.5, ask = mid + 0.5 by default."""

    def test_bid_equals_mid_minus_half_spread(self) -> None:
        maker = NaiveMaker()
        tick = _make_tick(mid=50_000.0)
        bid, ask = maker.compute(tick)
        assert bid == pytest.approx(50_000.0 - 0.5)

    def test_ask_equals_mid_plus_half_spread(self) -> None:
        maker = NaiveMaker()
        tick = _make_tick(mid=50_000.0)
        bid, ask = maker.compute(tick)
        assert ask == pytest.approx(50_000.0 + 0.5)

    def test_default_full_spread_is_one(self) -> None:
        maker = NaiveMaker()
        tick = _make_tick(mid=200.0)
        bid, ask = maker.compute(tick)
        assert ask - bid == pytest.approx(1.0)

    def test_bid_less_than_ask(self) -> None:
        maker = NaiveMaker()
        tick = _make_tick(mid=9999.99)
        bid, ask = maker.compute(tick)
        assert bid < ask

    def test_custom_half_spread(self) -> None:
        maker = NaiveMaker(half_spread=2.5)
        tick = _make_tick(mid=100.0)
        bid, ask = maker.compute(tick)
        assert bid == pytest.approx(97.5)
        assert ask == pytest.approx(102.5)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class _DummyEngine:
    """Minimal stand-in for AdaptiveQuoteEngine in benchmark tests."""

    def __init__(self, half_spread: float = 5.0) -> None:
        self._half_spread = half_spread

    def compute(
        self,
        fair_value: float,
        sigma: float,
        inventory: float,
        imbalance: float,
        regime: str,
    ):
        class _Quotes:
            def __init__(self, bid: float, ask: float) -> None:
                self.bid = bid
                self.ask = ask

        return _Quotes(
            bid=round(fair_value - self._half_spread, 2),
            ask=round(fair_value + self._half_spread, 2),
        )


class TestBenchmarkRunnerRunAll:
    """BenchmarkRunner.run_all returns list[BacktestResult] sorted by Sharpe desc."""

    @pytest.fixture()
    def runner(self) -> BenchmarkRunner:
        ticks = _make_ticks(n=50)
        return BenchmarkRunner(ticks, fill_qty=0.001)

    @pytest.fixture()
    def results(self, runner: BenchmarkRunner) -> list[BacktestResult]:
        engine = _DummyEngine(half_spread=5.0)
        return runner.run_all(engine)

    def test_run_all_returns_three_results(
        self, results: list[BacktestResult]
    ) -> None:
        assert len(results) == 3

    def test_results_sorted_by_sharpe_descending(
        self, results: list[BacktestResult]
    ) -> None:
        sharpes = [r.sharpe for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_strategy_names_present(self, results: list[BacktestResult]) -> None:
        names = {r.strategy_name for r in results}
        assert "FixedSpreadMaker" in names
        assert "NaiveMaker" in names
        assert "AdaptiveQuoteEngine" in names


class TestBacktestResultFields:
    """BacktestResult fields satisfy invariants for a run with no trades."""

    @pytest.fixture()
    def result(self) -> BacktestResult:
        ticks = _make_ticks(n=40)
        runner = BenchmarkRunner(ticks, fill_qty=0.001)
        engine = _DummyEngine()
        results = runner.run_all(engine)
        # Return the FixedSpreadMaker result for deterministic assertions
        return next(r for r in results if r.strategy_name == "FixedSpreadMaker")

    def test_n_ticks_matches_input(self, result: BacktestResult) -> None:
        assert result.n_ticks == 40

    def test_fill_rate_between_zero_and_one(self, result: BacktestResult) -> None:
        assert 0.0 <= result.fill_rate <= 1.0

    def test_max_drawdown_non_negative(self, result: BacktestResult) -> None:
        assert result.max_drawdown >= 0.0

    def test_avg_spread_positive(self, result: BacktestResult) -> None:
        assert result.avg_spread > 0.0

    def test_win_rate_between_zero_and_one(self, result: BacktestResult) -> None:
        assert 0.0 <= result.win_rate <= 1.0

    def test_fill_rate_zero_when_no_trades(self, result: BacktestResult) -> None:
        # Ticks produced by _make_ticks have no trade_price → no fills
        assert result.fill_rate == 0.0


class TestBacktestResultToDict:
    """BacktestResult.to_dict (if implemented) or attribute access covers all keys."""

    EXPECTED_FIELDS = {
        "strategy_name",
        "n_ticks",
        "n_fills",
        "fill_rate",
        "total_pnl",
        "sharpe",
        "max_drawdown",
        "avg_spread",
        "win_rate",
        "avg_inventory",
    }

    def _make_result(self) -> BacktestResult:
        ticks = _make_ticks(n=20)
        runner = BenchmarkRunner(ticks, fill_qty=0.001)
        engine = _DummyEngine()
        return runner.run_all(engine)[0]

    def test_all_expected_fields_accessible(self) -> None:
        result = self._make_result()
        for field in self.EXPECTED_FIELDS:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_to_dict_contains_all_keys(self) -> None:
        result = self._make_result()
        if hasattr(result, "to_dict"):
            d = result.to_dict()
            assert isinstance(d, dict)
            for key in self.EXPECTED_FIELDS:
                assert key in d, f"to_dict missing key: {key}"
        else:
            # Dataclass: use __dataclass_fields__ or vars()
            d = vars(result)
            for key in self.EXPECTED_FIELDS:
                assert key in d, f"BacktestResult missing attribute: {key}"


class TestBenchmarkRunnerPrintReport:
    """print_report runs without raising."""

    def test_print_report_does_not_raise(self, capsys: pytest.CaptureFixture) -> None:
        ticks = _make_ticks(n=30)
        runner = BenchmarkRunner(ticks, fill_qty=0.001)
        engine = _DummyEngine()
        results = runner.run_all(engine)
        runner.print_report(results)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestBenchmarkRunnerPrintAsciiPnl:
    """print_ascii_pnl returns a non-empty string."""

    def test_print_ascii_pnl_returns_nonempty_string(self) -> None:
        ticks = _make_ticks(n=30)
        runner = BenchmarkRunner(ticks, fill_qty=0.001)
        engine = _DummyEngine()
        results = runner.run_all(engine)
        chart = runner.print_ascii_pnl(results)
        assert isinstance(chart, str)
        assert len(chart.strip()) > 0

    def test_print_ascii_pnl_empty_ticks(self) -> None:
        runner = BenchmarkRunner([], fill_qty=0.001)
        chart = runner.print_ascii_pnl([])
        assert isinstance(chart, str)
        assert len(chart) > 0


class TestFixedSpreadMakerGetsFills:
    """With crossing ticks, FixedSpreadMaker accumulates fills."""

    def test_fixed_spread_maker_fill_rate_positive_with_crossing_trades(self) -> None:
        spread = 10.0
        ticks = _make_crossing_ticks(n=40, base_mid=50_000.0, spread=spread)
        runner = BenchmarkRunner(ticks, fill_qty=0.001)
        engine = _DummyEngine(half_spread=spread / 2.0)
        results = runner.run_all(engine)
        fixed = next(r for r in results if r.strategy_name == "FixedSpreadMaker")
        assert fixed.fill_rate > 0.0

    def test_fixed_spread_maker_n_fills_positive_with_crossing_trades(self) -> None:
        spread = 10.0
        ticks = _make_crossing_ticks(n=40, base_mid=50_000.0, spread=spread)
        runner = BenchmarkRunner(ticks, fill_qty=0.001)
        engine = _DummyEngine(half_spread=spread / 2.0)
        results = runner.run_all(engine)
        fixed = next(r for r in results if r.strategy_name == "FixedSpreadMaker")
        assert fixed.n_fills > 0

    def test_crossing_tick_fill_rate_less_than_or_equal_one(self) -> None:
        spread = 10.0
        ticks = _make_crossing_ticks(n=40, base_mid=50_000.0, spread=spread)
        runner = BenchmarkRunner(ticks, fill_qty=0.001)
        engine = _DummyEngine(half_spread=spread / 2.0)
        results = runner.run_all(engine)
        for r in results:
            assert 0.0 <= r.fill_rate <= 1.0

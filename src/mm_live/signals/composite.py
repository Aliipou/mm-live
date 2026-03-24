"""
Composite Edge Signal — stacks OFI + microprice + vol clustering.

Combines three signals into a single directional edge score.

    edge_score = w1·normalize(OFI) + w2·normalize(microprice_dev) + w3·normalize(vol_urgency)

Positive score → bullish edge (skew bid up, ask up)
Negative score → bearish edge (skew bid down, ask down)
Near zero → no edge (use base A-S without skew)

Includes online normalization via running mean/std (Welford's algorithm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mm_live.feed.orderbook import OrderBook
from mm_live.signals.imbalance import OrderFlowImbalance
from mm_live.signals.microprice import MicropriceSignal
from mm_live.signals.vol_clustering import VolClusteringSignal


class _WelfordNormalizer:
    """
    Online mean/std normalization using Welford's one-pass algorithm.

    Clips output to [-3, 3] standard deviations for outlier resistance.
    Returns the raw value unscaled until at least two observations have been
    seen (insufficient data for a meaningful std estimate).
    """

    _CLIP = 3.0

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0  # sum of squared deviations

    def normalize(self, x: float) -> float:
        """
        Update running statistics with ``x`` and return the z-score, clipped
        to ``[-3, 3]``.  Before sufficient data (n < 2) returns 0.0.
        """
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

        if self._n < 2:
            return 0.0

        variance = self._M2 / (self._n - 1)
        std = math.sqrt(variance) if variance > 0.0 else 1e-10
        z = (x - self._mean) / std
        return max(-self._CLIP, min(self._CLIP, z))


@dataclass
class CompositeEdgeSignal:
    """
    Stacked directional edge signal combining OFI, microprice, and vol clustering.

    Parameters
    ----------
    ofi_weight:
        Weight on the order-flow imbalance component (default 0.5).
    microprice_weight:
        Weight on the microprice-deviation component (default 0.3).
    vol_weight:
        Weight on the vol-clustering urgency component (default 0.2).

    Weights are normalised internally so they always sum to 1.0, preventing
    the edge score from drifting outside [-1, +1] even if custom weights do
    not sum to exactly 1.

    Sub-signals are owned by this class and updated together on each call to
    ``update()``.
    """

    ofi_weight: float = 0.5
    microprice_weight: float = 0.3
    vol_weight: float = 0.2

    # Sub-signals
    _ofi: OrderFlowImbalance = field(init=False)
    _microprice: MicropriceSignal = field(init=False)
    _vol: VolClusteringSignal = field(init=False)

    # Per-component Welford normalizers
    _norm_ofi: _WelfordNormalizer = field(init=False)
    _norm_mp: _WelfordNormalizer = field(init=False)
    _norm_vol: _WelfordNormalizer = field(init=False)

    # Latest computed values
    _score: float = field(init=False, default=0.0)
    _components: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._ofi = OrderFlowImbalance()
        self._microprice = MicropriceSignal()
        self._vol = VolClusteringSignal()

        self._norm_ofi = _WelfordNormalizer()
        self._norm_mp = _WelfordNormalizer()
        self._norm_vol = _WelfordNormalizer()

        self._components = {
            "ofi": 0.0,
            "microprice": 0.0,
            "vol_clustering": 0.0,
            "composite": 0.0,
        }

    def update(self, book: OrderBook, last_price: float) -> float:
        """
        Update all sub-signals and compute the composite edge score.

        Parameters
        ----------
        book:
            Current L2 order book state.
        last_price:
            Most recent trade price, fed to the vol clustering signal.

        Returns
        -------
        float
            Edge score in [-1, +1].  Positive = bullish, negative = bearish.
        """
        # --- raw signal values ---
        ofi_raw: float = self._ofi.update(book)

        self._microprice.update(book)
        mp_raw: float = self._microprice.deviation_from_mid(book)

        vol_forecast: float = self._vol.update(last_price)
        # vol urgency is already in [0, 1]; we sign it with OFI direction so
        # it amplifies the existing directional edge rather than always
        # appearing bullish.
        ofi_sign = 1.0 if ofi_raw >= 0.0 else -1.0
        vol_signed: float = ofi_sign * self._vol.urgency

        # --- normalize each component ---
        ofi_norm: float = self._norm_ofi.normalize(ofi_raw)
        mp_norm: float = self._norm_mp.normalize(mp_raw)
        vol_norm: float = self._norm_vol.normalize(vol_signed)

        # --- weight and sum ---
        total_w = self.ofi_weight + self.microprice_weight + self.vol_weight
        if total_w == 0.0:
            total_w = 1.0

        raw_score = (
            self.ofi_weight * ofi_norm
            + self.microprice_weight * mp_norm
            + self.vol_weight * vol_norm
        ) / total_w

        # Clip to [-1, +1]: the Welford normalizer clips each component to
        # [-3, 3] sigma, so the weighted sum can in theory slightly exceed 1.
        self._score = max(-1.0, min(1.0, raw_score))

        self._components = {
            "ofi": ofi_norm,
            "microprice": mp_norm,
            "vol_clustering": vol_norm,
            "composite": self._score,
        }

        return self._score

    @property
    def score(self) -> float:
        """Last computed edge score (0.0 before first update)."""
        return self._score

    @property
    def components(self) -> dict[str, float]:
        """
        Per-component normalised values plus the composite score.

        Keys: ``"ofi"``, ``"microprice"``, ``"vol_clustering"``, ``"composite"``.
        """
        return dict(self._components)

    def is_strong(self, threshold: float = 0.3) -> bool:
        """
        Return True when the absolute edge score exceeds ``threshold``.

        A strong signal indicates a clear directional edge; the strategy
        should apply quote skew.  Near-zero scores → symmetric quoting.
        """
        return abs(self._score) > threshold

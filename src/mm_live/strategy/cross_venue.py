"""
Cross-venue spread detection and hedge logic.

Analyses the ``UnifiedBook`` to identify arbitrage opportunities across venues
(e.g. Binance vs OKX) and determines when an inventory imbalance is large
enough to warrant a hedge trade.

Typical usage::

    strategy = CrossVenueStrategy(fee_bps=7.0, min_net_spread=5.0, max_qty=0.01)
    signal = strategy.check_arb(unified_book)
    if signal.exists:
        print(f"Arb: buy {signal.qty} BTC on {signal.buy_venue} @ {signal.buy_price}, "
              f"sell on {signal.sell_venue} @ {signal.sell_price}, "
              f"net profit/BTC = {signal.net_spread:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass

from mm_live.feed.unified_book import UnifiedBook


@dataclass
class ArbSignal:
    """
    Result of a single cross-venue arbitrage scan.

    Fields
    ------
    exists:
        ``True`` when the net spread clears the minimum threshold and a
        non-zero tradeable quantity is available.
    buy_venue:
        Venue where the cheaper ask was found.
    sell_venue:
        Venue where the higher bid was found.
    buy_price:
        Ask price on *buy_venue* (the price we would pay).
    sell_price:
        Bid price on *sell_venue* (the price we would receive).
    gross_spread:
        ``sell_price - buy_price`` in USD.
    net_spread:
        ``gross_spread`` minus estimated round-trip fee cost
        (``2 * fee_bps / 10_000 * mid``).  Negative values mean the arb
        is loss-making after fees.
    qty:
        Executable quantity in BTC = ``min(buy_ask_size, sell_bid_size, max_qty)``.
    """

    exists: bool
    buy_venue: str
    sell_venue: str
    buy_price: float
    sell_price: float
    gross_spread: float     # sell_price - buy_price  (USD)
    net_spread: float       # gross_spread - round-trip fee cost  (USD per BTC)
    qty: float              # min(buy_size, sell_size, max_qty)


# Sentinel returned when the unified book has insufficient data.
_NO_ARB = ArbSignal(
    exists=False,
    buy_venue="",
    sell_venue="",
    buy_price=0.0,
    sell_price=0.0,
    gross_spread=0.0,
    net_spread=0.0,
    qty=0.0,
)


@dataclass
class CrossVenueStrategy:
    """
    Cross-venue arbitrage detector and hedge advisor.

    Parameters
    ----------
    fee_bps:
        Estimated total round-trip fee in basis points.  Default ``7.0``
        corresponds to ~3.5 bps per side (typical taker fee on major venues).
    min_net_spread:
        Minimum net profit per unit (USD/BTC) required for ``ArbSignal.exists``
        to be ``True``.
    max_qty:
        Maximum BTC quantity to recommend per arbitrage leg.
    """

    fee_bps: float = 7.0
    min_net_spread: float = 5.0   # USD / BTC
    max_qty: float = 0.01         # BTC

    def check_arb(self, unified_book: UnifiedBook) -> ArbSignal:
        """
        Scan the unified book for a cross-venue arbitrage opportunity.

        The method checks whether the highest bid on any venue exceeds the
        lowest ask on a *different* venue.  When it does, it calculates the
        gross and net spread, sizes the trade, and returns an ``ArbSignal``.

        Parameters
        ----------
        unified_book:
            Current state of the multi-venue aggregator.

        Returns
        -------
        ArbSignal
            ``exists=True`` only when there is a net-positive arb that clears
            ``min_net_spread``.  Otherwise returns the ``_NO_ARB`` sentinel.
        """
        bid_quote = unified_book.best_bid()
        ask_quote = unified_book.best_ask()

        if bid_quote is None or ask_quote is None:
            return _NO_ARB

        # Cross-venue arb requires the legs to be on different venues.
        if bid_quote.venue == ask_quote.venue:
            return _NO_ARB

        gross_spread = bid_quote.bid - ask_quote.ask

        # Estimate fee drag: 2 sides × fee_bps / 10_000 × mid price.
        mid = (bid_quote.bid + ask_quote.ask) / 2.0
        fee_drag = 2.0 * (self.fee_bps / 10_000.0) * mid

        net_spread = gross_spread - fee_drag

        if net_spread < self.min_net_spread:
            return _NO_ARB

        qty = min(bid_quote.bid_size, ask_quote.ask_size, self.max_qty)

        if qty <= 0.0:
            return _NO_ARB

        return ArbSignal(
            exists=True,
            buy_venue=ask_quote.venue,    # we buy on the venue with the cheap ask
            sell_venue=bid_quote.venue,   # we sell on the venue with the high bid
            buy_price=ask_quote.ask,
            sell_price=bid_quote.bid,
            gross_spread=gross_spread,
            net_spread=net_spread,
            qty=qty,
        )

    def should_hedge(self, inventory: float, max_inventory: float) -> bool:
        """
        Return ``True`` when the absolute inventory position exceeds the
        *max_inventory* threshold and a hedge trade is warranted.

        Parameters
        ----------
        inventory:
            Current net position in BTC (positive = long, negative = short).
        max_inventory:
            Maximum tolerated absolute position before hedging is triggered.

        Returns
        -------
        bool
            ``True`` if ``|inventory| >= max_inventory``.
        """
        if max_inventory <= 0.0:
            return False
        return abs(inventory) >= max_inventory

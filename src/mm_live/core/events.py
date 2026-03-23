"""
Event types for the mm-live event loop.

Every piece of data that flows through the system is a typed event.
The engine dispatches events; handlers update state and produce new events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    BOOK_UPDATE = auto()    # order book changed
    TRADE = auto()          # a trade happened on exchange
    SIGNAL = auto()         # computed signal (fv, vol, imbalance)
    QUOTES = auto()         # new optimal bid/ask computed
    FILL = auto()           # simulated fill on our quote
    RISK_BREACH = auto()    # risk limit hit
    TICK = auto()           # periodic heartbeat


@dataclass(frozen=True)
class MarketEvent:
    type: EventType
    timestamp_ms: int
    data: dict[str, Any] = field(default_factory=dict)

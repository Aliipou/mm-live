"""
mm-live: Event-driven market making system — paper trading mode.

Run:
    python main.py

Env vars:
    MM_SYMBOL=btcusdt
    MM_GAMMA=0.05
    MM_K=1.5
    MM_T_HORIZON=600
    MM_MAX_INVENTORY=0.1
"""

from __future__ import annotations

import asyncio
import logging
import os

from mm_live.core.engine import Engine, EngineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _f(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


async def main() -> None:
    config = EngineConfig(
        gamma=_f("MM_GAMMA", 0.05),
        k=_f("MM_K", 1.5),
        T_horizon=_f("MM_T_HORIZON", 600),
        max_inventory=_f("MM_MAX_INVENTORY", 0.1),
    )
    engine = Engine(config)
    await engine.run_with_feed(os.environ.get("MM_SYMBOL", "btcusdt"))


if __name__ == "__main__":
    asyncio.run(main())

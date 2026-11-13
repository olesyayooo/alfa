"""Backward-compatible CLI: python -m alfa_search delegates to baseline."""

import asyncio
import logging

from .baseline.__main__ import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

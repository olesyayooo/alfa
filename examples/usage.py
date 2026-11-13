"""Run from project root: python -m examples.usage 'Как привлечь клиентов?'"""

import asyncio
import json
import sys

from alfa_search import BaselineSearch


async def main() -> None:
    # Create once in the backend's service layer; indexing is a separate job.
    baseline = BaselineSearch()  # DATABASE_URL
    results = await baseline.search(" ".join(sys.argv[1:]) or "Как привлечь клиентов?", top_k=50)
    print(json.dumps(results, default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

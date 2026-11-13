"""Run python -m examples.semantic_usage 'Как привлечь клиентов?' after building."""

import asyncio
import json
import sys

from alfa_search import SemanticSearch


async def main() -> None:
    semantic = SemanticSearch()  # One instance per backend worker; DATABASE_URL.
    results = await semantic.search(" ".join(sys.argv[1:]) or "Как привлечь клиентов?", top_k=50)
    print(json.dumps(results, default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

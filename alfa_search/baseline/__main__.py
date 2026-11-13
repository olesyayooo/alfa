import argparse
import asyncio
import json
import logging

from .service import BaselineSearch


async def main() -> None:
    parser = argparse.ArgumentParser(description="Alfa-Course PostgreSQL BM25F")
    parser.add_argument("command", choices=["build", "update", "search"])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    baseline = BaselineSearch()
    if args.command == "build":
        await baseline.build_index()
    elif args.command == "update":
        await baseline.update_index()
    else:
        print(json.dumps(await baseline.search(args.query, args.top_k), default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

import argparse
import asyncio
import json
import logging

from .config import SemanticConfig
from .encoder import BGEEncoder
from .service import SemanticSearch


async def main() -> None:
    parser = argparse.ArgumentParser(description="Local BGE-M3 semantic search with PostgreSQL pgvector")
    parser.add_argument("command", choices=["migrate", "model", "build", "update", "search"])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    config = SemanticConfig(batch_size=args.batch_size, max_tokens=args.max_tokens)
    if args.command == "model":
        logging.warning("Loading BGE-M3; first run downloads several GB of model files unless SEMANTIC_OFFLINE=1")
        await asyncio.to_thread(BGEEncoder(config).warmup)
        return
    semantic = SemanticSearch(config=config)
    if args.command == "migrate":
        await semantic.migrate()
    elif args.command in {"build", "update"}:
        logging.warning("Indexing may download BGE-M3 if missing from cache; compute can take time")
        if args.command == "build":
            await semantic.build_index()
        else:
            await semantic.update_index()
    else:
        print(json.dumps(await semantic.search(args.query, args.top_k), default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

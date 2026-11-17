"""Command-line interface: `docrag ingest`, `docrag ask`, `docrag serve`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _pdfs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.pdf")) if p.is_dir() else [p])
    return files


def cmd_ingest(args: argparse.Namespace) -> int:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings()
    files = _pdfs(args.paths)
    if not files:
        print("No PDF files found.", file=sys.stderr)
        return 1
    for path in files:
        r = pipeline.ingest(path, force_ocr=args.force_ocr)
        print(f"{r.document}: {r.pages} pages ({r.ocr_pages} via OCR) -> {r.chunks} chunks")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings()
    sources, tokens = pipeline.ask_stream(args.question, args.top_k)
    for token in tokens:
        print(token, end="", flush=True)
    print()
    if sources:
        print("\nSources:")
        for s in sources:
            print(f"  [{s.number}] {s.document}, page {s.page} (similarity {s.score:.2f})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "scanned_docs_rag.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docrag", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="show OCR progress logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="add PDFs (files or folders) to the index")
    p.add_argument("paths", nargs="+")
    p.add_argument("--force-ocr", action="store_true", help="OCR every page, ignore text layer")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question about the ingested documents")
    p.add_argument("question")
    p.add_argument("-k", "--top-k", type=int, default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

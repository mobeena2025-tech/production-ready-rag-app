"""Measure retrieval quality on a set of labelled questions.

Each line of the questions file is JSON:
  {"question": "...", "source": "file.pdf", "page": 3}

For each question we check whether a chunk from the expected page is in the
top-k results (hit rate / recall@k) and how high it ranks (MRR).

Usage:
  python scripts/evaluate.py eval/questions.jsonl --docs eval/docs --k 4
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Settings  # noqa: E402
from rag.pipeline import build_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", type=Path)
    ap.add_argument("--docs", type=Path, required=True, help="folder of documents to index")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--provider", default=None, help="gemini or local (default: from .env)")
    args = ap.parse_args()

    overrides = {"chroma_dir": tempfile.mkdtemp(), "min_score": 0.0}
    if args.provider:
        overrides["provider"] = args.provider
    rag = build_pipeline(Settings(**overrides))

    for f in sorted(args.docs.iterdir()):
        if f.is_file():
            r = rag.ingest(f.read_bytes(), f.name)
            print(f"indexed {f.name}: {r.chunks} chunks")

    rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    hits, rr = 0, 0.0
    for row in rows:
        results = rag.retrieve(row["question"], k=args.k)
        rank = next(
            (
                i
                for i, h in enumerate(results, 1)
                if h.source == row["source"] and h.page == row.get("page", h.page)
            ),
            None,
        )
        hits += rank is not None
        rr += 1 / rank if rank else 0
        print(f"{'✓' if rank else '✗'} rank={rank or '-':<2} {row['question']}")

    n = len(rows) or 1
    print(f"\nquestions: {len(rows)}  recall@{args.k}: {hits / n:.2f}  MRR: {rr / n:.2f}")


if __name__ == "__main__":
    main()

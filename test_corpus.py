# test_corpus.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.corpus.builder import (
    build_full_corpus,
    load_corpus,
    corpus_to_papers,
)


def main():
    print("=" * 60)
    print("STEP 1: Build full evaluation corpus")
    print("=" * 60)
    metadata = build_full_corpus()

    print("\n" + "=" * 60)
    print("STEP 2: Load and verify saved corpus")
    print("=" * 60)
    combined = load_corpus("corpus_combined.json")
    print(f"Loaded {len(combined)} papers from combined corpus")

    print("\n" + "=" * 60)
    print("STEP 3: Convert to Paper objects")
    print("=" * 60)
    papers = corpus_to_papers(combined)
    print(f"Converted {len(papers)} Paper objects")

    print("\n" + "=" * 60)
    print("STEP 4: Sample papers")
    print("=" * 60)
    for i, p in enumerate(papers[:5], 1):
        print(f"\n  {i}. {p.title}")
        print(f"     Authors : {', '.join(p.authors[:2])}")
        print(f"     Year    : {p.year}")
        print(f"     DOI     : {p.doi or 'N/A'}")
        print(f"     Relevance: {combined[i-1]['relevance']}")

    print("\n" + "=" * 60)
    print("CORPUS STATS")
    print("=" * 60)
    print(f"  Total papers     : {metadata['total_papers']}")
    print(f"  Core papers      : {metadata['core_total']}")
    print(f"  Peripheral papers: {metadata['peripheral_total']}")
    for tid, stats in metadata["topics"].items():
        print(f"\n  {stats['name']}")
        print(f"    Papers     : {stats['total']}")
        print(f"    Year range : {stats['year_range']}")


if __name__ == "__main__":
    main()
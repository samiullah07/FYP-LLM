import requests
import json
import csv
from pathlib import Path

BASE_URL = "https://api.openalex.org"

def search_openalex_works(query: str, per_page: int = 5) -> dict:
    url = f"{BASE_URL}/works"
    params = {
        "search": query,
        "per_page": per_page,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def main():
    query = "large language model hallucination"

    print(f"Querying OpenAlex for: {query!r}")
    data = search_openalex_works(query, per_page=5)

    # --- Console summary for the report ---
    print("\n=== OpenAlex summary (top 3 works) ===")
    for i, work in enumerate(data.get("results", [])[:3], start=1):
        wid = work.get("id")
        title = work.get("display_name")
        year = work.get("publication_year")
        cited_by = work.get("cited_by_count")
        doi = work.get("doi")
        print(f"{i}. id={wid}")
        print(f"   title={title}")
        print(f"   year={year}, cited_by={cited_by}, doi={doi}")

    # --- Save full JSON for evidence ---
    json_path = Path("openalex_results.json")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved full JSON response to: {json_path.resolve()}")

    # --- Save small CSV table for screenshot/table ---
    csv_path = Path("openalex_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "year", "cited_by", "doi"])
        for work in data.get("results", []):
            writer.writerow([
                work.get("id"),
                work.get("display_name"),
                work.get("publication_year"),
                work.get("cited_by_count"),
                work.get("doi"),
            ])
    print(f"Saved tabular results to: {csv_path.resolve()}")

if __name__ == "__main__":
    main()

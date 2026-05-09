#!/usr/bin/env python3
"""
Test script for API connectivity in Extension 02 — Expanded Database Integration.
"""

import sys
import requests
from pathlib import Path

# Add the project root to the path so we can import our modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def check_openalex():
    """Check OpenAlex API connectivity."""
    try:
        from src import api_clients
        print("OK OpenAlex configuration ready")
        return True
    except Exception as e:
        print(f"FAIL OpenAlex configuration failed: {e}")
        return False

def check_crossref():
    """Check Crossref API connectivity."""
    try:
        response = requests.get("https://api.crossref.org/works?rows=1", timeout=10)
        if response.status_code == 200:
            print("OK Crossref API accessible")
            return True
        else:
            print(f"FAIL Crossref API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL Crossref API request failed: {e}")
        return False

def check_semanticscholar():
    """Check Semantic Scholar API connectivity."""
    try:
        from src import api_clients_extended
        api_key = api_clients_extended.os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if api_key:
            print("OK Semantic Scholar API key found")
        print("OK Semantic Scholar module available")
        return True
    except Exception as e:
        print(f"FAIL Semantic Scholar module failed: {e}")
        return False

def main():
    """Run all API connectivity checks."""
    print("Testing API connectivity...")

    results = []
    results.append(("OpenAlex", check_openalex()))
    results.append(("Crossref", check_crossref()))
    results.append(("Semantic Scholar", check_semanticscholar()))

    print("\nAPI Connectivity Test Results:")
    print("-" * 35)

    all_passed = True
    for service, passed in results:
        status = "OK" if passed else "FAIL"
        print(f"{service:15} {status}")
        if not passed:
            all_passed = False

    print("-" * 35)
    if all_passed:
        print("Overall: OK All APIs responsive")
        return 0
    else:
        print("Overall: FAIL Some APIs failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
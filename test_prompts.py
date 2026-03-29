# test_prompts.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.prompts import get_prompt, PROMPT_VERSION, PROMPT_REGISTRY

def main():
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Total prompts registered: {len(PROMPT_REGISTRY)}")
    print("\nAvailable prompts:")
    for name in PROMPT_REGISTRY:
        prompt = get_prompt(name)
        print(f"  {name:<35} ({len(prompt)} chars)")

    # Test a specific prompt with formatting
    print("\n--- PLANNER USER PROMPT (formatted) ---")
    prompt = get_prompt("planner_user")
    print(prompt.format(topic="Hallucination mitigation in LLMs"))

    print("\n--- BASELINE PROMPT (first 200 chars) ---")
    print(get_prompt("baseline")[:200])

if __name__ == "__main__":
    main()
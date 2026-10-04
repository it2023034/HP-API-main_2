import json
import re
from typing import List, Dict, Any

def load_file(path: str) -> str:
    """Reads and returns trimmed text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_json(path: str, data: Any) -> None:
    """Saves data formatted with each element/triple on a separate line."""
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, list):
            f.write("[\n")
            lines = [f"  {json.dumps(item, ensure_ascii=False)}" for item in data]
            f.write(",\n".join(lines))
            f.write("\n]\n")
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)


def remove_duplicates(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates a list of triple objects based on (entity, attribute, value)."""
    seen = set()
    result = []
    for item in triples:
        t = item["triple"]
        key = (t["entity"], t["attribute"], t["value"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def sort_triples_custom_priority(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sorts triple entries by priority."""
    def get_priority(item: Dict[str, Any]) -> int:
        triple = item.get("triple", {})
        attr = str(triple.get("attribute", "")).strip()
        has_exp = bool(str(item.get("explanation", "")).strip())

        if has_exp:
            return 0
        if attr == "communicatesWith":
            return 1
        return 2

    return sorted(results, key=get_priority)


def is_valid_phone(val: str) -> bool:
    """Validates phone numbers."""
    val = val.strip()
    if re.search(r'[a-zA-Z]', val):
        return False
    digits_only = re.sub(r'\D', '', val)
    return 7 <= len(digits_only) <= 15


def is_valid_email(val: str) -> bool:
    """Validates complete email structure."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, val.strip()))
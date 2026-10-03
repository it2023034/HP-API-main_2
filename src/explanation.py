import json
import re
from typing import List, Tuple, Dict, Any
from src.prompts import build_explanation_prompt


def load_triples_from_json(path: str) -> List[Tuple[str, str, str]]:
    """Loads structured triples from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triples = []
    for item in data:
        triple = item.get("triple", {})
        entity = triple.get("entity", "").strip()
        attribute = triple.get("attribute", "").strip()
        value = triple.get("value", "").strip()

        if entity and attribute and value:
            triples.append((entity, attribute, value))

    return triples


def extract_explanation_for_triple(dialogue_text: str, triple: Tuple[str, str, str], llm: Any) -> str:
    """Generates and cleans a natural language explanation for a single triple using the LLM."""
    prompt = build_explanation_prompt(dialogue_text, triple)
    raw = llm.invoke(prompt).strip()

    # Clean system prompt markers from raw output
    for marker in ["<|assistant|>", "TASK:", "RULES:", "DIALOGUE:", "TRIPLE:", "OUTPUT:"]:
        if marker in raw:
            raw = raw.split(marker)[-1].strip()

    text = " ".join(raw.split()).strip()

    if "Explanation:" in text:
        text = text.split("Explanation:", 1)[-1].strip()

    # Remove repeated triple string from generated explanation
    entity, attribute, value = triple
    triple_text = f"{entity} | {attribute} | {value}"
    if triple_text in text:
        text = text.replace(triple_text, "").strip()

    # Retain only the first full sentence
    match = re.search(r"^.*?[.!?](?=\s|$)", text)
    if match:
        text = match.group(0).strip()

    return text


def build_explanations(dialogue_text: str, triples: List[Tuple[str, str, str]], llm: Any) -> List[Dict[str, Any]]:
    """Generates explanations for a list of triples and formats output dictionary."""
    results = []

    for triple in triples:
        entity, attribute, value = triple
        explanation = extract_explanation_for_triple(dialogue_text, triple, llm)

        results.append({
            "triple": {
                "entity": entity,
                "attribute": attribute,
                "value": value
            },
            "explanation": explanation
        })

    return results
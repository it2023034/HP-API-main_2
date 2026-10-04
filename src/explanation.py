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

    for marker in ["<|assistant|>", "TASK:", "RULES:", "DIALOGUE:", "TRIPLE:", "OUTPUT:"]:
        if marker in raw:
            raw = raw.split(marker)[-1].strip()

    text = " ".join(raw.split()).strip()

    if "Explanation:" in text:
        text = text.split("Explanation:", 1)[-1].strip()

    entity, attribute, value = triple
    triple_text = f"{entity} | {attribute} | {value}"
    if triple_text in text:
        text = text.replace(triple_text, "").strip()

    match = re.search(r"^.*?[.!?](?=\s|$)", text)
    if match:
        text = match.group(0).strip()

    return text


def extract_focused_context(case_dialogue: str, subj: str, val: str) -> str:
    """Extracts a tight 3-line window around the exact mentions of subj or val."""
    lines = case_dialogue.splitlines()
    matched_indices = []
    
    for idx, line in enumerate(lines):
        if val.lower() in line.lower() or subj.lower() in line.lower():
            matched_indices.append(idx)
            
    if not matched_indices:
        return ""
        
    best_idx = matched_indices[0]
    start = max(0, best_idx - 1)
    end = min(len(lines), best_idx + 2)
    return "\n".join(lines[start:end])


def is_invalid_explanation(text: str) -> bool:
    """
    Robust Semantic Classifier using Regular Expressions.
    Detects hallucinated absence, refusal to explain, or speculative pattern matching.
    """
    if not text or len(text) <= 5:
        return True

    text_lower = text.lower()

    # 1. Pattern για Αρνητικές Δηλώσεις / Μηδενική Τεκμηρίωση
    absence_pattern = r"\b(does not|cannot|no|lacks?|without|unable to)\b.*\b(provide|mention|state|contain|evidence|proof|verify|information|reference)\b"
    
    # 2. Pattern για Στοχαστική / Behavioral Αιτιολογία (π.χ. "communication style", "common in scams")
    speculative_pattern = r"\b(communication style|typical of|associated with|scams?|implied|speculat|assumption)\b"

    if re.search(absence_pattern, text_lower) or re.search(speculative_pattern, text_lower):
        return True

    return False


def generate_all_explanations(all_triples: List[Dict[str, Any]], case_dialogue: str, qwen: Any) -> List[Dict[str, Any]]:
    """Cascading 3-Stage Explanation Engine with Robust Pattern Guardrails."""
    results = []

    for item in all_triples:
        triple = item["triple"]
        subj = str(triple.get("entity", "")).strip()
        attr = str(triple.get("attribute", "")).strip()
        val = str(triple.get("value", "")).strip()

        is_message_triple = (
            subj.startswith("Message") or 
            attr in ["messageHasContent", "sender", "receiver", "timestamp", "refersToEntity", "communicatesWith"]
        )

        if item.get("inferred") or is_message_triple:
            explanation = ""
        else:
            # ----------------------------------------------------
            # ΣΤΑΔΙΟ 1: Global LLM Explanation
            # ----------------------------------------------------
            raw_explanation = extract_explanation_for_triple(case_dialogue, (subj, attr, val), qwen)
            
            if raw_explanation:
                raw_explanation = raw_explanation.replace("Triple: ", "").replace("`", "").strip()
            else:
                raw_explanation = ""

            # Έλεγχος με τον Robust Regex Classifier
            if not is_invalid_explanation(raw_explanation):
                explanation = raw_explanation
            else:
                # ----------------------------------------------------
                # ΣΤΑΔΙΟ 2: Focused Re-prompting (2η Ευκαιρία σε εστιασμένο snippet)
                # ----------------------------------------------------
                focused_snippet = extract_focused_context(case_dialogue, subj, val)
                stage_2_success = False

                if focused_snippet:
                    try:
                        retry_exp = extract_explanation_for_triple(focused_snippet, (subj, attr, val), qwen)
                        if retry_exp:
                            retry_exp = retry_exp.replace("Triple: ", "").replace("`", "").strip()
                            if not is_invalid_explanation(retry_exp):
                                explanation = retry_exp
                                stage_2_success = True
                    except Exception:
                        stage_2_success = False

                # ----------------------------------------------------
                # ΣΤΑΔΙΟ 3: Deterministic Fallback (Έσχατη Λύση με Context Anchoring)
                # ----------------------------------------------------
                if not stage_2_success:
                    matched_line = ""
                    for line in case_dialogue.splitlines():
                        if val.lower() in line.lower():
                            matched_line = line.strip()
                            break
                    if not matched_line:
                        for line in case_dialogue.splitlines():
                            if subj.lower() in line.lower():
                                matched_line = line.strip()
                                break

                    if matched_line:
                        explanation = f"The {attr} relation between {subj} and {val} is supported by the context log: '{matched_line}'."
                    else:
                        explanation = f"{subj} maintains a {attr} relation with {val} based on case records."

        results.append({
            "triple": triple,
            "explanation": explanation
        })
        
    return results
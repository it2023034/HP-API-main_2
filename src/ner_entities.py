import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from src.utils import save_json

# Standard RFC-compliant RegEx for Generic Data Literal Sanitization
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-C0-9.-]+\.[a-zA-C]{2,}$")
URL_REGEX = re.compile(r"^(https?://|www\.)[^\s/$.?#].[^\s]*$", re.IGNORECASE)


# =====================================================================
# 1. LLM Extraction & Dynamic Classification with Cascading Engine
# =====================================================================

def parse_ner_output(raw_output: str) -> List[Dict[str, str]]:
    """Robustly parses NER outputs from LLM response."""
    ner_results = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            subj = parts[0].replace("Triple: ", "").replace("`", "").strip()
            pred = parts[1].replace("Triple: ", "").replace("`", "").strip()
            obj = parts[2].replace("Triple: ", "").replace("`", "").strip()

            if pred in ["rdfs:label", "name", "label"]:
                pred = "rdfs:label"
            elif pred in ["rdf:type", "type"]:
                pred = "rdf:type"

            ner_results.append({"subject": subj, "predicate": pred, "object": obj})
    return ner_results


def extract_ner_focused_snippet(context_text: str, entity_name: str) -> str:
    """Extracts a tight 3-line snippet around the mention of entity_name."""
    lines = context_text.splitlines()
    matched_indices = [idx for idx, line in enumerate(lines) if entity_name.lower() in line.lower()]
    
    if not matched_indices:
        return ""
        
    best_idx = matched_indices[0]
    start = max(0, best_idx - 1)
    end = min(len(lines), best_idx + 2)
    return "\n".join(lines[start:end])


def is_invalid_ner_explanation(exp_text: str, entity_name: str) -> bool:
    """
    Generic Semantic Classifier using Structural Regular Expressions.
    Detects hallucinated absence, refusal to explain, or prompt leakage.
    """
    if not exp_text or len(exp_text) <= 5:
        return True
        
    exp_lower = exp_text.lower()
    
    # 1. Structural Absentive Pattern
    absence_pattern = r"\b(does not|cannot|no|lacks?|without|unable to)\b.*\b(provide|mention|state|contain|evidence|proof|verify|information|reference)\b"
    if re.search(absence_pattern, exp_lower) or "no_evidence" in exp_lower or "not mentioned" in exp_lower:
        return True
        
    # 2. Structural Prompt Leakage or Invalid Artifacts
    if "task:" in exp_lower or "rules:" in exp_lower or "allowed classes" in exp_lower:
        return True

    return False


def is_primitive_data_value(val: str, allowed_classes: set) -> bool:
    """Determines if a string is a standard literal value or ontology meta-class name."""
    val_clean = val.strip()
    val_lower = val_clean.lower()

    # Skip empty or single characters
    if not val_clean or len(val_clean) <= 1 or val_clean.isdigit():
        return True

    # Skip ontology meta-class names (e.g. raw string "Account" or "Transaction")
    if val_lower in {c.lower() for c in allowed_classes}:
        return True

    # Skip RFC-compliant Emails and URLs (Data Property values)
    if EMAIL_REGEX.match(val_clean) or URL_REGEX.match(val_clean):
        return True

    # Skip monetary amounts / numeric measurements
    if re.match(r"^[\d,\.\s]+[a-zA-Z€\$£]+$", val_clean):
        return True

    return False


def run_ner_pipeline(unique_triples: List[Dict[str, Any]], context_text: str, qwen: Any) -> List[Dict[str, Any]]:
    """Dynamically extracts core domain entities directly from clean triples and classifies via Qwen."""
    from src.prompts import build_ner_explanation_prompt

    allowed_classes = {"Person", "Organization", "Location", "Account", "Transaction"}
    raw_entity_names = set()

    for item in unique_triples:
        triple = item.get("triple", {})
        subj = str(triple.get("entity", "")).strip()
        attr = str(triple.get("attribute", "")).strip()
        obj = str(triple.get("value", "")).strip()

        # Αγνοούμε δομικά μηνύματα ΚΑΙ data properties που εκφράζουν ρόλους / επικοινωνίες
        if subj.startswith("Message") or attr in [
            "hasRole", "messageHasContent", "sender", "receiver", "timestamp", 
            "refersToEntity", "hasPhoneNumber", "hasEmail", "website"
        ]:
            continue

        for val in [subj, obj]:
            if val and not val.startswith("Message"):
                if not is_primitive_data_value(val, allowed_classes):
                    raw_entity_names.add(val.strip())

    if not raw_entity_names:
        return []

    entities_list_str = "\n".join(f"- {name}" for name in sorted(raw_entity_names))

    classification_prompt = f"""
TASK: Classify each string into ONE strict ontology class ONLY IF it represents a distinct domain entity.
Allowed classes: [Person, Organization, Location, Account, Transaction]

RULES:
1. Rejects job titles, roles (e.g., Advisor, Consultant), primitive literals, emails, URLs, or standalone units.
2. If an item is a role title or literal value, DO NOT include it.

OUTPUT FORMAT (Pipe-separated):
String Name | ClassName

STRINGS TO EVALUATE:
{entities_list_str}

OUTPUT:
""".strip()

    print("[DEBUG NER] Dynamically classifying domain entities...")
    classification_result = qwen.invoke(classification_prompt)

    extracted_entities = []

    for line in classification_result.splitlines():
        line = line.strip()
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 2:
                name, ent_class = parts[0], parts[1]
                if ent_class in allowed_classes and not is_primitive_data_value(name, allowed_classes):
                    extracted_entities.append({"class": ent_class, "name": name})

    print(f"[DEBUG NER] Mapped {len(extracted_entities)} clean domain entities.")

    ner_with_explanations = []
    for entity in extracted_entities:
        entity_class = entity["class"]
        entity_name = entity["name"]
        explanation = ""

        print(f"[DEBUG NER] Generating explanation for {entity_name} ({entity_class})...")
        try:
            # ----------------------------------------------------
            # STAGE 1: Global Context Prompting
            # ----------------------------------------------------
            exp_prompt = build_ner_explanation_prompt(context_text, entity_class, entity_class, entity_name)
            raw_exp = qwen.invoke(exp_prompt).strip()
            cleaned_exp = raw_exp.replace("Triple: ", "").replace("`", "").strip()

            if not is_invalid_ner_explanation(cleaned_exp, entity_name):
                explanation = cleaned_exp
            else:
                # ----------------------------------------------------
                # STAGE 2: Focused Re-prompting (2nd Chance)
                # ----------------------------------------------------
                snippet = extract_ner_focused_snippet(context_text, entity_name)
                stage_2_success = False
                
                if snippet:
                    try:
                        retry_prompt = build_ner_explanation_prompt(snippet, entity_class, entity_class, entity_name)
                        retry_exp = qwen.invoke(retry_prompt).strip().replace("Triple: ", "").replace("`", "").strip()
                        if not is_invalid_ner_explanation(retry_exp, entity_name):
                            explanation = retry_exp
                            stage_2_success = True
                    except Exception:
                        stage_2_success = False

                # ----------------------------------------------------
                # STAGE 3: Deterministic Context Fallback
                # ----------------------------------------------------
                if not stage_2_success:
                    matched_line = next((line.strip() for line in context_text.splitlines() if entity_name.lower() in line.lower()), "")
                    if matched_line:
                        explanation = f"{entity_name} is classified as a {entity_class} based on its direct reference in the context: '{matched_line}'."
                    else:
                        explanation = f"{entity_name} is classified as a {entity_class} based on its semantic role in the case log."

        except Exception as e:
            print(f"  -> ERROR for {entity_name}: {e}")
            explanation = f"{entity_name} is classified as a {entity_class} based on context."

        ner_with_explanations.append({
            "triple": {"entity": entity_class, "attribute": "rdfs:label", "value": entity_name},
            "explanation": "",
        })
        ner_with_explanations.append({
            "triple": {"entity": entity_class, "attribute": "rdf:type", "value": entity_class},
            "explanation": explanation,
        })

    return ner_with_explanations


# =====================================================================
# 2. Global Entity ID Assignment & Mapping
# =====================================================================

def assign_global_entity_ids(raw_ner_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministically assigns incremental global IDs (Person1, Person2...)."""
    explanations_map = {}
    current_label = None
    
    for item in raw_ner_items:
        triple = item.get("triple", {})
        attr = triple.get("attribute", "").strip()
        val = triple.get("value", "").strip()
        exp = item.get("explanation", "").strip()

        if attr in ["label", "rdfs:label"] and val:
            current_label = val
        elif attr in ["rdf:type", "type"] and current_label:
            entity_class = val
            if exp:
                explanations_map[(entity_class.lower(), current_label.lower())] = exp
            current_label = None

    class_counters = defaultdict(int)
    label_to_id = {}
    processed_items = []
    
    for item in raw_ner_items:
        triple = item.get("triple", {})
        entity_class = triple.get("entity", "").strip()
        attr = triple.get("attribute", "").strip()
        val = triple.get("value", "").strip()
        
        if attr in ["label", "rdfs:label"] and val:
            lookup_key = (entity_class.lower(), val.lower())
            
            if lookup_key not in label_to_id:
                class_counters[entity_class] += 1
                assigned_id = f"{entity_class}{class_counters[entity_class]}"
                label_to_id[lookup_key] = assigned_id
            
            assigned_id = label_to_id[lookup_key]
            entity_exp = explanations_map.get(lookup_key, "")
            
            processed_items.append({
                "triple": {"entity": assigned_id, "attribute": "rdf:type", "value": entity_class},
                "explanation": entity_exp
            })
            processed_items.append({
                "triple": {"entity": assigned_id, "attribute": "rdfs:label", "value": val},
                "explanation": ""
            })
            
    return processed_items


def clean_triples(extract_path: str):
    """Maps text entity names in extracted triples to sequential Global IDs."""
    extract_dir = Path(extract_path)
    ner_file = extract_dir / "ner_entities_with_explanations.json"
    triples_file = extract_dir / "triples_explanations.json"

    if not ner_file.exists() or not triples_file.exists():
        print(f"Skipping clean_triples: missing input files in {extract_path}")
        return

    with open(ner_file, "r", encoding="utf-8") as f:
        ner_data = json.load(f)

    name_to_id = {}
    for item in ner_data:
        triple = item.get("triple", {})
        if triple.get("attribute") in ["label", "rdfs:label"]:
            entity_id = triple.get("entity", "").strip()
            name = str(triple.get("value", "")).strip().lower()
            if name and entity_id:
                name_to_id[name] = entity_id

    with open(triples_file, "r", encoding="utf-8") as f:
        triples_data = json.load(f)

    for item in triples_data:
        triple = item.get("triple", {})
        if not triple:
            continue
            
        subj = str(triple.get("entity", "")).strip().lower()
        obj = str(triple.get("value", "")).strip().lower()

        if subj in name_to_id:
            item["triple"]["entity"] = name_to_id[subj]
                
        if obj in name_to_id:
            item["triple"]["value"] = name_to_id[obj]

    final_clean_path = extract_dir / "final_clean_triples.json"
    save_json(final_clean_path, triples_data)
    print(f"Successfully generated {final_clean_path} with mapped IDs!")


if __name__ == "__main__":
    clean_triples("files/output")
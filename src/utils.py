import json
import re
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple
from collections import defaultdict
from rdflib import Graph, RDF, RDFS, OWL


# =====================================================================
# File Operations & Storage
# =====================================================================

def load_file(path: str) -> str:
    """Reads and returns trimmed text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_json(path: str, data: Any) -> None:
    """Saves data to a formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =====================================================================
# Sorting & Priority Utilities
# =====================================================================

def sort_triples_custom_priority(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sorts triple entries by priority:
    1. Entries with non-empty explanations.
    2. Entries with 'communicatesWith' attribute.
    3. All remaining entries.
    """
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


# =====================================================================
# Date-Based & Sliding Window Chunking Module (Strict Date Grouping)
# =====================================================================

def extract_date_str(timestamp_val: Any) -> str:
    """Extracts date string (YYYY-MM-DD or DD/MM/YYYY) from timestamp field."""
    if not timestamp_val or str(timestamp_val).strip() == "":
        return "UNKNOWN_DATE"
    
    val_str = str(timestamp_val).strip()
    return val_str.split(" ")[0].split("T")[0]


def create_chunks_by_date_and_window(
    messages: List[Dict[str, Any]], 
    max_messages_per_chunk: int = 15,
    overlap: int = 2
) -> List[List[Dict[str, Any]]]:
    """
    Groups messages strictly by date boundaries. 
    If a single day exceeds max_messages_per_chunk, applies a sliding window on that day only.
    """
    if not messages:
        return []

    grouped_by_date = defaultdict(list)
    for msg in messages:
        raw_ts = msg.get("Date", msg.get("Timestamp", msg.get("timestamp", "")))
        date_key = extract_date_str(raw_ts)
        grouped_by_date[date_key].append(msg)

    final_chunks: List[List[Dict[str, Any]]] = []
    step = max_messages_per_chunk - overlap

    for date_key, date_messages in grouped_by_date.items():
        total_msgs = len(date_messages)
        
        if total_msgs <= max_messages_per_chunk:
            final_chunks.append(date_messages)
        else:
            for start_idx in range(0, total_msgs, step):
                sub_chunk = date_messages[start_idx : start_idx + max_messages_per_chunk]
                if sub_chunk:
                    final_chunks.append(sub_chunk)
                if start_idx + max_messages_per_chunk >= total_msgs:
                    break

    return final_chunks


def format_chunk_to_text(chunk: List[Dict[str, Any]]) -> str:
    """Formats a message chunk into a clean string for LLM prompting."""
    formatted_lines = []
    for msg in chunk:
        sender = msg.get("Sender", msg.get("sender", "Unknown"))
        receiver = msg.get("Receiver", msg.get("receiver", "Unknown"))
        content = msg.get("Message", msg.get("content", ""))
        timestamp = msg.get("Timestamp", msg.get("timestamp", msg.get("Date", "")))

        if timestamp:
            formatted_lines.append(f"[{timestamp}] {sender} -> {receiver}: {content}")
        else:
            formatted_lines.append(f"{sender} -> {receiver}: {content}")

    return "\n".join(formatted_lines)


# =====================================================================
# LLM Semantic Canonicalization (Domain-Agnostic)
# =====================================================================

def canonicalize_attributes_with_llm(triples: List[Dict[str, Any]], attribute_name: str, qwen_model: Any) -> List[Dict[str, Any]]:
    """
    Dynamically groups paraphrased string values for a specific attribute 
    into standard canonical terms using LLM. Fully domain-agnostic.
    """
    if not qwen_model:
        return triples

    # Gather unique values for the target attribute
    raw_values = list({
        str(item["triple"]["value"]).strip().strip('"\'') 
        for item in triples 
        if item.get("triple", {}).get("attribute") == attribute_name
    })

    if len(raw_values) <= 1:
        return triples

    prompt = f"""
TASK:
You are provided with a list of extracted values for the attribute '{attribute_name}'.
Group paraphrased or near-synonymous values into concise, canonical semantic terms.

RULES:
1. Merge exact paraphrases or closely related terms into single concise labels.
2. Output ONLY a valid JSON object mapping each original string to its canonical term.

INPUT VALUES:
{json.dumps(raw_values, ensure_ascii=False)}

OUTPUT JSON ONLY:
""".strip()

    try:
        raw_res = qwen_model.invoke(prompt)
        json_match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if json_match:
            mapping = json.loads(json_match.group(0))
            for item in triples:
                if item.get("triple", {}).get("attribute") == attribute_name:
                    orig_val = str(item["triple"]["value"]).strip().strip('"\'')
                    if orig_val in mapping:
                        item["triple"]["value"] = mapping[orig_val]
    except Exception as e:
        print(f"Canonicalization skipped for {attribute_name} due to error: {e}")

    return triples


# =====================================================================
# Parsing & Filtering Utility Functions
# =====================================================================

def parse_triples(raw_output: str) -> List[Dict[str, Any]]:
    """Parses pipe-separated triple outputs from LLM response."""
    triples = []

    for line in raw_output.splitlines():
        line = line.strip()
        line = re.sub(r"^(\d+\.|\-|\*)\s*", "", line)
        line = line.replace("**", "").replace("`", "")

        if line.count("|") != 2:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue

        subj = parts[0].replace("Triple: ", "").strip()
        attr = parts[1].replace("Triple: ", "").strip()
        obj = parts[2].replace("Triple: ", "").strip()

        if obj.lower().startswith("http://") or obj.lower().startswith("https://"):
            obj = obj.split()[0].rstrip(":")
        elif ":" in obj and not obj.startswith("http"):
            obj = obj.split(":")[0].strip()

        triples.append({
            "triple": {
                "entity": subj,
                "attribute": attr,
                "value": obj
            }
        })

    return triples


def extract_allowed_relations(schema_text: str) -> Set[str]:
    """Extracts valid schema relation names from schema text."""
    relations = set()

    for line in schema_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            relations.add(parts[1])

    return relations


def normalize_relation(r: str) -> str:
    """Trims whitespace from relation string."""
    return r.strip()


def filter_triples(triples: List[Dict[str, Any]], allowed_relations: Set[str]) -> List[Dict[str, Any]]:
    """Filters extracted triples against allowed ontology relations and schema guardrails."""
    filtered = []
    normalized_allowed = {normalize_relation(r) for r in allowed_relations}
    primitive_types = {"string", "integer", "int", "double", "float", "boolean", "datetime", "datetimestamp"}

    for triple in triples:
        attr = normalize_relation(triple["triple"]["attribute"])
        subj = str(triple["triple"]["entity"]).strip()
        obj = str(triple["triple"]["value"]).strip()

        # Rule 1: Reject primitive datatype declarations
        if attr == "type" and obj.lower() in primitive_types:
            continue

        # Rule 2: Reject raw phone numbers or digit sequences operating as Subjects
        if subj.startswith("+") or (subj.isdigit() and len(subj) >= 8):
            continue

        # Rule 3: General Schema Relation Filter
        if attr in normalized_allowed:
            triple["triple"]["attribute"] = attr
            filtered.append(triple)

    return filtered


def filter_final_triples(triples: List[Dict[str, Any]], qwen_model: Any = None) -> List[Dict[str, Any]]:
    """Fully Domain-Agnostic Post-Processing Guardrails + LLM Canonicalization."""
    # Step 1: LLM Semantic Consolidation on text attributes (e.g. hasPurpose)
    if qwen_model:
        triples = canonicalize_attributes_with_llm(triples, "hasPurpose", qwen_model)

    clean_list = []
    seen_keys = set()

    invalid_placeholders = {
        "(not provided)", "not provided", "unknown", "none", 
        "n/a", "0", "0.0", "string", "double", "null"
    }

    for item in triples:
        triple = item.get("triple", {})
        subj = str(triple.get("entity", "")).strip().strip('"\'')
        attr = str(triple.get("attribute", "")).strip()
        val = str(triple.get("value", "")).strip().strip('"\'')

        # 1. Structural Filter: Drop empty / zero / datatype placeholders / rdf:type
        if val.lower() in invalid_placeholders or subj.lower() in invalid_placeholders or attr == "rdf:type":
            continue

        # 2. Datatype Enforcement: involvesAmount MUST be numeric
        if attr == "involvesAmount":
            numeric_match = re.search(r'\d+(\.\d+)?', val)
            if numeric_match:
                val = numeric_match.group(0)
            else:
                continue
        
        if attr == "hasEmail" and "@" not in val:
            continue

        # 3. Format Normalization: Clean formatting noise
        val = re.sub(r'\s+', ' ', val).strip()

        # Update cleaned triple
        item["triple"] = {"entity": subj, "attribute": attr, "value": val}

        # 4. Global Deduplication across chunks
        dedup_key = (subj.lower(), attr.lower(), val.lower())
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            clean_list.append(item)

    return clean_list


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


def parse_ner_output(raw_output: str) -> List[Dict[str, str]]:
    """Parses comma-separated NER outputs from LLM response."""
    ner_results = []
    
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
            
        parts = [p.strip() for p in line.split(", ")]
        
        if len(parts) == 3:
            subj = parts[0].replace("Triple: ", "").replace("`", "")
            pred = parts[1].replace("Triple: ", "").replace("`", "")
            obj = parts[2].replace("Triple: ", "").replace("`", "")
            ner_results.append({
                "subject": subj,
                "predicate": pred,
                "object": obj
            })
            
    return ner_results


# =====================================================================
# Ontology & Manual Extraction Functions
# =====================================================================

def short(uri: Any) -> str:
    """Extracts short local name from URI string."""
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.split("/")[-1]


def ttl_to_metapaths(path: str) -> str:
    """Parses Turtle ontology file and generates domain-property-range metapaths."""
    g = Graph()
    g.parse(path, format="turtle")

    rows = []
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)
        if domain and range_:
            rows.append(f"{short(domain)} | {short(prop)} | {short(range_)}")

    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)
        if domain and range_:
            rows.append(f"{short(domain)} | {short(prop)} | {short(range_)}")

    for cls in g.subjects(RDF.type, OWL.Class):
        rows.append(f"{short(cls)} | type | Class")
        
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        rows.append(f"{short(s)} | subClassOf | {short(o)}")

    return "\n".join(rows)


def extract_message_triples_manually(chat_text: str, file_name: str = "") -> List[Dict[str, Any]]:
    """Deterministically extracts structural Message triples from chat logs."""
    triples = []
    
    clean_file_suffix = ""
    if file_name:
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', Path(file_name).stem)
        clean_file_suffix = f"_{clean_name}"

    participants_match = re.search(r"Participants:\s*(.*)", chat_text)
    participants = []
    if participants_match:
        p_str = participants_match.group(1).replace(" and ", ",")
        participants = [p.strip() for p in p_str.split(",")]

    pattern = r"\[(.*?)\] (.*?): (.*)"
    
    msg_counter = 1
    for line in chat_text.splitlines():
        match = re.search(pattern, line)
        if match:
            timestamp = match.group(1).strip()
            sender = match.group(2).strip()
            content = match.group(3).strip()
            
            msg_id = f"Message_{msg_counter}{clean_file_suffix}"
            
            receiver = None
            if len(participants) == 2:
                receiver = participants[1] if sender == participants[0] else participants[0]
            
            triples.append({"triple": {"entity": msg_id, "attribute": "type", "value": "Message"}, "inferred": True})
            triples.append({"triple": {"entity": msg_id, "attribute": "sender", "value": sender}, "inferred": True})
            
            if receiver:
                triples.append({"triple": {"entity": msg_id, "attribute": "receiver", "value": receiver}, "inferred": True})
                
            triples.append({"triple": {"entity": msg_id, "attribute": "timestamp", "value": timestamp}, "inferred": True})
            triples.append({"triple": {"entity": msg_id, "attribute": "messageHasContent", "value": content}, "inferred": True})
            
            msg_counter += 1
            
    return triples


# =====================================================================
# High-Level Pipeline Functions
# =====================================================================

def run_ner_pipeline(case_dialogue: str, qwen: Any) -> List[Dict[str, Any]]:
    """Runs NER extraction and builds entity triples with explanations."""
    from src.prompts import build_ner_prompt, build_ner_explanation_prompt

    ner_prompt = build_ner_prompt(case_dialogue)
    ner_raw_result = qwen.invoke(ner_prompt)
    
    parsed_ner = parse_ner_output(ner_raw_result)
    
    clean_ner_lines = []
    ner_with_explanations = []
    entities_dict: Dict[str, Dict[str, str]] = {}
    
    for item in parsed_ner:
        if not item["subject"].startswith("-"):
            subj = item["subject"]
            pred = item["predicate"]
            obj = item["object"]
            
            clean_ner_lines.append(f"{subj} | {pred} | {obj}")
            
            if subj not in entities_dict:
                entities_dict[subj] = {"type": "", "name": ""}
                
            if pred == "rdf:type":
                entities_dict[subj]["type"] = obj
            elif pred == "label":
                entities_dict[subj]["name"] = obj
                
    for item in parsed_ner:
        if not item["subject"].startswith("-"):
            subj = item["subject"]
            pred = item["predicate"]
            obj = item["object"]
            
            if pred == "rdf:type":
                entity_name = entities_dict[subj]["name"]
                if entity_name:
                    exp_prompt = build_ner_explanation_prompt(case_dialogue, subj, obj, entity_name)
                    explanation = qwen.invoke(exp_prompt).strip()
                    explanation = explanation.replace("Triple: ", "").replace("`", "")
                else:
                    explanation = ""
            else:
                explanation = ""
                
            ner_with_explanations.append({
                "triple": {
                    "entity": subj,
                    "attribute": pred,
                    "value": obj
                },
                "explanation": explanation
            })
            
    seen_triples = set()
    final_clean_list = []
    
    for item in ner_with_explanations:
        unique_key = (
            item["triple"]["entity"],
            item["triple"]["attribute"],
            item["triple"]["value"],
            item["explanation"]
        )
        if unique_key not in seen_triples:
            seen_triples.add(unique_key)
            final_clean_list.append(item)
                
    return final_clean_list


def process_and_clean_triples(raw_result: str, schema: str, case_dialogue: str, file_name: str = "") -> List[Dict[str, Any]]:
    """Parses, filters, and combines model triples with manual message triples."""
    triples = parse_triples(raw_result)
    allowed_relations = extract_allowed_relations(schema)
    filtered_triples = filter_triples(triples, allowed_relations)
    
    manual_message_triples = extract_message_triples_manually(case_dialogue, file_name)
    
    all_triples = filtered_triples + manual_message_triples
    return remove_duplicates(all_triples)


def generate_all_explanations(all_triples: List[Dict[str, Any]], case_dialogue: str, qwen: Any) -> List[Dict[str, Any]]:
    """Generates LLM text explanations and HARD DROPS triples where LLM detects no proof."""
    from src.explanation import extract_explanation_for_triple

    results = []
    for item in all_triples:
        triple = item["triple"]
        entity = str(triple.get("entity", "")).strip()
        attribute = str(triple.get("attribute", "")).strip()

        is_message_triple = (
            entity.startswith("Message") or 
            attribute in ["messageHasContent", "sender", "receiver", "timestamp", "refersToEntity", "communicatesWith"]
        )

        if item.get("inferred") or is_message_triple:
            explanation = ""
        else:
            raw_explanation = extract_explanation_for_triple(
                case_dialogue,
                (triple["entity"], triple["attribute"], triple["value"]),
                qwen
            )
            
            if raw_explanation is None:
                explanation = ""
            else:
                explanation = raw_explanation.replace("Triple: ", "").replace("`", "")
                explanation = explanation.split("You can stop now")[0].strip()

        # HARD DROP: If the explanation admits there is no evidence in the text, drop the triple!
        exp_lower = explanation.lower()
        if "does not provide" in exp_lower or "no direct evidence" in exp_lower or "no evidence" in exp_lower:
            continue

        results.append({
            "triple": triple,
            "explanation": explanation
        })
        
    return results
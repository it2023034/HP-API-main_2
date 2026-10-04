import json
import re
from pathlib import Path
from typing import List, Dict, Set, Any
from rdflib import Graph, RDF, RDFS, OWL
from src.utils import is_valid_phone, is_valid_email, remove_duplicates


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

        triples.append({"triple": {"entity": subj, "attribute": attr, "value": obj}})
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


def filter_triples(triples: List[Dict[str, Any]], allowed_relations: Set[str]) -> List[Dict[str, Any]]:
    """Filters extracted triples against allowed ontology relations."""
    filtered = []
    normalized_allowed = {r.strip() for r in allowed_relations}
    primitive_types = {"string", "integer", "int", "double", "float", "boolean", "datetime", "datetimestamp"}

    for triple in triples:
        subj = str(triple["triple"]["entity"]).strip()
        attr = triple["triple"]["attribute"].strip()
        obj = str(triple["triple"]["value"]).strip()

        if re.search(r'^\d+\.\s*\*\*', subj) or len(subj) > 70 or len(obj) > 120:
            continue
        if attr == "type" and obj.lower() in primitive_types:
            continue
        if subj.startswith("+") or (subj.isdigit() and len(subj) >= 8):
            continue

        if attr in normalized_allowed:
            triple["triple"]["attribute"] = attr
            filtered.append(triple)

    return filtered


def canonicalize_attributes_with_llm(triples: List[Dict[str, Any]], attribute_name: str, qwen_model: Any) -> List[Dict[str, Any]]:
    """Canonicalizes paraphrased values dynamically using LLM."""
    if not qwen_model:
        return triples

    raw_values = list({
        str(item["triple"]["value"]).strip().strip('"\'') 
        for item in triples if item.get("triple", {}).get("attribute") == attribute_name
    })

    if len(raw_values) <= 1:
        return triples

    prompt = f"""
TASK: Group paraphrased or near-synonymous values into concise, canonical semantic terms.
INPUT VALUES: {json.dumps(raw_values, ensure_ascii=False)}
OUTPUT JSON ONLY mapping original strings to canonical terms:
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
        print(f"Canonicalization skipped for {attribute_name}: {e}")

    return triples


def filter_final_triples(triples: List[Dict[str, Any]], qwen_model: Any = None) -> List[Dict[str, Any]]:
    """Post-processing validation guardrails."""
    if qwen_model:
        triples = canonicalize_attributes_with_llm(triples, "hasPurpose", qwen_model)

    discovered_iban = None
    for item in triples:
        t = item.get("triple", {})
        if t.get("attribute") == "hasAccountIdentifier":
            val = str(t.get("value", "")).strip()
            if len(val) > 10 and any(char.isdigit() for char in val):
                discovered_iban = val.replace(" ", "")
                break

    clean_list = []
    seen_keys = set()
    invalid_placeholders = {"(not provided)", "not provided", "unknown", "none", "n/a", "0", "0.0", "string", "double", "null"}
    currency_symbols = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY"}

    for item in triples:
        triple = item.get("triple", {})
        subj = str(triple.get("entity", "")).strip().strip('"\'')
        attr = str(triple.get("attribute", "")).strip()
        val = str(triple.get("value", "")).strip().strip('"\'')

        # 1. Ντετερμινιστικός καθαρισμός ταυτολογιών (Self-Loops: Subject == Object)
        if subj.lower() == val.lower() and attr != "communicatesWith":
            continue

        if val.lower() in invalid_placeholders or subj.lower() in invalid_placeholders or attr == "rdf:type":
            continue

        if attr == "involvesAmount":
            numeric_match = re.search(r'\d+(\.\d+)?', val)
            if numeric_match:
                val = numeric_match.group(0)
            else:
                continue
        
        if attr == "denominatedIn":
            val = currency_symbols.get(val, val.upper())

        if attr == "hasPhoneNumber" and not is_valid_phone(val):
            continue

        if attr == "hasEmail" and not is_valid_email(val):
            continue

        if subj == "Account":
            subj = f"Account_{discovered_iban}" if discovered_iban else "User_Account"
        if val == "Account":
            val = f"Account_{discovered_iban}" if discovered_iban else "User_Account"

        val = re.sub(r'\s+', ' ', val).strip()
        item["triple"] = {"entity": subj, "attribute": attr, "value": val}

        dedup_key = (subj.lower(), attr.lower(), val.lower())
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            clean_list.append(item)

    return clean_list


def extract_message_triples_manually(chat_text: str, file_name: str = "") -> List[Dict[str, Any]]:
    """Extracts structural Message triples from chat logs."""
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


def process_and_clean_triples(raw_result: str, schema: str, case_dialogue: str, file_name: str = "") -> List[Dict[str, Any]]:
    """Parses, filters, and combines model triples with manual message triples."""
    triples = parse_triples(raw_result)
    allowed_relations = extract_allowed_relations(schema)
    filtered_triples = filter_triples(triples, allowed_relations)
    
    manual_message_triples = extract_message_triples_manually(case_dialogue, file_name)
    
    all_triples = filtered_triples + manual_message_triples
    return remove_duplicates(all_triples)
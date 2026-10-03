import json
from pathlib import Path

def clean_triples(extract_path: str):
    ner_file = Path(extract_path) / "ner_entities_with_explanations.json"
    triples_file = Path(extract_path) / "triples_explanations.json"

    with open(ner_file, "r", encoding="utf-8") as f:
        ner_data = json.load(f)

    name_to_id = {}
    for item in ner_data:
        triple = item.get("triple", {})
        if triple.get("attribute") == "label":
            entity_id = triple.get("entity")
            name = triple.get("value").strip().lower()
            name_to_id[name] = entity_id

    with open(triples_file, "r", encoding="utf-8") as f:
        triples_data = json.load(f)

    for item in triples_data:
        triple = item.get("triple", {})
        if not triple:
            continue
            
        subj = triple.get("entity", "").strip().lower()
        obj = triple.get("value", "").strip().lower()

        for name, eid in name_to_id.items():
            if name == subj:
                item["triple"]["entity"] = eid
                break
                
        for name, eid in name_to_id.items():
            if name == obj:
                item["triple"]["value"] = eid
                break

    final_clean_path = Path(extract_path) / "final_clean_triples.json"
    with open(final_clean_path, "w", encoding="utf-8") as f:
        json.dump(triples_data, f, ensure_ascii=False, indent=2)
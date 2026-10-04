import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from src.model import qwen
from src.prompts import build_triple_extraction_prompt
from src.counterfactuals import generate_combined_counterfactuals

# ==========================================
# REFACTORED IMPORTS FROM NEW MODULES
# ==========================================
from src.utils import (
    load_file,
    save_json,
    remove_duplicates,
    sort_triples_custom_priority,
)
from src.chunking import (
    create_chunks_by_date_and_window,
    format_chunk_to_text,
)
from src.ontology import (
    ttl_to_metapaths,
    process_and_clean_triples,
    filter_final_triples,
)
from src.ner_entities import (
    run_ner_pipeline,
    assign_global_entity_ids,
    clean_triples,
)
from src.explanation import (
    generate_all_explanations,
)

OUTPUT_DIR = Path("files/output")
CASE_FILE_PATH = Path("files/input/case_merged.csv")


def get_context_text(case_file_path: Path = CASE_FILE_PATH) -> str:
    """Utility to load or recreate the full context text."""
    case_path_str = str(case_file_path)
    if case_path_str.endswith(".csv"):
        df = pd.read_csv(case_path_str)
        messages = df.to_dict(orient="records")
        chunks = create_chunks_by_date_and_window(messages, max_messages_per_chunk=15, overlap=2)
        return "\n".join([format_chunk_to_text(c) for c in chunks])
    return load_file(case_path_str)


# ==========================================
# INDIVIDUAL PIPELINE STEPS
# ==========================================

def step_1_extract_triples(case_file_path: Path = CASE_FILE_PATH, max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
    """Step 1: Extract raw triples from chunks."""
    print("\n--- STEP 1: Triple Extraction ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    schema = ttl_to_metapaths("files/help/GANNDALF-onto.ttl")
    with open(OUTPUT_DIR / "schema_from_ttl.txt", "w", encoding="utf-8") as f:
        f.write(schema)

    example_out = load_file("files/help/example_out.txt")
    example_dialogue = load_file("files/help/example.txt")

    case_path_str = str(case_file_path)
    if case_path_str.endswith(".csv"):
        df = pd.read_csv(case_path_str)
        messages = df.to_dict(orient="records")
    else:
        case_dialogue = load_file(case_path_str)
        messages = [{"Message": case_dialogue}]

    chunks = create_chunks_by_date_and_window(messages, max_messages_per_chunk=15, overlap=2)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    all_extracted_triples = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {idx}/{len(chunks)}...")
        chunk_text = format_chunk_to_text(chunk)
        prompt = build_triple_extraction_prompt(schema, example_dialogue, example_out, chunk_text)
        raw_result = qwen.invoke(prompt)
        chunk_triples = process_and_clean_triples(raw_result, schema, chunk_text, file_name=case_file_path.name)
        all_extracted_triples.extend(chunk_triples)

    unique_triples = remove_duplicates(all_extracted_triples)
    save_json(OUTPUT_DIR / "triples_ttl.json", unique_triples)
    print(f"Step 1 Complete! Saved {len(unique_triples)} triples to triples_ttl.json")
    return unique_triples


def step_2_generate_explanations() -> List[Dict[str, Any]]:
    """Step 2: Generate explanations & filter final domain triples."""
    print("\n--- STEP 2: Explanations & Filtering ---")
    triples_path = OUTPUT_DIR / "triples_ttl.json"
    if not triples_path.exists():
        print("⚠ triples_ttl.json not found! Running Step 1 first...")
        unique_triples = step_1_extract_triples()
    else:
        with open(triples_path, "r", encoding="utf-8") as f:
            unique_triples = json.load(f)

    context_text = get_context_text()
    results = generate_all_explanations(unique_triples, context_text, qwen)
    clean_results = filter_final_triples(results, qwen_model=qwen)

    filtered_explanations = []
    structural_triples = []

    for item in clean_results:
        triple = item.get("triple", {})
        entity = str(triple.get("entity", "")).strip()
        attr = str(triple.get("attribute", "")).strip()
        explanation = str(item.get("explanation", "")).strip()

        is_structural = (
            entity.startswith("Message") or 
            attr in ["messageHasContent", "sender", "receiver", "timestamp", "refersToEntity"]
        )

        if is_structural:
            structural_triples.append({"triple": triple, "explanation": explanation})
        else:
            filtered_explanations.append(item)

    sorted_primary_results = sort_triples_custom_priority(filtered_explanations)
    save_json(OUTPUT_DIR / "triples_explanations.json", sorted_primary_results)
    save_json(OUTPUT_DIR / "message_structure_triples.json", structural_triples)
    print("Step 2 Complete! Saved triples_explanations.json")
    return sorted_primary_results


def step_3_run_ner() -> List[Dict[str, Any]]:
    """Step 3: Dynamic NER Execution & Global IDs Assignment."""
    print("\n--- STEP 3: Dynamic NER & Entity IDs ---")
    triples_path = OUTPUT_DIR / "triples_ttl.json"
    if not triples_path.exists():
        print("⚠ triples_ttl.json not found! Running Step 1 first...")
        unique_triples = step_1_extract_triples()
    else:
        with open(triples_path, "r", encoding="utf-8") as f:
            unique_triples = json.load(f)

    context_text = get_context_text()
    raw_ner_results = run_ner_pipeline(unique_triples, context_text, qwen)
    final_ner_with_ids = assign_global_entity_ids(raw_ner_results)
    save_json(OUTPUT_DIR / "ner_entities_with_explanations.json", final_ner_with_ids)
    print("Step 3 Complete! Saved ner_entities_with_explanations.json")
    return final_ner_with_ids


def step_4_map_ids_and_counterfactuals() -> List[Dict[str, Any]]:
    """Step 4: Map Entity IDs to triples & Generate Counterfactuals."""
    print("\n--- STEP 4: Entity Mapping & Counterfactuals ---")
    
    if not (OUTPUT_DIR / "ner_entities_with_explanations.json").exists():
        print("⚠ NER entities file missing! Running Step 3 first...")
        step_3_run_ner()
        
    if not (OUTPUT_DIR / "triples_explanations.json").exists():
        print("⚠ Domain triples missing! Running Step 2 first...")
        step_2_generate_explanations()

    print("Mapping literal names to Global Entity IDs...")
    clean_triples(str(OUTPUT_DIR))

    clean_triples_path = OUTPUT_DIR / "final_clean_triples.json"
    with open(clean_triples_path, "r", encoding="utf-8") as f:
        mapped_clean_triples = json.load(f)

    context_text = get_context_text()
    print("Generating counterfactual explanations...")
    counterfactual_results = generate_combined_counterfactuals(mapped_clean_triples, context_text, qwen)
    save_json(OUTPUT_DIR / "triples_with_counterfactuals.json", counterfactual_results)
    print("Step 4 Complete! Saved triples_with_counterfactuals.json")
    return counterfactual_results


def run_full_pipeline():
    """Executes all steps sequentially."""
    step_1_extract_triples()
    step_2_generate_explanations()
    step_3_run_ner()
    step_4_map_ids_and_counterfactuals()
    print("\n🎉 Full Pipeline finished successfully!")


# ==========================================
# INTERACTIVE CLI MENU
# ==========================================

def main_menu():
    while True:
        print("\n" + "="*45)
        print("      KNOWLEDGE GRAPH PIPELINE MENU")
        print("="*45)
        print("1. Run Step 1: Extract Raw Triples")
        print("2. Run Step 2: Explanations & Filtering")
        print("3. Run Step 3: Dynamic NER & Assign Entity IDs")
        print("4. Run Step 4: Map IDs & Generate Counterfactuals")
        print("5. Run FULL Pipeline (All Steps 1-4)")
        print("0. Exit")
        print("="*45)

        choice = input("Select an option (0-5): ").strip()

        if choice == "1":
            step_1_extract_triples()
        elif choice == "2":
            step_2_generate_explanations()
        elif choice == "3":
            step_3_run_ner()
        elif choice == "4":
            step_4_map_ids_and_counterfactuals()
        elif choice == "5":
            run_full_pipeline()
        elif choice == "0":
            print("Exiting pipeline menu. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid option. Please enter a number from 0 to 5.")


if __name__ == "__main__":
    main_menu()
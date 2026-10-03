import json
from pathlib import Path
from typing import Optional
import pandas as pd
from src.model import qwen
from src.prompts import build_triple_extraction_prompt
from src.utils import (
    load_file,
    ttl_to_metapaths,
    create_chunks_by_date_and_window,
    format_chunk_to_text,
    process_and_clean_triples,
    remove_duplicates,
    filter_final_triples,
    generate_all_explanations,
    sort_triples_custom_priority
)


def run_pipeline_process(
    extract_path: str = "files/output", 
    case_file_path: Path = Path("files/input/case_merged.csv"),
    max_chunks: Optional[int] = None
):
    """Executes Knowledge Graph extraction pipeline using chunked conversation processing."""
    
    # Ensure output directory exists
    output_dir = Path(extract_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ontology from files/help and export metapaths schema
    schema = ttl_to_metapaths("files/help/GANNDALF-onto.ttl")
    with open(output_dir / "schema_from_ttl.txt", "w", encoding="utf-8") as f:
        f.write(schema)

    # 2. Load prompt examples from files/help
    example_out = load_file("files/help/example_out.txt")
    example_dialogue = load_file("files/help/example.txt")

    # 3. Read input dataset (supports CSV or raw text files)
    case_path_str = str(case_file_path)
    if case_path_str.endswith(".csv"):
        df = pd.read_csv(case_path_str)
        messages = df.to_dict(orient="records")
    else:
        # Fallback for plain text dialogue files
        case_dialogue = load_file(case_path_str)
        messages = [{"Message": case_dialogue}]

    # 4. Create chunks grouped strictly by date boundaries and window limits
    chunks = create_chunks_by_date_and_window(messages, max_messages_per_chunk=15, overlap=2)

    # Slice chunks if max_chunks is explicitly specified (e.g. for debug mode)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        print(f"Running pipeline on first {len(chunks)} chunks (test mode enabled)...")
    else:
        print(f"Running pipeline on ALL {len(chunks)} date-based chunks...")

    all_extracted_triples = []
    processed_texts = []

    # 5. Process each conversation chunk through Qwen 2.5 14B
    for idx, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {idx}/{len(chunks)}...")
        chunk_text = format_chunk_to_text(chunk)
        processed_texts.append(chunk_text)

        prompt = build_triple_extraction_prompt(
            schema,
            example_dialogue,
            example_out,
            chunk_text,
        )
        
        raw_result = qwen.invoke(prompt)

        # Parse and clean triples for current chunk
        chunk_triples = process_and_clean_triples(
            raw_result, 
            schema, 
            chunk_text, 
            file_name=case_file_path.name
        )
        all_extracted_triples.extend(chunk_triples)

    # 6. Deduplicate combined triples across processed chunks & Canonicalize with LLM
    unique_triples = remove_duplicates(all_extracted_triples)
    clean_triples = filter_final_triples(unique_triples, qwen_model=qwen)  # <-- ΕΔΩ ΜΠΗΚΕ ΤΟ qwen_model=qwen

    # Save raw intermediate triples
    with open(output_dir / "triples_ttl.json", "w", encoding="utf-8") as f:
        json.dump(clean_triples, f, ensure_ascii=False, indent=2)

    # 7. Generate natural language explanations over processed context
    context_text = "\n".join(processed_texts) if case_path_str.endswith(".csv") else load_file(case_path_str)
    results = generate_all_explanations(clean_triples, context_text, qwen)

    # Filter into primary explanations and structural triples
    filtered_explanations = []
    structural_triples = []

    for item in results:
        triple = item.get("triple", {})
        attr = str(triple.get("attribute", "")).strip()
        has_exp = bool(str(item.get("explanation", "")).strip())

        if has_exp or attr == "communicatesWith":
            filtered_explanations.append(item)
        else:
            structural_triples.append(item)

    # Sort primary results so entries with explanations appear first
    sorted_primary_results = sort_triples_custom_priority(filtered_explanations)

    # Save high-value triples with explanations
    with open(output_dir / "triples_explanations.json", "w", encoding="utf-8") as f:
        json.dump(sorted_primary_results, f, ensure_ascii=False, indent=2)

    # Save structural message triples separately
    with open(output_dir / "message_structure_triples.json", "w", encoding="utf-8") as f:
        json.dump(structural_triples, f, ensure_ascii=False, indent=2)

    print(f"Pipeline finished successfully! Output saved to {output_dir}")
    return sorted_primary_results


if __name__ == "__main__":
    # Default execution processes the FULL dataset (max_chunks=None)
    run_pipeline_process(max_chunks=None)
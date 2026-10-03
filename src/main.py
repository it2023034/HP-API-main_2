import json
import os
import shutil
import zipfile
from pathlib import Path
from time import sleep

from counterfactuals import generate_combined_counterfactuals
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from model import qwen
from ner_entities import clean_triples
from prompts import build_triple_extraction_prompt
from utils import (
    filter_final_triples,
    generate_all_explanations,
    load_file,
    process_and_clean_triples,
    run_ner_pipeline,
    ttl_to_metapaths,
)

API_TOKEN = os.getenv("GRIPHIA_KMODELC_API_TOKEN")
print(f"[DEBUG] Initializing API_TOKEN: {'Set' if API_TOKEN else 'Not Set'}")

app = FastAPI()

# Base paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / "files"
HELP_DIR = FILES_DIR / "help"
INPUT_DIR = FILES_DIR / "input"
OUTPUT_DIR = FILES_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def kmodelc(job_id: str, extract_path: str):
  """Core pipeline execution function."""
  print(f"[DEBUG] [{job_id}] Starting task processing...")
  sleep(2)

  def identify_files(path_str: str) -> list[Path]:
    path_obj = Path(path_str)
    valid_files = []
    for file_path in path_obj.rglob("*"):
      if file_path.is_file():
        if file_path.name.startswith("._") or file_path.name == ".DS_Store":
          continue
        if file_path.name.lower().startswith("case"):
          valid_files.append(file_path)
    return valid_files

  def initiate_process(files: list[Path]):
    if not files:
      print(f"[WARNING] [{job_id}] No matching case files found to process.")
      return

    case_file_path = files[0]
    print(f"[INFO] [{job_id}] Processing file: {case_file_path.name}")

    # Load Ontology Schema & Example references
    schema = ttl_to_metapaths(str(HELP_DIR / "GANNDALF-onto.ttl"))
    example_out = load_file(str(HELP_DIR / "example_out.txt"))
    example_dialogue = load_file(str(HELP_DIR / "example.txt"))
    case_dialogue = load_file(str(case_file_path))

    # --- Step 1: Named Entity Recognition (NER) Pipeline ---
    print(f"[DEBUG] [{job_id}] Step 1/4: Running NER Pipeline...")
    ner_with_explanations = run_ner_pipeline(case_dialogue, qwen)
    with open(
        Path(extract_path) / "ner_entities_with_explanations.json",
        "w",
        encoding="utf-8",
    ) as f:
      json.dump(ner_with_explanations, f, ensure_ascii=False, indent=2)

    # --- Step 2: Knowledge Graph Triples Extraction ---
    print(f"[DEBUG] [{job_id}] Step 2/4: Extracting triples via LLM...")
    prompt = build_triple_extraction_prompt(
        schema, example_dialogue, example_out, case_dialogue
    )
    result = qwen.invoke(prompt)

    all_triples = process_and_clean_triples(
        result, schema, case_dialogue, file_name=case_file_path.name
    )
    all_triples = filter_final_triples(all_triples)

    with open(
        Path(extract_path) / "triples_ttl.json", "w", encoding="utf-8"
    ) as f:
      json.dump(all_triples, f, ensure_ascii=False, indent=2)

    # --- Step 3: Explanation Generation ---
    print(f"[DEBUG] [{job_id}] Step 3/4: Generating explanations...")
    results = generate_all_explanations(all_triples, case_dialogue, qwen)
    with open(
        Path(extract_path) / "triples_explanations.json", "w", encoding="utf-8"
    ) as f:
      json.dump(results, f, ensure_ascii=False, indent=2)

    # --- Step 4: Counterfactual Generation & Cleanup ---
    print(f"[DEBUG] [{job_id}] Step 4/4: Generating counterfactuals...")
    final_output = generate_combined_counterfactuals(
        results, case_dialogue, qwen
    )
    with open(
        Path(extract_path) / "triples_with_counterfactuals.json",
        "w",
        encoding="utf-8",
    ) as f:
      json.dump(final_output, f, ensure_ascii=False, indent=2)

    clean_triples(extract_path)
    print(f"[INFO] [{job_id}] Pipeline completed successfully.")

  target_files = identify_files(extract_path)
  initiate_process(target_files)


def process_task(
    job_id: str,
    extract_path: str,
    background_tasks: BackgroundTasks,
    callback_url: str,
):
  print(f"[DEBUG] [{job_id}] Scheduling background task. Callback URL: {callback_url}")
  background_tasks.add_task(kmodelc, job_id, extract_path)
  return {"jobId": job_id, "status": "STARTED"}


def verify_token(authorization: str = Header(...)):
  expected = f"Bearer {API_TOKEN}"
  if authorization != expected:
    raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/uploadZip")
async def upload_zip(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    callback_url: str = Form(...),
    file: UploadFile = File(...),
    _: None = Depends(verify_token),
):
  if not file.filename.lower().endswith(".zip"):
    raise HTTPException(status_code=400, detail="Only ZIP files are allowed")

  extract_dir = UPLOAD_DIR / job_id
  extract_dir.mkdir(parents=True, exist_ok=True)

  zip_path = extract_dir / file.filename

  with open(zip_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)

  try:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
      for member in zip_ref.infolist():
        if member.filename.startswith("__MACOSX/"):
          continue

        member_path = extract_dir / member.filename
        if not member_path.resolve().is_relative_to(extract_dir.resolve()):
          raise HTTPException(status_code=400, detail="Unsafe ZIP contents")

        zip_ref.extract(member, extract_dir)
      zip_path.unlink()

  except zipfile.BadZipFile:
    shutil.rmtree(extract_dir, ignore_errors=True)
    raise HTTPException(status_code=400, detail="Invalid ZIP file")

  return process_task(job_id, str(extract_dir), background_tasks, callback_url)


# --- LOCAL CLI EXECUTION ---
if __name__ == "__main__":
  print("=== Running Pipeline Locally (CLI Mode) ===")
  kmodelc(job_id="local_execution", extract_path=str(INPUT_DIR))
  print(f"=== Process Finished! Results saved in: {OUTPUT_DIR} ===")
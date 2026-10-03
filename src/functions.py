import json
from pipeline import main as run_pipeline

def get_results_func():
    run_pipeline()

    with open("files/triples_explanations.json", "r", encoding="utf-8") as f:
        return json.load(f)
from src.prompts import build_minimal_counterfactual_prompt

def generate_combined_counterfactuals(true_results, dialogue_text, qwen):
    valid_inputs = [
        item for item in true_results 
        if item.get("explanation") and item["explanation"].strip() != ""
    ]
    
    combined_results = []

    for item in valid_inputs:
        new_entry = {
            "triple": item["triple"],
            "explanation": item["explanation"]
        }
            
        explanation_prompt = build_minimal_counterfactual_prompt(
            dialogue_text,
            item["triple"], 
            item["explanation"]
        )
        
        # Καλούμε το invoke χωρίς το 'stop' keyword argument
        raw_output = qwen.invoke(explanation_prompt).strip()
        
        # Κρατάμε μόνο την πρώτη γραμμή της απάντησης (ισοδύναμο με stop=["\n"])
        first_line = raw_output.splitlines()[0] if raw_output else ""
        
        # Καθαρίζουμε τυχόν διπλά κενά και εισαγωγικά
        cf_explanation = " ".join(first_line.split()).strip().strip('"').strip("'")
            
        new_entry["counterfactual_explanation"] = cf_explanation
        
        combined_results.append(new_entry)
        
    return combined_results
from prompts import build_minimal_counterfactual_prompt

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
        
        raw_exp = qwen.invoke(
            explanation_prompt, 
            stop=["\n", "Explain", "The task", "Output:", "Note:", "To arrive"]
        ).strip()
        
        cf_explanation = " ".join(raw_exp.split()).strip().strip('"').strip("'")
            
        new_entry["counterfactual_explanation"] = cf_explanation
        
        combined_results.append(new_entry)
        
    return combined_results
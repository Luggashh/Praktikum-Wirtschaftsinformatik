import json
import re
from difflib import SequenceMatcher
from typing import List

# --- Haystack Imports ---
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

# ==========================================
# 1. EVALUATION COMPONENT (Precision/Recall)
# ==========================================
class BPMNEvaluator:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def _is_match(self, str1, str2):
        # Returns True if similarity is above threshold
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() >= self.threshold

    def calculate_metrics(self, ground_truth_list, extracted_list):
        true_positives = []
        false_positives = []
        # Copy ground truth to track what we miss (False Negatives)
        remaining_gt = ground_truth_list[:]

        for extracted in extracted_list:
            match_found = False
            matched_gt = None
            
            # Check if extracted item matches any item in ground truth
            for gt in remaining_gt:
                if self._is_match(extracted, gt):
                    match_found = True
                    matched_gt = gt
                    break
            
            if match_found:
                true_positives.append(extracted)
                remaining_gt.remove(matched_gt) # Don't match the same GT item twice
            else:
                false_positives.append(extracted)

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(remaining_gt)

        # Avoid DivisionByZero errors
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "tp": true_positives,
            "fp": false_positives,
            "fn": remaining_gt
        }

# ==========================================
# 2. DATA SETUP (3 Cases)
# ==========================================
test_cases = [
    {
        "name": "Case 1: Passenger Security",
        "text": """
        First, the Passenger shows boarding pass. Then the Passenger goes to security check.
        The officer decides if the passenger is Suspicious?
        If yes, the Passenger is checked manually.
        Finally, the Passenger goes to the gate.
        """,
        "ground_truth": [
            "Passenger shows boarding pass", "Passenger goes to security check", 
            "Suspicious?", "Passenger is checked manually", "Passenger goes to gate"
        ]
    },
    {
        "name": "Case 2: Order Processing",
        "text": """
        An Order is received. The system Checks stock availability.
        If stock is available?, the order is Shipped.
        If not, an Error Email is sent.
        The process ends when the order is fulfilled.
        """,
        "ground_truth": [
            "Order is received", "Checks stock availability", "stock is available?", 
            "Shipped", "Error Email is sent", "order is fulfilled"
        ]
    },
    {
        "name": "Case 3: Loan Application",
        "text": """
        The client submits a loan application. The bank Assess credit risk.
        Is the risk acceptable? If yes, Approve Loan.
        If no, Reject Loan.
        """,
        "ground_truth": [
            "submits a loan application", "Assess credit risk", 
            "Is the risk acceptable?", "Approve Loan", "Reject Loan"
        ]
    }
]

# ==========================================
# 3. PIPELINE DEFINITION
# ==========================================

# Define the JSON Schema for the LLM
json_schema = json.dumps({
    "tasks": ["task_name_1", "task_name_2"],
    "gateways": ["gateway_name"],
    "events": ["event_name"]
})

# Define the Prompt Template
template_text = """
Extract all **task names**, **gateways** and **events** from the BPMN description provided in the passage: {{passage}}.

Return the extracted elements as a single JSON object that adheres strictly to the following schema:
{{json_schema}}

Only return the JSON. Do not include markdown formatting like ```json.
"""

# Initialize Haystack Components
# We wrap the string in ChatMessage.from_user
prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(template_text)],required_variables=["passage", "json_schema"]
)

# --- FIX APPLIED HERE ---
# 1. URL changed to "http://localhost:11434" (No /api/generate)
# 2. Model changed to "llama3" (Ensure you have run 'ollama pull llama3')
llm = OllamaChatGenerator(
    model="llama3", 
    url="http://localhost:11434",
    generation_kwargs={
        "format": "json" # Forces Ollama to output valid JSON
    }
)

# ==========================================
# 4. RUNNING THE PIPELINE LOOP (DEBUG VERSION)
# ==========================================
evaluator = BPMNEvaluator(threshold=0.7)

print(f"{'TEST CASE':<30} | {'PRECISION':<10} | {'RECALL':<10}")
print("-" * 60)

for case in test_cases:
    # 1. Build Prompt
    res = prompt_builder.run(
        passage=case["text"], 
        json_schema=json_schema
    )
    
    try:
        # --- THE PROBLEMATIC LINE ---
        # We try to run it, but catch errors if it fails
        result = llm.run(messages=res["prompt"])
        response_text = result["replies"][0].content
        
        # 3. Parse JSON
        # Look for JSON content between braces {} to ignore extra text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            data = json.loads(clean_json)
        else:
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
        # Flatten dictionary for evaluation
        extracted_flat = []
        for key in ["tasks", "gateways", "events"]:
            extracted_flat.extend(data.get(key, []))
            
    except Exception as e:
        # --- ERROR HANDLER ---
        print(f"\n[!] CRITICAL ERROR on {case['name']}:")
        print(f"    Reason: {e}")
        print("    Suggestion: Check if Ollama is running and if you pulled the model.")
        print("    Command to fix: 'ollama run llama3'")
        extracted_flat = []

    # 4. Evaluate
    metrics = evaluator.calculate_metrics(case["ground_truth"], extracted_flat)
    
    # 5. Print Row
    print(f"{case['name']:<30} | {metrics['precision']:<10} | {metrics['recall']:<10}")
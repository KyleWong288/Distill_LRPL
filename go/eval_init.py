import json
import os
from tqdm import tqdm
from utils import *


NUM_SAMPLES = 10

# Evaluates the file and writes to repair file
def evaluation(generation_path):
    # Load data 
    with open(generation_path, "r") as file:
        input_data = json.load(file)
    print("USING INPUT DATA:", generation_path)

    # Evaluate generations
    num_correct = [0] * len(input_data)
    results = []
    for i, sample in tqdm(enumerate(input_data)):
        answers, errors = [], []
        for answer in sample["answers"]:
            clean_code = extract_clean_code(answer)
            run_code = sample["prompt"] + "\n" + clean_code + "\n" + sample["test"]
            error = evaluate_answer(run_code)
            answers.append(sample["prompt"] + "\n" + clean_code)
            errors.append(extract_clean_error(error))
            num_correct[i] += (not error)
        results.append({
            "id": sample["id"],
            "name": sample["name"],
            "prompt": sample["prompt"],
            "answers": answers,
            "test": sample["test"],
            "errors": errors
        })

    # Configure eval file
    eval_file = f"./eval/{DATASET}/{MODEL_NAME}/r0.json"
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    print("EVAL FILE:", eval_file)
    
    # Evaluate pass@k and write to eval file
    print(num_correct)
    eval_dict = overall_pass_at_k(num_correct, NUM_SAMPLES)
    with open(eval_file, "w") as file:
        json.dump(eval_dict, file, indent=4)

    # Configure repair file for next round
    repair_file = f"./repair/{DATASET}/{MODEL_NAME}/r0.json"
    os.makedirs(os.path.dirname(repair_file), exist_ok=True)
    print("REPAIR FILE:", repair_file)

    # Write to repair file
    with open(repair_file, "w") as file:
        json.dump(results, file, indent=4)


# Creates the repair file based on the initial generations
if __name__ == "__main__":

    DATASET = "mbxp"
    MODEL_NAME = "mistral-7b"
    generation_path = f"./output/{DATASET}/{MODEL_NAME}/r0.json"

    evaluation(generation_path)

    print("DONE EVALUATING!")
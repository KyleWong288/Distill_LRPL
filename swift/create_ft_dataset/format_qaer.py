import json
import os
import random
from utils_create_ft_dataset import *


SPLIT_RATIO = 0.90


# Creates the finalized text with (Q, A, E, R)
def create_text(sample):
    question = sample["prompt"]
    answer = extract_clean_code(sample["answer"], sample["entry_point"])
    error = extract_clean_error(sample["error"])
    repair = sample["repair"]
    repair = repair.replace("    ", "\t")
    code = question + "\n" + answer
    instruction = "You are given an incorrect swift function and an error message. Explain how to fix the error, and then write an updated swift function with the correct code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{code}
### Error: {error}
### Response:
{repair}
'''
    return res


# Creates SFT Trainer friendly objects
# Input: {id, name, prompt, answer, error, repair, test, entry_point}
# Output: {name, text}
def create_object(sample):
    text = create_text(sample)
    print("---------------------------------------------------------------------------NEW---------------------------------------------------------")
    print(text)
    res = {
        "name": sample["name"],
        "text": text
    }
    return res


def main(generation_file, output_dir):

    random.seed(17)

    # Load generation file
    with open(generation_file, "r") as file:
        generations = json.load(file)
    random.shuffle(generations)

    # Configure output files
    train_file = os.path.join(output_dir, "train.jsonl")
    dev_file = os.path.join(output_dir, "dev.jsonl")
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(dev_file), exist_ok=True)

    # Split data
    split_idx = int(SPLIT_RATIO * len(generations))
    train_data = generations[:split_idx]
    dev_data = generations[split_idx:]

    # Write to output files
    with open(train_file, "w") as file:
        for sample in train_data:
            file.write(json.dumps(create_object(sample)) + "\n")

    with open(dev_file, "w") as file:
        for sample in dev_data:
            file.write(json.dumps(create_object(sample)) + "\n")

    print("TRAIN LEN:", len(train_data))
    print("DEV LEN:", len(dev_data))
    


if __name__ == "__main__":

    MODEL_NAME = "cl-7b-instruct"

    generation_path = f"./data/{MODEL_NAME}/qaer.json"
    output_dir = f"../data/{MODEL_NAME}"

    main(generation_path, output_dir)

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from utils_create_ft_dataset import *


MAX_SAMPLES = 20


# Creates dataset of (Q, A, E, R), where R passes the given tests
def main(data_path, output_file):

    # Load train data
    with open(data_path, "r") as file:
        input_data = json.load(file)
    
    # Configure output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("OUTPUT FILE:", output_file)

    # Load gpt client
    load_dotenv()
    client = OpenAI()

    # Generate with GPT
    output = []
    for sample in tqdm(input_data):
        clean_code = extract_clean_code(sample["prompt"] + sample["answer"])
        clean_error = extract_clean_error(sample["error"])
        prompt = create_prompt_gpt(clean_code, clean_error)
        for i in range(MAX_SAMPLES):
            # Generate response
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an intelligent expert at repairing code."},
                        {"role": "user", "content": prompt} 
                    ],
                    max_tokens=1024,
                    seed=i+1
                )
                repair = response.choices[0].message.content
            except:
                repair = ""

            # Evaluate response
            clean_code = extract_clean_code(repair)
            has_error = evaluate_answer(clean_code, sample["test"], sample["entry_point"], TEST_DIR)
            print("-----------------------------------------------------------------------ERROR--------------------------------------------------------------")
            print(has_error)

            if not has_error:
                # print("------------------------------------------------PROMPT------------------------------------------------")
                # print(prompt)
                # print("------------------------------------------------REPAIR------------------------------------------------")
                # print(repair)
                output.append({
                    "id": sample["id"],
                    "name": sample["name"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "error": sample["error"],
                    "repair": repair,
                    "test": sample["test"],
                    "entry_point": sample["entry_point"],
                })
                with open(output_file, 'w') as file:
                    json.dump(output, file, indent=4)
                break


if __name__ == "__main__":

    MODEL_NAME = "mistral-7b"
    TEST_DIR = "test_mistral"
    data_path = f"./data/{MODEL_NAME}/qae.json"
    output_path = f"./data/{MODEL_NAME}/qaer.json"

    main(data_path, output_path)
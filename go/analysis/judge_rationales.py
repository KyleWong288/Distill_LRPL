import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# Judge prompt to get the rationale judgement
def create_judge_prompt(incorrect_code, error, rationale):
    instruction = "You are given an incorrect go function, an error message, and a rationale to fix the error. Classify if the rationale is 'Good' or 'Bad'. If the rationale provides enough detail to fix the code, output 'Good'. Otherwise, output 'Bad'.\n"
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error:
{error}
### Rationale: {rationale}
### Response: '''
    return res


# Given some (Q, A, E, rationale), returns either "Good" or "Bad"
def create_judgement(client, incorrect_code, error, rationale, seed):
    prompt = create_judge_prompt(incorrect_code, error, rationale)
    try:
        answer = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an intelligent expert at coding. You can only say 'Good' or 'Bad'."},
                {"role": "user", "content": prompt} 
            ],
            max_tokens=64,
            seed=seed
        )
        verdict = answer.choices[0].message.content
    except Exception as e:
        print(f"An exception occurred: {e}")
        verdict = ""
    return verdict


# 1 = good, 0 = bad, 0 invalid response (very rare)
def parse_verdict(verdict):
    verdict = verdict.lower()
    return "good" in verdict


def main():

    # Load gpt client
    load_dotenv()
    client = OpenAI()

    datasets = ["humaneval"]
    models = ["cl-7b-instruct"]
    
    for DATASET in datasets:
        for MODEL in models:
            # Load the rationales
            rationale_file = f"./rationales/{DATASET}/{MODEL}/{RUN_NAME}/r1.json"
            with open(rationale_file, "r") as file:
                input_data = json.load(file)
            print("INPUT FILE:", rationale_file)
            
            # Configure output file
            output_file = f"./rationales/{DATASET}/{MODEL}/{RUN_NAME}/r1_judge.json"
            os.makedirs(os.path.dirname(rationale_file), exist_ok=True)
            print("OUTPUT FILE:", output_file)
            
            # For each question, we have 5 samples
            # For each sample, we generate 1 judgements
            output = []
            for idx, question in enumerate(input_data):
                all_judgements = []
                for i in range(NUM_SAMPLES):
                    judgements = []
                    if question["errors"][i]:
                        for j in range(NUM_JUDGEMENTS):
                            verdict = create_judgement(client, question["answers"][i], question["errors"][i], question["rationales"][i], seed=j+1)
                            judgements.append(parse_verdict(verdict))
                    all_judgements.append(sum(judgements))

                output.append({
                    "id": question["id"],
                    "name": question["name"],
                    "prompt": question["prompt"],
                    "rationales": question["rationales"],
                    "r1_correct": question["r1_correct"],
                    "judgements": all_judgements
                })

                with open(output_file, "w") as file:
                    json.dump(output, file, indent=4)


if __name__ == "__main__":

    NUM_SAMPLES = 5
    NUM_JUDGEMENTS = 1
    RUN_NAME = "icl"

    main()

    print("DONE!")
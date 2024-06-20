import argparse
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from utils import *


NUM_SAMPLES = 5


def main():

    # Load gpt client
    load_dotenv()
    client = OpenAI()
    repair_file = f"./repair/{args.dataset}/{args.model_name}/r0.json"

    # Loop for multiple repair rounds
    for round_num in range(1, args.num_rounds+1):

        # Load previous round data
        with open(repair_file, "r") as file:
            input_data = json.load(file)
        print("USING INPUT DATA:", repair_file)
        
        # Configure output file
        output_file = f"./output/{args.dataset}/{args.model_name}/gpt/r{round_num}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print("OUTPUT FILE:", output_file)

        # Generate the repairs
        output = []
        for idx, sample in enumerate(input_data):
            print(f"ON {idx} OF {len(input_data)}")
            answers = []
            for i in range(NUM_SAMPLES):
                if not sample["errors"][i]:
                    answers.append(sample["answers"][i])
                else:
                    # Generate code
                    try:
                        prompt = create_repair_prompt_oneshot(sample["answers"][i], sample["errors"][i])
                        answer = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an intelligent expert at repairing code."},
                                {"role": "user", "content": prompt} 
                            ],
                            max_tokens=1024,
                            seed=i+1
                        )
                        answer = answer.choices[0].message.content
                    except:
                        answer = ""
                    answers.append(answer)

            output.append({
                "id": sample["id"],
                "name": sample["name"],
                "prompt": sample["prompt"],
                "answers": answers,
                "test": sample["test"],
            })

            with open(output_file, 'w') as file:
                json.dump(output, file, indent=4)

        # Evaluate the new results
        eval_file = f"./eval/{args.dataset}/{args.model_name}/gpt/r{round_num}.json"
        repair_file = f"./repair/{args.dataset}/{args.model_name}/gpt/r{round_num}.json"
        evaluation_repair(output_file, eval_file, repair_file, NUM_SAMPLES, suffix="gpt2")


# Generates the repair with GPT
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="See DSC_MODEL_DICT for choices")
    parser.add_argument("--dataset", type=str, required=True, help="Testing dataset, either mbxp or humaneval")
    parser.add_argument("--num_rounds", type=int, required=True, help="The number of repair rounds")
    args = parser.parse_args()

    main()

    print("DONE GENERATING!")
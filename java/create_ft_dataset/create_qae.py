import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
from utils_create_ft_dataset import *


MAX_SAMPLES = 10


def load_pretrained_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        DSC_MODEL_DICT[args.model_name],
        device_map = "auto",
        trust_remote_code = True,
        torch_dtype = torch.bfloat16
    )
    print(f"Base Model {args.model_name} loaded")
    return model


# Returns one (A,E), or returns None if all generated answers were correct after MAX_SAMPLES samples
def predict(model, tokenizer, gen_config, sample):
    prompt = fix_prompt(sample["prompt"])
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    input_ids = input["input_ids"]
    raw_text_len = len(input_ids[0])
    
    for _ in range(MAX_SAMPLES):
        # Safeguard against no code being generated
        answer = ""
        while len(answer) < 20:
            with torch.no_grad():
                output = model.generate(
                    **input,
                    generation_config=gen_config,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                answer = tokenizer.decode(output[0][raw_text_len:], skip_special_tokens=True)
                while answer.startswith("\n"):
                    answer = answer[1:]

        # Run the generated code and see if error
        clean_code = extract_clean_code(prompt + answer)
        error = evaluate_answer(clean_code, sample["test"], sample["entry_point"], args.test_dir)
        if error:
            print("-----------------------------------------------NEW-------------------------------------------------")
            print(prompt)
            print(answer)
            print(error)
            return answer, error

    return None, None


# Generates the dataset of (Q, A, E)
def main(args):

    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.cuda.empty_cache()
    set_random_seed(17)

    # Load model and set up gen config
    model = load_pretrained_model(args)
    generation_config = GenerationConfig(
        do_sample = True,
        temperature = args.temperature,
        top_p = args.top_p,
        max_new_tokens = args.max_new_tokens
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DSC_MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_data = [json.loads(line) for line in open(args.data_path).readlines()]
    print("USING TRAIN DATA:", args.data_path)

    # Configure output file
    output_file = f"./data/{args.model_name}/qae.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("OUTPUT FILE:", output_file)
    
    # Generate
    output = []
    for _, sample in tqdm(enumerate(train_data)):
        answer, error = predict(model, tokenizer, generation_config, sample)
        if answer and error:
            output.append({
                "id": sample["id"],
                "name": sample["name"],
                "prompt": fix_prompt(sample["prompt"]),
                "answer": answer,
                "error": error,
                "test": sample["test"],
                "entry_point": sample["entry_point"]
            })
            with open(output_file, 'w') as file:
                json.dump(output, file, indent=4)

    print("DATASET SIZE:", len(output))


# Given Q, generates A, evaluates to get E, saves (Q,A,E) object
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='cl-7b-instruct')
    parser.add_argument('--data_path', type=str, default="./data/train.jsonl")
    parser.add_argument('--test_dir', type=str, default="./test")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling softmax temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--max_new_tokens", type=int, default=800, help="The maximum number of tokens to generate")
    parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision", default=True)
    args = parser.parse_args()

    main(args)

    print("DONE GENERATING!")
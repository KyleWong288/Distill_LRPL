import argparse
import json
import os
import torch
from transformers import AutoTokenizer, GenerationConfig
from utils import *


NUM_SAMPLES = 10


# Gets NUM_SAMPLES generations, handles input encoding and output decoding
def predict(model, tokenizer, gen_config, prompt):
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    input_ids = input["input_ids"]
    raw_text_len = len(input_ids[0])
    
    generations = []
    for _ in range(NUM_SAMPLES):
        res = ""
        while len(res) < 1:
            with torch.no_grad():
                output = model.generate(
                    **input,
                    generation_config=gen_config,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                res = tokenizer.decode(output[0][raw_text_len:], skip_special_tokens=True)
                while res.startswith("\n"):
                    res = res[1:]
        generations.append(res)
    
    print("----------------------------------------------------------------------------NEW----------------------------------------------------------------")
    print(prompt)
    print(generations[0])
    return generations


def main():

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

    # Load test data and fix prompts
    test_data = [json.loads(line) for line in open(args.data_path).readlines()]
    print("USING INPUT FILE:", args.data_path)

    # Configure output file
    output_file = f"./output/{args.dataset}/{args.model_name}/r0.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("USING OUTPUT FILE:", output_file)
    
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    set_random_seed(17)

    # Generate output
    output = []
    for idx, sample in enumerate(test_data):
        print(f"ON {idx} OF {len(test_data)}")
        result_obj = {
            "id": sample["id"],
            "name": sample["name"],
            "prompt": sample["prompt"],
            "answers": None,
            "test": sample["test"],
            "entry_point": sample["entry_point"]
        }
        answers = predict(model, tokenizer, generation_config, sample["prompt"])
        result_obj["answers"] = answers
        output.append(result_obj)

        with open(output_file, 'w') as file:
            json.dump(output, file, indent=4)
  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="See DSC_MODEL_DICT for choices")
    parser.add_argument("--dataset", type=str, required=True, help="Testing dataset, either mbxp or humaneval")
    parser.add_argument('--data_path', type=str, required=True, help="Path to test file")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling softmax temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="The maximum number of tokens to generate")
    parser.add_argument("--load_in_8bit", action='store_true', default=True, help="load the model in 8 bits precision")
    args = parser.parse_args()

    main()

    print("DONE GENERATING!")
import json
import numpy as np
import os
import random
import re
import subprocess
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM


DSC_MODEL_DICT = {
    "cl-7b-instruct": "codellama/CodeLlama-7b-Instruct-hf",
    "cl-7b-base": "codellama/CodeLlama-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}


# Few shot strings
fs_codes = [
    "package main\n\nimport (\n\t\"encoding/json\"\n\t\"reflect\"\n)\n// Code should be written in Go/Golang\n// Write a golang function to check whether the given string is starting with a vowel or not\n// Examples:\n// >>> check_str(\"annie\")\n// >>> 'Valid'\n// >>> check_str(\"dawood\")\n// >>> 'Invalid'\n// >>> check_str(\"Else\")\n// >>> 'Valid'\nfunc check_str (string0 string) string {\n\n\tif string0[0] == 'a' || string0[0] == 'e' || string0[0] == 'i' || string0[0] == 'o' || string0[0] == 'u' {\n\t\treturn \"Valid\"\n\t} else {\n\t\treturn \"Invalid\"\n\t}\n}\n",
    "package main\n\nimport (\n\t\"math\"\n\t\"encoding/json\"\n\t\"reflect\"\n)\n// Code should be written in Go/Golang\n// Write a golang function to find the perimeter of a pentagon.\n// Examples:\n// >>> perimeter_pentagon(5)\n// >>> 25\n// >>> perimeter_pentagon(10)\n// >>> 50\n// >>> perimeter_pentagon(15)\n// >>> 75\nfunc perimeter_pentagon (a int) int {\n\treturn a * 5\n}\n\n",
    "package main\n\nimport (\n\t\"encoding/json\"\n\t\"reflect\"\n)\n// Code should be written in Go/Golang\n// Write a golang function to find the volume of a triangular prism.\n// Examples:\n// >>> find_Volume(10,8,6)\n// >>> 240\n// >>> find_Volume(3,2,2)\n// >>> 6\n// >>> find_Volume(1,2,1)\n// >>> 1\nfunc find_Volume (l int, b int, h int) float64 {\n\n\treturn float64(l*b*h)\n}\n\n",
]
fs_errors = [
    "panic: Exception --- test case 2 failed to pass\n",
    "# command-line-arguments\n./temp.go:4:2: imported and not used: \"math\"\n",
    "panic: Exception --- test case 0 failed to pass\n",
]
fs_repairs = [
    "```go\npackage main\n\nimport (\n\t\"encoding/json\"\n\t\"reflect\"\n\t\"strings\"\n)\n// Code should be written in Go/Golang\n// Write a golang function to check whether the given string is starting with a vowel or not.\n// Examples:\n// >>> check_str(\"annie\")\n// >>> 'Valid'\n// >>> check_str(\"dawood\")\n// >>> 'Invalid'\n// >>> check_str(\"Else\")\n// >>> 'Valid'\nfunc check_str (string0 string) string {\n\tstring0 = strings.ToLower(string0)\n\tif string0[0] == 'a' || string0[0] == 'e' || string0[0] == 'i' || string0[0] == 'o' || string0[0] == 'u' {\n\t\treturn \"Valid\"\n\t} else {\n\t\treturn \"Invalid\"\n\t}\n}\n```",
    "```go\npackage main\n\nimport (\n\t\"math\"\n\t\"encoding/json\"\n\t\"reflect\"\n)\n// Code should be written in Go/Golang\n// Write a golang function to find the perimeter of a pentagon.\n// Examples:\n// >>> perimeter_pentagon(5)\n// >>> 25\n// >>> perimeter_pentagon(10)\n// >>> 50\n// >>> perimeter_pentagon(15)\n// >>> 75\nfunc perimeter_pentagon (a int) int {\n\treturn a * 5\n}\n```",
    "```go\npackage main\n\nimport (\n\t\"encoding/json\"\n\t\"reflect\"\n)\n// Code should be written in Go/Golang\n// Write a golang function to find the volume of a triangular prism.\n// Examples:\n// >>> find_Volume(10,8,6)\n// >>> 240\n// >>> find_Volume(3,2,2)\n// >>> 6\n// >>> find_Volume(1,2,1)\n// >>> 1\nfunc find_Volume (l int, b int, h int) float64 {\n\n\treturn float64(l*b*h) * 0.5\n}\n```",
]
fs_cot = [
    "Test case 2 has input \"Else\", and expects output \'Valid\'. The incorrect function is wrong because it does not check for uppercase vowels. To fix this, the correct function should convert the input to lowercase, and then check for vowels.\n",
    "The incorrect function imports \"math\", but doesn't use it. To fix this, we just need to remove \"math\" from our import statement.\n",
    "Test case 0 is expecting 240 as output. The incorrect function is using l*b*h to calculate volume, but the correct formula is 0.5*l*b*h.\n",
]


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def load_pretrained_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        DSC_MODEL_DICT[args.model_name],
        device_map = "auto",
        trust_remote_code = True,
        torch_dtype = torch.bfloat16
    )
    print(f"Base Model {args.model_name} loaded")
    return model


def load_finetuned_model(args):
    base_model = load_pretrained_model(args)
    ft_model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    print(f"Finetuned Model {args.checkpoint_path} loaded")
    return ft_model


# Takes a list of # of correct code for each problem, returns pass@k metrics
def overall_pass_at_k(num_correct, NUM_SAMPLES):

    def estimate_pass_at_k(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    alls = []
    means = []
    for i in range(1, NUM_SAMPLES+1):
        pass_all = [round(estimate_pass_at_k(NUM_SAMPLES, c, i), 8) for c in num_correct]
        alls.append(pass_all)
        means.append(np.array(pass_all).mean())

    # Add means
    res = {}
    for i in range(NUM_SAMPLES):
        key = f"pass@{i+1}"
        res[key] = means[i]

    # Add alls
    for i in range(NUM_SAMPLES):
        key = f"details_{i+1}"
        res[key] = {k: v for k, v in enumerate(alls[i])},

    return res


# Extracts runnable code from raw generation
# Format 1: code enclosed in ``` ```, used by ft model and gpt
# Format 2: code not enclosed in ``` ```, used by base and icl
def extract_clean_code(answer):
    # Remove unnecessary comments
    answer = answer.replace("// Write your code here", "")
    answer = answer.replace("// Your code goes here", "")
    answer = answer.replace("// your code goes here", "")
    # Cleaning format 1
    pattern = re.compile(r'```(.*?)```', flags=re.DOTALL)
    matches = re.findall(pattern, answer)
    if matches:
        answer = matches[0]
        if answer.startswith("go\n"):
            answer = answer[len("go\n"):]
        # Removes any explanation after
        pattern = re.compile(r'^(.*?)\n[A-Z]', flags=re.DOTALL)
        matches = re.findall(pattern, answer)
        if matches:
            answer = matches[0]
    # Cleaning format 2
    code_begin = max(answer.find("\nfunc"), 0)
    stops = ["\n//", "func main", "###"]
    for stop in stops:
        idx = answer.find(stop, code_begin)
        if idx != -1:
            answer = answer[:idx]
    return answer


# Extracts useful error from raw error
def extract_clean_error(error):
    if not error:
        return None
    stops = ["\ngoroutine"]
    for stop in stops:
        idx = error.find(stop)
        if idx != -1:
            error = error[:idx]
    return error


# Runs one generated answer, returns error message or None
def evaluate_answer(run_code, suffix=""):
    temp_file = f"temp{suffix}.go"
    with open(temp_file, "w", encoding="utf-8") as file:
        file.write(run_code)
    process = subprocess.Popen(["go", "run", temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=5)
        error = stderr.decode()
        if process.returncode == 0 and not error:
            return None
    except:
        process.kill()
        error = "ERROR: Timeout occurred\n"
    return error


# Evaluates all samples in the file and writes to repair
def evaluation_repair(generation_file, eval_file, repair_file, num_samples, suffix=""):
    # Load data
    with open(generation_file, "r") as file:
        input_data = json.load(file)
    print("USING INPUT DATA:", generation_file)

    # Evaluate generations
    num_correct = [0] * len(input_data)
    results = []
    for i, sample in tqdm(enumerate(input_data)):
        clean_codes, errors = [], []
        for answer in sample["answers"]:
            clean_code = extract_clean_code(answer)
            run_code = clean_code + "\n" + sample["test"]
            error = evaluate_answer(run_code, suffix=suffix)
            clean_codes.append(clean_code)
            errors.append(extract_clean_error(error))
            num_correct[i] += (not error)
        sample["answers"] = clean_codes
        sample["errors"] = errors
        results.append(sample)

    # Evaluate pass@k
    print(num_correct)
    eval_dict = overall_pass_at_k(num_correct, num_samples)

    # Write to eval file
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    print("EVAL FILE:", eval_file)
    with open(eval_file, "w") as file:
        json.dump(eval_dict, file, indent=4)

    # # Write to repair file
    os.makedirs(os.path.dirname(repair_file), exist_ok=True)
    print("REPAIR FILE:", repair_file)
    with open(repair_file, "w") as file:
        json.dump(results, file, indent=4)


# Used by fine-tune model
def create_repair_prompt_finetune(incorrect_code, error):
    instruction = "You are given an incorrect golang function and an error message. Explain how to fix the error, and then write an updated golang function with the correct code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error: {error}
### Response:
'''
    return res


# Used by base model and GPT so the formatting is correct
# Empirically, one shot is enough for aligning base models with the correct format
def create_repair_prompt_oneshot(incorrect_code, error):
    instruction = "You are given an incorrect golang function and an error message. Explain how to fix the error, and then write an updated function with the correct code. Do not remove any of the import statements."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{fs_codes[0]}
### Error: {fs_errors[0]}
### Response: {fs_cot[0] + fs_repairs[0]}

### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error: {error}
### Response: '''
    return res


# Used by ICL to create the prompt that only gets a rationale 
def create_rationale_prompt(incorrect_code, error):
    instruction = "You are given an incorrect golang function and an error message. Explain how to modify the code to fix the error, but do not write any actual code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error: {error}
### Response: '''
    return res
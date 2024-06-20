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
HEADER = "#!/usr/bin/perl"


# Few shot strings
fs_codes = [
    "\n# You are an expert Perl programmer, and here is your task.\n# # Write a Perl function to toggle only first and last bits of a given number.\n#\n# >>> toggle_f_and_l_bits(10)\n# 3\n# >>> toggle_f_and_l_bits(15)\n# 6\n# >>> toggle_f_and_l_bits(20)\n# 5\nsub toggle_f_and_l_bits\n{\n  my ($n) = @_;\n return ($n ^ 1) ^ ($n ^ 8);\n}\n",
    "\n# You are an expert Perl programmer, and here is your task.\n# # Write a perl function to find the division of first even and odd number of a given list.\n#\n# >>> div_even_odd([1, 3, 5, 7, 4, 1, 6, 8])\n# 4\n# >>> div_even_odd([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])\n# 2\n# >>> div_even_odd([1, 5, 7, 9, 10])\n# 10\nsub div_even_odd\n{\n  my ($list1) = @_;\n my $even = 0;\n  my $odd = 0;\n  foreach my $i (@$list1)\n  {\n    if ($i % 2 == 0)\n    {\n      $even += $i;\n    }\n    else\n    {\n      $odd += $i;\n    }\n  }\n  return $even / $odd;\n}\n",
    "\n# You are an expert Perl programmer, and here is your task.\n# # Write a Perl function to find the smallest missing number from the given array.\n#\n# >>> find_first_missing([0, 1, 2, 3], 0, 3)\n# 4\n# >>> find_first_missing([0, 1, 2, 6, 9], 0, 4)\n# 3\n# >>> find_first_missing([2, 3, 5, 8, 9], 0, 4)\n# 0\nsub find_first_missing\n{\n  my ($array, $start, $end) = @_;\n my $i = $start;\n  while ($i <= $end)\n  {\n    if (!defined($array->[$i]))\n    {\n      return $i;\n    }\n    $i++;\n  }\n  return $i;\n}\n"
]
fs_errors = [
    "Exception -- test case 0 did not pass. at temp.pl line 27.\n",
    "Exception -- test case 0 did not pass. at temp.pl line 40.\n",
    "Exception -- test case 1 did not pass. at temp.pl line 46.\n",
]
fs_repairs = [
    "```perl\n# You are an expert Perl programmer, and here is your task.\n# # Write a Perl function to toggle only first and last bits of a given number.\n#\n# >>> toggle_f_and_l_bits(10)\n# 3\n# >>> toggle_f_and_l_bits(15)\n# 6\n# >>> toggle_f_and_l_bits(20)\n# 5\nsub toggle_f_and_l_bits\n{\n  my ($n) = @_;\n  my $num_bits = length(sprintf(\"%b\", $n));\n  my $mask = (1 << ($num_bits - 1)) | 1;\n  return $n ^ $mask;\n}\n```",
    "```perl\n# You are an expert Perl programmer, and here is your task.\n# # Write a perl function to find the division of first even and odd number of a given list.\n#\n# >>> div_even_odd([1, 3, 5, 7, 4, 1, 6, 8])\n# 4\n# >>> div_even_odd([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])\n# 2\n# >>> div_even_odd([1, 5, 7, 9, 10])\n# 10\nsub div_even_odd {\n    my ($list) = @_;\n    \n    my ($even, $odd);\n    my $found_even = 0;\n    my $found_odd = 0;\n    \n    foreach my $num (@$list) {\n        if ($num % 2 == 0 && !$found_even) {\n            $even = $num;\n            $found_even = 1;\n        } elsif ($num % 2 != 0 && !$found_odd) {\n            $odd = $num;\n            $found_odd = 1;\n        }\n        last if $found_even && $found_odd;\n    }\n    \n    return $even / $odd;\n}\n```",
    "```perl\n# You are an expert Perl programmer, and here is your task.\n# # Write a Perl function to find the smallest missing number from the given array.\n#\n# >>> find_first_missing([0, 1, 2, 3], 0, 3)\n# 4\n# >>> find_first_missing([0, 1, 2, 6, 9], 0, 4)\n# 3\n# >>> find_first_missing([2, 3, 5, 8, 9], 0, 4)\n# 0\nsub find_first_missing {\n    my ($array, $start, $end) = @_;\n    for my $i ($start .. $end) {\n        return $i unless grep { $_ == $i } @$array;\n    }\n    return $end + 1;\n}\n```"
]
fs_cot = [
    "The incorrect function is trying to use bitwise xor to toggle bits, but it is flipping the wrong bits. To fix this, the correct function should create a bitmask with the least significant bit set and most significant bit set. Then, it should return the bitwise xor between the input and the mask.\n",
    "The error states we are failing the first test case. This is because the incorrect function is tracking a sum of even and odd numbers instead of using the first even and odd number. The correct function can use two booleans to track if we've found an even and odd number.\n",
    "The incorrect code is iterating through each value from start to end, and checking if that index is defined. However, the correct function should be checking if that value exists in the entire array. We can use grep to see if the given array contains a certain value.\n"
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
# Format 2: code not enclosed in ``` ```, used by base
def extract_clean_code(answer):
    # Remove unnecessary comments
    answer = answer.replace("# Write your code here", "")
    answer = answer.replace("# your code goes here", "")
    answer = answer.replace("# Your code goes here", "")
    # Cleaning format 1
    pattern = re.compile(r'```(.*?)```', flags=re.DOTALL)
    matches = re.findall(pattern, answer)
    if matches:
        answer = matches[0]
        if answer.startswith("perl\n"):
            answer = answer[len("perl\n"):]
        # Removes any explanation after
        pattern = re.compile(r'^(.*?)\n[A-Z]', flags=re.DOTALL)
        matches = re.findall(pattern, answer)
        if matches:
            answer = matches[0]
    # Cleaning format 2
    code_begin = max(answer.find("\nsub"), 0) + 4
    stops = ["\n#", "\nprint", "\n\n\n"]
    for stop in stops:
        idx = answer.find(stop, code_begin)
        if idx != -1:
            answer = answer[:idx]
    return answer


# Runs one generated answer, returns error message or None
def evaluate_answer(run_code, suffix=""):
    temp_file = f"temp{suffix}.pl"
    with open(temp_file, "w", encoding="utf-8") as file:
        file.write(run_code)
    process = subprocess.Popen(["perl", temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            errors.append(error)
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
    instruction = "You are given an incorrect perl function and an error message. Explain how to fix the error, and then write an updated perl function with the correct code."
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
    instruction = "You are given an incorrect perl function and an error message. Explain how to fix the error, and then write an updated perl function with the correct code."
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
    instruction = "You are given an incorrect perl function and an error message. Explain how to modify the code to fix the error, but do not write any actual code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error: {error}
### Response: '''
    return res
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
    "/**\n * * Write a javascript function to remove lowercase substrings from a given string by using regex.\n *\n * > removeLowercase(\"KDeoALOklOOHserfLoAJSIskdsf\")\n * \"KDALOOOHLAJSI\"\n * > removeLowercase(\"ProducTnamEstreAmIngMediAplAYer\")\n * \"PTEAIMAAY\"\n * > removeLowercase(\"maNufacTuredbYSheZenTechNolOGIes\")\n * \"NTYSZTNOGI\"\n */\nfunction removeLowercase(str1) {\n return str1.replace(/\\b[a-z]+\\b/g, \"\");\n}\n",
    "/**\n * * Write a JavaScript function to get the position of rightmost set bit.\n *\n * > getFirstSetBitPos(12)\n * 3\n * > getFirstSetBitPos(18)\n * 2\n * > getFirstSetBitPos(16)\n * 5\n */\nfunction getFirstSetBitPos(n) {\n if (n === 0) return -1;\n  return 32 - Math.floor(Math.log2(n));\n}\n",
    "/**\n * * Write a javascript function to abbreviate 'road' as 'rd.' in a given string.\n *\n * > roadRd(\"ravipadu Road\")\n * \"ravipadu Rd.\"\n * > roadRd(\"palnadu Road\")\n * \"palnadu Rd.\"\n * > roadRd(\"eshwar enclave Road\")\n * \"eshwar enclave Rd.\"\n */\nfunction roadRd(street) {\n return street.replace(/road/gi, \"rd.\");\n}\n",
]
fs_errors = [
    "\n/mnt/edward/data2/knw/lrpl/javascript/create_ft_dataset/temp1.js:27\n    throw 'Error at 1th assert statement. Value = ' + JSON.stringify(x0)  \n    ^\nError at 1th assert statement. Value = \"KDeoALOklOOHserfLoAJSIskdsf\"\n(Use `node --trace-uncaught ...` to show where the exception was thrown)\n",
    "\n/mnt/edward/data2/knw/lrpl/javascript/create_ft_dataset/temp.js:28\n    throw 'Error at 1th assert statement. Value = ' + JSON.stringify(x0)  \n    ^\nError at 1th assert statement. Value = 29\n(Use `node --trace-uncaught ...` to show where the exception was thrown)\n",
    "\n/mnt/edward/data2/knw/lrpl/javascript/create_ft_dataset/temp.js:27\n    throw 'Error at 1th assert statement. Value = ' + JSON.stringify(x0)  \n    ^\nError at 1th assert statement. Value = \"ravipadu rd.\"\n(Use `node --trace-uncaught ...` to show where the exception was thrown)\n",
]
fs_repairs = [
    '```javascript\n/**\n * * Write a javascript function to remove lowercase substrings from a given string by using regex.\n *\n * > removeLowercase("KDeoALOklOOHserfLoAJSIskdsf")\n * "KDALOOOHLAJSI"\n * > removeLowercase("ProducTnamEstreAmIngMediAplAYer")\n * "PTEAIMAAY"\n * > removeLowercase("maNufacTuredbYSheZenTechNolOGIes")\n * "NTYSZTNOGI"\n */\nfunction removeLowercase(str1) {\n    return str1.replace(/[a-z]+/g, \'\');\n}\n```',
    "```javascript\n/**\n * * Write a JavaScript function to get the position of rightmost set bit.\n *\n * > getFirstSetBitPos(12)\n * 3\n * > getFirstSetBitPos(18)\n * 2\n * > getFirstSetBitPos(16)\n * 5\n */\nfunction getFirstSetBitPos(n) {\n    if (n === 0) return 0;\n    const rightmostSetBit = n & -n;\n    const position = Math.log2(rightmostSetBit) + 1;\n    return position;\n}\n```",
    '```javascript\n/**\n * * Write a javascript function to abbreviate \'road\' as \'rd.\' in a given string.\n *\n * > roadRd("ravipadu Road")\n * "ravipadu Rd."\n * > roadRd("palnadu Road")\n * "palnadu Rd."\n * > roadRd("eshwar enclave Road")\n * "eshwar enclave Rd."\n */\nfunction roadRd(street) {\n    return street.replace(/\\bRoad\\b/g, "Rd.");\n}\n```',
]
fs_cot = [
    "The incorrect code is using the word boundary 'b' delimiter, which matches complete words that are lowercase. However, the function description wants to match substrings that are lowercase, which may not be full words. To fix this, the correct code should update the regex to search for general substrings.\n",
    "The incorrect function is trying to compute the first set bit, but is doing this incorrectly since it floors the result of Math.log2, leading to an off by 1 error. To fix this, the correct function can isolate the first set bit by computing the bitwise AND between n and -n, and then calculate the final result with Math.log2.\n",
    "The error states we are failing the first test case. This is because the incorrect function is replacing 'road' with 'rd.', but the given test cases actually use the uppercase versions of 'Road' and 'Rd.'. The correct function should fix the replacement to handle the uppercase versions.\n",
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
    answer = answer.replace("// Write your code here", "")
    answer = answer.replace("// your code goes here", "")
    answer = answer.replace("// Your code goes here", "")
    # Cleaning format 1
    pattern = re.compile(r'```(.*?)```', flags=re.DOTALL)
    matches = re.findall(pattern, answer)
    if matches:
        answer = matches[0]
        if answer.startswith("javascript\n"):
            answer = answer[len("javascript\n"):]
        # Removes any explanation after
        pattern = re.compile(r'^(.*?)\n[A-Z]', flags=re.DOTALL)
        matches = re.findall(pattern, answer)
        if matches:
            answer = matches[0]
    # Cleaning format 2
    code_begin = max(answer.find("\nfunction"), 0) + 1
    stops = ["\n//", "\nconsole.log", "\n\n\n", "\n/*", "module.exports"]
    for stop in stops:
        idx = answer.find(stop, code_begin)
        if idx != -1:
            answer = answer[:idx]
    return answer


# Extracts useful error from raw error
def extract_clean_error(error):
    if not error:
        return None
    # Prevents infinitely long error messages
    error = error[:1600]
    return error


# Runs one generated answer, returns error message or None
def evaluate_answer(run_code, suffix=""):
    temp_file = f"temp{suffix}.js"
    with open(temp_file, "w", encoding="utf-8") as file:
        file.write(run_code)
    process = subprocess.Popen(["node", temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    instruction = "You are given an incorrect javascript function and an error message. Explain how to fix the error, and then write an updated javascript function with the correct code."
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
    instruction = "You are given an incorrect javascript function and an error message. Explain how to fix the error, and then write an updated javascript function with the correct code."
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
    instruction = "You are given an incorrect javascript function and an error message. Explain how to modify the code to fix the error, but do not write any actual code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error: {error}
### Response: '''
    return res
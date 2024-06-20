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
TEST_HEADER = "import java.io.*;\nimport java.lang.*;\nimport java.util.*;\nimport java.math.*;\n\n"


# Few shot strings
fs_codes = [
    "import java.io.*;\nimport java.lang.*;\nimport java.util.*;\nimport java.math.*;\n\n\nclass ToggleFAndLBits {\n    /**\n     * * Write a Java function to toggle only first and last bits of a given number.\n     *\n     * > toggleFAndLBits(10)\n     * 3\n     * > toggleFAndLBits(15)\n     * 6\n     * > toggleFAndLBits(20)\n     * 5\n     */\n    public static int toggleFAndLBits(int n) {\n       int firstBit = n & 1;\n        int lastBit = n & 16;\n        int mask = 15;\n        int result = n & mask;\n        result = result ^ (firstBit ^ lastBit);\n        return result;\n    }\n\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        int result = toggleFAndLBits(n);\n        System.out.println(result);\n        sc.close();\n    }\n}\n",
    "import java.io.*;\nimport java.lang.*;\nimport java.util.*;\nimport java.math.*;\n\n\nclass RoadRd {\n    /**\n     * * Write a java function to abbreviate 'road' as 'rd.' in a given string.\n     *\n     * > roadRd(\"ravipadu Road\")\n     * \"ravipadu Rd.\"\n     * > roadRd(\"palnadu Road\")\n     * \"palnadu Rd.\"\n     * > roadRd(\"eshwar enclave Road\")\n     * \"eshwar enclave Rd.\"\n     */\n    public static String roadRd(String street) {\n       return street.replaceAll(\"road\", \"rd.\");\n    }\n\n    public static void main(String[] args) {\n        System.out.println(roadRd(\"ravipadu Road\"));\n        System.out.println(roadRd(\"palnadu Road\"));\n        System.out.println(roadRd(\"eshwar enclave Road\"));\n    }\n}\n",
    "import java.io.*;\nimport java.lang.*;\nimport java.util.*;\nimport java.math.*;\n\n\nclass SequentialSearch {\n    /**\n     * * Write a java function to search an element in the given array by using sequential search.\n     *\n     * > sequentialSearch([11, 23, 58, 31, 56, 77, 43, 12, 65, 19], 31)\n     * [true, 3]\n     * > sequentialSearch([12, 32, 45, 62, 35, 47, 44, 61], 61)\n     * [true, 7]\n     * > sequentialSearch([9, 10, 17, 19, 22, 39, 48, 56], 48)\n     * [true, 6]\n     */\n    public static List<Object> sequentialSearch(List<Integer> dlist, int item) {\n       int i = 0;\n        for (Integer element : dlist) {\n            if (element == item) {\n                return Arrays.asList(true, i);\n            }\n            i++;\n        }\n        return Arrays.asList(false, -1);\n    }\n}\n\npublic class SequentialSearchApp {\n    public static void main(String[] args) {\n        List<Integer> dlist = Arrays.asList(11, 23, 58, 31, 56, 77, 43, 12, 65, 19);\n        List<Object> result = SequentialSearch.sequentialSearch(dlist, 31);\n        System.out.println(result);\n    }\n}\n"
]
fs_errors = [
    "Exception in thread \"main\" java.lang.Exception: Exception -- test case 0 did not pass. x0 = 10\n\tat Main.main(Main.java:24)\n",
    "Exception in thread \"main\" java.lang.Exception: Exception -- test case 0 did not pass. x0 = ravipadu Road\n\tat Main.main(Main.java:24)\n",
    "Compilation Error: ./test/SequentialSearch.java:30: error: class SequentialSearchApp is public, should be declared in a file named SequentialSearchApp.java\npublic class SequentialSearchApp {\n       ^\n1 error\n"
]
fs_repairs = [
    "```java\nimport java.io.*;\nimport java.lang.*;\nimport java.util.*;\nimport java.math.*;\n\n\nclass ToggleFAndLBits {\n    /**\n     * * Write a Java function to toggle only first and last bits of a given number.\n     *\n     * > toggleFAndLBits(10)\n     * 3\n     * > toggleFAndLBits(15)\n     * 6\n     * > toggleFAndLBits(20)\n     * 5\n     */\n    public static int toggleFAndLBits(int n) {\n        // Find the position of the most significant bit\n        int msbPos = 0;\n        while ((1 << msbPos) <= n) {\n            msbPos++;\n        }\n        // Toggle the first and last bits\n        return n ^ (1 << (msbPos - 1)) ^ 1;\n    }\n\n    public static void main(String[] args) {\n        System.out.println(toggleFAndLBits(10)); // Output: 3\n        System.out.println(toggleFAndLBits(15)); // Output: 6\n        System.out.println(toggleFAndLBits(20)); // Output: 5\n    }\n}\n```",
    "```java\nimport java.io.*;\nimport java.lang.*;\nimport java.util.*;\nimport java.math.*;\n\n\nclass RoadRd {\n    /**\n     * * Write a java function to abbreviate 'road' as 'rd.' in a given string.\n     *\n     * > roadRd(\"ravipadu Road\")\n     * \"ravipadu Rd.\"\n     * > roadRd(\"palnadu Road\")\n     * \"palnadu Rd.\"\n     * > roadRd(\"eshwar enclave Road\")\n     * \"eshwar enclave Rd.\"\n     */\n    public static String roadRd(String street) {\n       return street.replaceAll(\"Road\", \"Rd.\");\n    }\n\n    public static void main(String[] args) {\n        System.out.println(roadRd(\"ravipadu Road\"));\n        System.out.println(roadRd(\"palnadu Road\"));\n        System.out.println(roadRd(\"eshwar enclave Road\"));\n    }\n}\n```",
    "```java\nimport java.io.*;\nimport java.lang.*;\nimport java.util.*;\nimport java.math.*;\n\n\nclass SequentialSearch {\n    /**\n     * * Write a java function to search an element in the given array by using sequential search.\n     *\n     * > sequentialSearch([11, 23, 58, 31, 56, 77, 43, 12, 65, 19], 31)\n     * [true, 3]\n     * > sequentialSearch([12, 32, 45, 62, 35, 47, 44, 61], 61)\n     * [true, 7]\n     * > sequentialSearch([9, 10, 17, 19, 22, 39, 48, 56], 48)\n     * [true, 6]\n     */\n    public static List<Object> sequentialSearch(List<Integer> dlist, int item) {\n       int i = 0;\n        for (Integer element : dlist) {\n            if (element == item) {\n                return Arrays.asList(true, i);\n            }\n            i++;\n        }\n        return Arrays.asList(false, -1);\n    }\n\n    public static void main(String[] args) {\n        List<Integer> dlist = Arrays.asList(11, 23, 58, 31, 56, 77, 43, 12, 65, 19);\n        List<Object> result = SequentialSearch.sequentialSearch(dlist, 31);\n        System.out.println(result);\n    }\n}\n```"
]
fs_cot = [
    "The incorrect function is trying to use bitwise xor to toggle bits, but it is flipping the wrong bits. To fix this, the correct function should find the position of the most signifcant bit. Then, it should bitwise xor the input number with its most significant bit and the least significant bit.\n",
    "The error states we are failing the first test case. We can see that the comments show the test cases use strings containing \'Road\' and want output containing \'Rd.\'. So to fix this, the correct function should check for the capitalized versions instead.\n",
    "The incorrect function is creating a new class. However, we should keep all the code in the SequentialSearch class. We can fix this by removing the unnecessary SequentialSearchApp class and putting the main function in the original SequentialSearch class.\n"
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
        if answer.startswith("java\n"):
            answer = answer[len("java\n"):]
        # Removes any explanation after
        pattern = re.compile(r'^(.*?)\n[A-Z]', flags=re.DOTALL)
        matches = re.findall(pattern, answer)
        if matches:
            answer = matches[0]
    # Cleaning format 2
    code_begin = max(answer.find("public static"), 0) + 1
    stops = ["\n//", "\n\n\n"]
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


# Removes solution files safely based on entry point name
def safe_remove_files(test_dir, entry_point):
    java_file = f"{test_dir}/{entry_point}.java"
    if os.path.exists(java_file):
        os.remove(java_file)
    # Clean class files
    files = os.listdir(test_dir)
    for name in files:
        if ".class" in name:
            del_name = f"{test_dir}/{name}"
            os.remove(del_name)


# Runs one generated answer, returns error message or None
# Entry point file for the answer, Main file for the test cases 
# File names are prefixed with a test_dir_name
def evaluate_answer(class_code, test_code, entry_point, test_dir):
    
    # Write to files
    os.makedirs(test_dir, exist_ok=True)
    solve_file = f"{test_dir}/{entry_point}.java"
    with open(solve_file, "w", encoding="utf-8") as file:
        file.write(class_code)
    test_file = f"{test_dir}/Main.java"
    with open(test_file, "w", encoding="utf-8") as file:
        file.write(TEST_HEADER + test_code)

    # Compile files
    process = subprocess.Popen(["javac", solve_file, test_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error = stderr.decode()
        safe_remove_files(test_dir, entry_point)
        return "Compilation Error: " + error
    
    # Execute test code
    process = subprocess.Popen(["java", "-cp", test_dir, "Main"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=5)
        error = stderr.decode()
        if process.returncode == 0 and not error:
            safe_remove_files(test_dir, entry_point)
            return None
    except:
        process.kill()
        error = "ERROR: Timeout occurred\n"
    safe_remove_files(test_dir, entry_point)
    return error


# Evaluates all samples in the file and writes to repair
def evaluation_repair(generation_file, eval_file, repair_file, num_samples, test_dir):
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
            error = evaluate_answer(clean_code, sample["test"], sample["entry_point"], test_dir)
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
    instruction = "You are given an incorrect java function and an error message. Explain how to fix the error, and then write an updated java function with the correct code."
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
    instruction = "You are given an incorrect java function and an error message. Explain how to fix the error, and then write an updated java function with the correct code."
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
    instruction = "You are given an incorrect java function and an error message. Explain how to modify the code to fix the error, but do not write any actual code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error: {error}
### Response: '''
    return res
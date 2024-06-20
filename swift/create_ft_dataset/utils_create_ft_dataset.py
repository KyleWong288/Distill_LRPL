import json
import numpy as np
import os
import random
import re
import subprocess
import torch


DSC_MODEL_DICT = {
    "cl-7b-instruct": "codellama/CodeLlama-7b-Instruct-hf",
    "cl-7b-base": "codellama/CodeLlama-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}
HEADER = "import Foundation\n"


# Few shot strings
fs_codes = [
    "/**\n * You are an expert Swift programmer, and here is your task.\n * * Write a swift function to find all pairs in an integer array whose sum is equal to a given number.\n *\n * >>> getPairsCount([1, 5, 7, -1, 5], 5, 6)\n * 3\n * >>> getPairsCount([1, 5, 7, -1], 4, 6)\n * 2\n * >>> getPairsCount([1, 1, 1, 1], 4, 2)\n * 6\n */\nfunc getPairsCount(arr : [Int], n : Int, sum : Int) -> Int {\n   var count = 0\n    for i in 0..<arr.count {\n        for j in i..<arr.count {\n            if arr[i] + arr[j] == sum {\n                count += 1\n            }\n        }\n    }\n    return count\n}\n",
    "/**\n * You are an expert Swift programmer, and here is your task.\n * * Write a swift function to abbreviate 'road' as 'rd.' in a given string.\n *\n * >>> roadRd(\"ravipadu Road\")\n * \"ravipadu Rd.\"\n * >>> roadRd(\"palnadu Road\")\n * \"palnadu Rd.\"\n * >>> roadRd(\"eshwar enclave Road\")\n * \"eshwar enclave Rd.\"\n */\nfunc roadRd(street : String) -> String {\n   var newStreet = street\n    if street.hasSuffix(\"road\") {\n        newStreet = street.replacingOccurrences(of: \"road\", with: \"rd.\")\n    }\n    return newStreet\n}\n",
    "/**\n * You are an expert Swift programmer, and here is your task.\n * * Write a Swift function to get the position of rightmost set bit.\n *\n * >>> getFirstSetBitPos(12)\n * 3\n * >>> getFirstSetBitPos(18)\n * 2\n * >>> getFirstSetBitPos(16)\n * 5\n */\nfunc getFirstSetBitPos(n : Int) -> Int {\n   var pos = 0\n    while n > 0 {\n        n >>= 1\n        pos += 1\n    }\n    return pos\n}\n",
]
fs_errors = [
    "temp/temp.swift:46: Assertion failed: Exception -- test case 2 did not pass. x2 = \n",
    "temp/temp.swift:25: Assertion failed: Exception -- test case 0 did not pass. x0 = \n",
    "temp.swift:16:11: error: left side of mutating operator isn't mutable: 'n' is a 'let' constant\n        n >>= 1\n        ~ ^\n",
]
fs_repairs = [
    "```swift\n/**\n * You are an expert Swift programmer, and here is your task.\n * * Write a swift function to find all pairs in an integer array whose sum is equal to a given number.\n *\n * >>> getPairsCount([1, 5, 7, -1, 5], 5, 6)\n * 3\n * >>> getPairsCount([1, 5, 7, -1], 4, 6)\n * 2\n * >>> getPairsCount([1, 1, 1, 1], 4, 2)\n * 6\n */\nfunc getPairsCount(arr : [Int], n : Int, sum : Int) -> Int {\n   var count = 0\n    for i in 0..<arr.count {\n        for j in (i+1)..<arr.count {\n            if arr[i] + arr[j] == sum {\n                count += 1\n            }\n        }\n    }\n    return count\n}\n```",
    '```swift\n/**\n * You are an expert Swift programmer, and here is your task.\n * * Write a swift function to abbreviate \'road\' as \'rd.\' in a given string.\n *\n * >>> roadRd("ravipadu Road")\n * "ravipadu Rd."\n * >>> roadRd("palnadu Road")\n * "palnadu Rd."\n * >>> roadRd("eshwar enclave Road")\n * "eshwar enclave Rd."\n */\nfunc roadRd(street: String) -> String {\n    let newStreet = street.replacingOccurrences(of: " Road", with: " Rd.", options: .caseInsensitive, range: nil)\n    return newStreet\n}\n```',
    "```swift\n/**\n * You are an expert Swift programmer, and here is your task.\n * * Write a Swift function to get the position of rightmost set bit.\n *\n * >>> getFirstSetBitPos(12)\n * 3\n * >>> getFirstSetBitPos(18)\n * 2\n * >>> getFirstSetBitPos(16)\n * 5\n */\n\nfunc getFirstSetBitPos(n: Int) -> Int {\n    var number = n\n    var pos = 1\n    while number > 0 {\n        if number & 1 == 1 {\n            return pos\n        }\n        number >>= 1\n        pos += 1\n    }\n    return 0\n}\n```",
]
fs_cot = [
    "The incorrect function is trying to loop through all pairs in the nested loops, but it has an indexing issue that affects the counting of pairs. The inner loop should start from i + 1 instead of i to ensure an item is not paired with itself. To fix this, the correct function should start at i + 1 instead.\n",
    "The error states we are failing the first test case. We can see that the comments show the test cases use strings containing \'Road\' and want output containing \'Rd.\'. So to fix this, the correct function should work for both the uppercase and lowercase versions instead.\n",
    "The incorrect code is getting a compile error because we are trying to modify n, but n was created with 'let'. To fix this, we can just create a copy of n using 'var', which creates a mutable variable. The correct code also need to update the logic to get the first set bit on the right side, which can be done by using bitwise AND to iteratively check the last bit.\n"
]


# Few (3) shot prompt
def create_prompt_gpt(incorrect_code, error):
    instruction = "You are given an incorrect swift function and an error message. Explain how to fix the error, and then write an updated swift function with the correct code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{fs_codes[0]}
### Error:
{fs_errors[0]}
### Response: {fs_cot[0] + fs_repairs[0]}

### Instruction: {instruction}
### Incorrect Code:
{fs_codes[1]}
### Error: {fs_errors[1]}
### Response: {fs_cot[1] + fs_repairs[1]}

### Instruction: {instruction}
### Incorrect Code:
{fs_codes[2]}
### Error: {fs_errors[2]}
### Response: {fs_cot[2] + fs_repairs[2]}

### Instruction: {instruction}
### Incorrect Code:
{incorrect_code}
### Error: {error}
### Response:
'''
    return res


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def fix_prompt(prompt):
    prompt = prompt.replace("Write a function", "Write a swift function")
    return prompt


# Extracts runnable code from raw generation
# Format 1: code enclosed in ``` ```, used by ft model and gpt
# Format 2: code not enclosed in ``` ```, used by base
def extract_clean_code(answer, func_name):
    # Remove unnecessary comments
    answer = answer.replace("// Write your code here", "")
    answer = answer.replace("// your code goes here", "")
    answer = answer.replace("// Your code goes here", "")
    # Cleaning format 1
    pattern = re.compile(r'```(.*?)```', flags=re.DOTALL)
    matches = re.findall(pattern, answer)
    if matches:
        answer = matches[0]
        if answer.startswith("swift\n"):
            answer = answer[len("swift\n"):]
        # Removes any explanation after
        pattern = re.compile(r'^(.*?)\n[A-Z]', flags=re.DOTALL)
        matches = re.findall(pattern, answer)
        if matches:
            answer = matches[0]
    # Cleaning format 2
    code_begin = max(answer.find("\nfunc"), 0) + 4
    stops = ["\n//", "\nprint", "\n/*", "\n\n\n", f"\n{func_name}"]
    for stop in stops:
        idx = answer.find(stop, code_begin)
        if idx != -1:
            answer = answer[:idx]
    return answer


# Extracts useful error from raw error
# Ignores docker stack dump
def extract_clean_error(error):
    if not error:
        return None
    # Prevents infinitely error messages
    error = error[:1024]
    stops = ["Stack dump"]
    for stop in stops:
        idx = error.find(stop)
        if idx != -1:
            error = error[:idx]
    return error


# Runs one generated answer, returns error message or None
def evaluate_answer(run_code, test_dir):
    temp_file = f"{test_dir}/temp.swift"
    with open(temp_file, "w", encoding="utf-8") as file:
        file.write(HEADER + run_code)
    # Docker command for executing swift code
    command = [
        "docker", "run", "--rm",
        "-v", f"/data3/knw/lrpl/swift/create_ft_dataset/{test_dir}:/code",
        "-w", "/code",
        "swift", "swift", "temp.swift"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=5)
        error = stderr.decode()
        if process.returncode == 0 and not error:
            return None
    except:
        process.terminate()
        error = "ERROR: Timeout occurred\n"
    return error
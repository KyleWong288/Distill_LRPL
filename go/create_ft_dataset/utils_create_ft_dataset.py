import numpy as np
import random
import re
import torch


# Model dictionary
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


# Few (3) shot prompt
def create_prompt_gpt(prompt, answer, error):
    code = prompt + "\n" + answer
    instruction = "You are given an incorrect golang function and an error message. Explain how to fix the error, and then write an updated golang function with the correct code."
    res = f'''### Instruction: {instruction}
### Incorrect Code:
{fs_codes[0]}
### Error: {fs_errors[0]}
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
{code}
### Error: {error}
### Response:
'''
    return res


def fix_prompt(prompt):
    prompt = prompt.replace("Write a function", "Write a golang function")
    return prompt


# Extracts runnable code from raw generation
# Format 1: code enclosed in ``` ```, used by ft model and gpt
# Format 2: code not enclosed in ``` ```, used by base
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
    stops = ["\ngoroutine"]
    for stop in stops:
        idx = error.find(stop)
        if idx != -1:
            error = error[:idx]
    return error


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
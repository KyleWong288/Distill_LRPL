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
    "\ndef check_String(str):\n\t\"\"\"\n\tWrite a python function to check whether a string has at least one letter and one number.\n\t>>> check_String('thishasboth29')\n\tTrue\n\t>>> check_String('python')\n\tFalse\n\t>>> check_String ('string')\n\tFalse\n\t\"\"\"\n\tif (str.isalnum()):\n\t\treturn True\n\telse:\n\t\treturn False\n",
    "\ndef text_match_three(text):\n\t\"\"\"\n\tWrite a python function that matches a string that has an a followed by three 'b'.\n\t>>> text_match_three(\"ac\")\n\t('Not matched!')\n\t>>> text_match_three(\"dc\")\n\t('Not matched!')\n\t>>> text_match_three(\"abbbba\")\n\t('Found a match!')\n\t\"\"\"\n\n\tif re.search(r\"a[b]{3}\", text):\n\t\treturn \"Found a match!\"\n\telse:\n\t\treturn \"Not matched!\"\n\n",
    "\ndef position_max(list1):\n\t\"\"\"\n\tWrite a python function to find all index positions of the maximum values in a given list.\n\t>>> position_max([12,33,23,10,67,89,45,667,23,12,11,10,54])\n\t[7]\n\t>>> position_max([1,2,2,2,4,4,4,5,5,5,5])\n\t[7,8,9,10]\n\t>>> position_max([2,1,5,6,8,3,4,9,10,11,8,12])\n\t[11]\n\t\"\"\"\n\tlist1 = list(set(list1))\n\tmax_val = max(list1)\n\treturn [i for i,x in enumerate(list1) if x == max_val]\n"
]
fs_errors = [
    "Traceback (most recent call last):\n  File \"temp.py\", line 27, in <module>\n    check(check_String)\n  File \"temp.py\", line 23, in check\n    assert candidate('python') == False\nAssertionError\n",
    "Traceback (most recent call last):\n  File \"temp.py\", line 28, in <module>\n    check(text_match_three)\n  File \"temp.py\", line 23, in check\n    assert candidate(\"ac\")==('Not matched!')\n  File \"temp.py\", line 15, in text_match_three\n    if re.search(r\"a[b]{3}\", text):\nNameError: name 're' is not defined\n",
    "Traceback (most recent call last):\n  File \"temp.py\", line 25, in <module>\n    check(position_max)\n  File \"temp.py\", line 20, in check\n    assert candidate([12,33,23,10,67,89,45,667,23,12,11,10,54])==[7]\nAssertionError\n"
]
fs_repairs = [
    "```python\ndef check_String(str):\n\t\"\"\"\n\tWrite a python function to check whether a string has atleast one letter and one number.\n\t>>> check_String('thishasboth29')\n\tTrue\n\t>>> check_String('python')\n\tFalse\n\t>>> check_String ('string')\n\tFalse\n\t\"\"\"\n\thas_letter = False\n\thas_number = False\n\tfor char in str:\n\t\tif char.isalpha():\n\t\t\thas_letter = True\n\t\telif char.isdigit():\n\t\t\thas_number = True\n\t\tif has_letter and has_number:\n\t\t\treturn True\n\treturn False\n```",
    "```python\nimport re\n\ndef text_match_three(text):\n\t\"\"\"\n\tWrite a python function that matches a string that has an a followed by three 'b'.\n\t>>> text_match_three(\"ac\")\n\t('Not matched!')\n\t>>> text_match_three(\"dc\")\n\t('Not matched!')\n\t>>> text_match_three(\"abbbba\")\n\t('Found a match!')\n\t\"\"\"\n\n\tif re.search(r\"a[b]{3}\", text):\n\t\treturn \"Found a match!\"\n\telse:\n\t\treturn \"Not matched!\"\n\n```",
    "```python\ndef position_max(list1):\n\t\"\"\"\n\tWrite a python function to find all index positions of the maximum values in a given list.\n\t>>> position_max([12,33,23,10,67,89,45,667,23,12,11,10,54])\n\t[7]\n\t>>> position_max([1,2,2,2,4,4,4,5,5,5,5])\n\t[7,8,9,10]\n\t>>> position_max([2,1,5,6,8,3,4,9,10,11,8,12])\n\t[11]\n\t\"\"\"\n\tmax_val = max(list1)\n\treturn [i for i,x in enumerate(list1) if x == max_val]\n```"
]
fs_cot = [
    "The incorrect function is returning True for an input of \"python\", but it should return False, since there is no number. This is because the incorrect code does not check for both a letter and number. To fix this, the correct function can use two booleans to find both a letter and a number.\n",
    "The error states the module 're' is not defined. To fix this, we need to import the 're' module to search for regular expressions.\n",
    "The incorrect code is converting the input list into a set, and then converting the set back into a list. This is wrong because converting to a set can change the order of the original indexes. The correct function just needs to remove the line that converts the list into a set.\n"
]


# Few (3) shot prompt
def create_prompt_gpt(incorrect_code, error):
    instruction = "You are given an incorrect python function and an error message. Explain how to fix the error, and then write an updated python function with the correct code."
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


def fix_prompt(prompt):
    prompt = prompt.replace("Write a function", "Write a python function")
    return prompt


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
        if answer.startswith("python\n"):
            answer = answer[len("python\n"):]
        # Removes any explanation after
        pattern = re.compile(r'^(.*?)\n[A-Z]', flags=re.DOTALL)
        matches = re.findall(pattern, answer)
        if matches:
            answer = matches[0]
    # Cleaning format 2
    code_begin = max(answer.find("\ndef"), 0) + 4
    stops = ["\n#", "\ndef", "###", "\nif __name__", "\nprint", "\n\n\n\n", "\n\t\n\t\n\t"]
    for stop in stops:
        idx = answer.find(stop, code_begin)
        if idx != -1:
            answer = answer[:idx]
    return answer


# Extracts useful error from raw error
# Replaces full file name with "temp.py"
def extract_clean_error(error):
    if not error:
        return None
    # Regex to match file paths ending with temp<NUMBER>.py or temp.py
    file_path_pattern = r"(/[\w/]+/)?(temp\d*\.py)"
    def replace_with_fixed_name(match):
        return "temp.py"
    error = re.sub(file_path_pattern, replace_with_fixed_name, error)
    # Prevents infinitely long error message
    error = error[:1600]
    return error


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

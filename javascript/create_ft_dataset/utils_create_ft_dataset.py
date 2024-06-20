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
    "/**\n * * Write a JavaScript function to get the position of rightmost set bit.\n *\n * > getFirstSetBitPos(12)\n * 3\n * > getFirstSetBitPos(18)\n * 2\n * > getFirstSetBitPos(16)\n * 5\n */\nfunction getFirstSetBitPos(n) {\n if (n === 0) return -1;\n  return 32 - Math.floor(Math.log2(n));\n}\n",
    "/**\n * * Write a javascript function to abbreviate 'road' as 'rd.' in a given string.\n *\n * > roadRd(\"ravipadu Road\")\n * \"ravipadu Rd.\"\n * > roadRd(\"palnadu Road\")\n * \"palnadu Rd.\"\n * > roadRd(\"eshwar enclave Road\")\n * \"eshwar enclave Rd.\"\n */\nfunction roadRd(street) {\n return street.replace(/road/gi, \"rd.\");\n}\n",
    "/**\n * * Write a javascript function to remove lowercase substrings from a given string by using regex.\n *\n * > removeLowercase(\"KDeoALOklOOHserfLoAJSIskdsf\")\n * \"KDALOOOHLAJSI\"\n * > removeLowercase(\"ProducTnamEstreAmIngMediAplAYer\")\n * \"PTEAIMAAY\"\n * > removeLowercase(\"maNufacTuredbYSheZenTechNolOGIes\")\n * \"NTYSZTNOGI\"\n */\nfunction removeLowercase(str1) {\n return str1.replace(/\\b[a-z]+\\b/g, \"\");\n}\n"
]
fs_errors = [
    "\n/mnt/edward/data2/knw/lrpl/javascript/create_ft_dataset/temp.js:28\n    throw 'Error at 1th assert statement. Value = ' + JSON.stringify(x0)  \n    ^\nError at 1th assert statement. Value = 29\n(Use `node --trace-uncaught ...` to show where the exception was thrown)\n",
    "\n/mnt/edward/data2/knw/lrpl/javascript/create_ft_dataset/temp.js:27\n    throw 'Error at 1th assert statement. Value = ' + JSON.stringify(x0)  \n    ^\nError at 1th assert statement. Value = \"ravipadu rd.\"\n(Use `node --trace-uncaught ...` to show where the exception was thrown)\n",
    "\n/mnt/edward/data2/knw/lrpl/javascript/create_ft_dataset/temp1.js:27\n    throw 'Error at 1th assert statement. Value = ' + JSON.stringify(x0)  \n    ^\nError at 1th assert statement. Value = \"KDeoALOklOOHserfLoAJSIskdsf\"\n(Use `node --trace-uncaught ...` to show where the exception was thrown)\n"
]
fs_repairs = [
    "```javascript\n/**\n * * Write a JavaScript function to get the position of rightmost set bit.\n *\n * > getFirstSetBitPos(12)\n * 3\n * > getFirstSetBitPos(18)\n * 2\n * > getFirstSetBitPos(16)\n * 5\n */\nfunction getFirstSetBitPos(n) {\n    if (n === 0) return 0;\n    const rightmostSetBit = n & -n;\n    const position = Math.log2(rightmostSetBit) + 1;\n    return position;\n}\n```",
    '```javascript\n/**\n * * Write a javascript function to abbreviate \'road\' as \'rd.\' in a given string.\n *\n * > roadRd("ravipadu Road")\n * "ravipadu Rd."\n * > roadRd("palnadu Road")\n * "palnadu Rd."\n * > roadRd("eshwar enclave Road")\n * "eshwar enclave Rd."\n */\nfunction roadRd(street) {\n    return street.replace(/\\bRoad\\b/g, "Rd.");\n}\n```',
    '```javascript\n/**\n * * Write a javascript function to remove lowercase substrings from a given string by using regex.\n *\n * > removeLowercase("KDeoALOklOOHserfLoAJSIskdsf")\n * "KDALOOOHLAJSI"\n * > removeLowercase("ProducTnamEstreAmIngMediAplAYer")\n * "PTEAIMAAY"\n * > removeLowercase("maNufacTuredbYSheZenTechNolOGIes")\n * "NTYSZTNOGI"\n */\nfunction removeLowercase(str1) {\n    // Use a regular expression to remove all lowercase substrings\n    return str1.replace(/[a-z]+/g, \'\');\n}\n```'
]
fs_cot = [
    "The incorrect function is trying to compute the first set bit, but is doing this incorrectly since it floors the result of Math.log2, leading to an off by 1 error. To fix this, the correct function can isolate the first set bit by computing the bitwise AND between n and -n, and then calculate the final result with Math.log2.\n",
    "The error states we are failing the first test case. This is because the incorrect function is replacing 'road' with 'rd.', but the given test cases actually use the uppercase versions of 'Road' and 'Rd.'. The correct function should fix the replacement to handle the uppercase versions.\n",
    "The incorrect code is using the word boundary 'b' delimiter, which matches complete words that are lowercase. However, the function description wants to match substrings that are lowercase, which may not be full words. To fix this, the correct code should update the regex to search for general substrings.\n"
]


# Few (3) shot prompt
def create_prompt_gpt(incorrect_code, error):
    instruction = "You are given an incorrect javascript function and an error message. Explain how to fix the error, and then write an updated javascript function with the correct code."
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
{incorrect_code}
### Error: {error}
### Response:
'''
    return res


def fix_prompt(prompt):
    prompt = prompt.replace("Write a function", "Write a javascript function")
    return prompt


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
# For now, does nothing
def extract_clean_error(error):
    if not error:
        return None
    return error


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
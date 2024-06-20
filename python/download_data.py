import datasets
import json
import os
import random
import re


# Adds a line that actual calls the check() function and removes the metadata
def fix_test(test, func_name):
    pattern = r"\n\nMETADATA = \{.*\}\n\n"
    test = re.sub(pattern, "", test)
    test = test.replace("    ", "\t")
    return test + f"\ncheck({func_name})"


# Fixes the spacing within the comment
def fix_comment(comment):
    comment = re.sub(r' {2,}', '', comment)
    comment = comment.replace("\t", "\n")
    lines = comment.split("\n")
    res = ""
    for line in lines:
        if line:
            res += "\t" + line + "\n"
    res = "\t\"\"\"\n" + res + "\t\"\"\"\n"
    return res


# Fixes the prompt by combining the function header with the fixed comment
def fix_prompts(all_data, is_mbxp):
    for sample in all_data:
        text = sample["prompt"]
        if is_mbxp:
            func_header = text.find(":\n")
            func_header = text[:func_header + 2]
            start = text.find("\"\"\"") + 3
            end = text.find("\"\"\"", start)
            comment = text[start:end]
            comment = fix_comment(comment)
            text = func_header + comment
        text = text.replace("    ", "\t")
        sample["prompt"] = text
    return all_data


def write_local(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as file:
        for i, sample in enumerate(data):
            test = fix_test(sample["test"], sample["entry_point"])
            obj = {
                "id": i,
                "name": sample["task_id"],
                "prompt": sample["prompt"],
                "test": test,
                "entry_point": sample["entry_point"]
            }
            print(obj["prompt"])
            print(obj["test"])
            file.write(json.dumps(obj) + "\n")


# For some reason, the original MBXP spacing/tabbing in python is awful, so manually fix this
if __name__ == "__main__":

    # Download mbxp data, splits into train and test
    problems = datasets.load_dataset("mxeval/mbxp", "python")
    all_data = list(problems["test"])
    all_data = fix_prompts(all_data, is_mbxp=True)
    random.seed(17)
    random.shuffle(all_data)

    train_data = all_data[:800]
    test_data = all_data[800:]
    train_file = os.path.join("create_ft_dataset", "data", "train.jsonl")
    test_file = os.path.join("data", "mbxp", "test.jsonl")
    write_local(train_file, train_data)
    write_local(test_file, test_data)

    # Download multi-humaneval data, test only
    problems = datasets.load_dataset("mxeval/multi-humaneval", "python")
    test_data = list(problems["test"])
    test_data = fix_prompts(test_data, is_mbxp=False)
    test_file = os.path.join("data", "humaneval", "test.jsonl")
    write_local(test_file, test_data)


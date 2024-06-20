import datasets
import json
import os
import random
import re


# Tests declare variables as x0, x1, etc., but assert statements use only x
def fix_test(test):
    counter = 0
    def replace_xk(match):
        nonlocal counter
        result = f"JSON.stringify(x{counter})"
        counter += 1
        return result
    test = re.sub(r'JSON\.stringify\(x\)', replace_xk, test)
    return test


def write_local(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as file:
        for i, sample in enumerate(data):
            obj = {
                "id": i,
                "name": sample["task_id"],
                "prompt": sample["prompt"],
                "test": fix_test(sample["test"]),
                "entry_point": sample["entry_point"]
            }
            file.write(json.dumps(obj) + "\n")
            if i < 5:
                print(fix_test(sample["test"]))


if __name__ == "__main__":

    # Download mbxp data, splits into train and test
    problems = datasets.load_dataset("mxeval/mbxp", "javascript")
    all_data = list(problems["test"])
    random.seed(17)
    random.shuffle(all_data)
    train_data = all_data[:800]
    test_data = all_data[800:]
    train_file = os.path.join("create_ft_dataset", "data", "train.jsonl")
    test_file = os.path.join("data", "mbxp", "test.jsonl")
    write_local(train_file, train_data)
    write_local(test_file, test_data)

    # Download multi-humaneval data, test only
    problems = datasets.load_dataset("mxeval/multi-humaneval", "javascript")
    test_data = list(problems["test"])
    test_file = os.path.join("data", "humaneval", "test.jsonl")
    write_local(test_file, test_data)

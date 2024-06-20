import datasets
import json
import os
import random


def write_local(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as file:
        for i, sample in enumerate(data):
            obj = {
                "id": i,
                "name": sample["task_id"],
                "prompt": sample["prompt"],
                "test": sample["test"],
                "entry_point": sample["entry_point"]
            }
            print("------------------------------------------------------NEW-----------------------------------------------------")
            print(obj["prompt"])
            print(obj["test"])
            file.write(json.dumps(obj) + "\n")


# For some reason, the spacing/tabbing in python is awful, so manually fix this
if __name__ == "__main__":

    # Download mbxp data, splits into train and test
    problems = datasets.load_dataset("mxeval/mbxp", "swift")
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
    problems = datasets.load_dataset("mxeval/multi-humaneval", "swift")
    test_data = list(problems["test"])
    test_file = os.path.join("data", "humaneval", "test.jsonl")
    write_local(test_file, test_data)


import json
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


def main():
    input_file = "./output/mbxp/cl-7b-base/r0_copy.json"
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    for sample in input_data:
        sample["test"] = fix_test(sample["test"])
        print(sample["test"])

    output_file = "./output/mbxp/cl-7b-base/r0.json"
    with open(output_file, "w") as file:
        json.dump(input_data, file, indent=4)


main()
import json


def has_typo(r):
    if not r:
        return False
    if "1st" in r and "1th" in r:
        return True
    if "2nd" in r and "2th" in r:
        return True
    if "3rd" in r and "3th" in r:
        return True
    return False


def main():

    datasets = ["humaneval", "mbxp"]
    models = ["cl-7b-instruct", "cl-7b-base", "mistral-7b"]

    datasets = ["humaneval"]
    models = ["cl-7b-instruct"]

    for DATASET in datasets:
        for MODEL in models:
            # Load the judgement
            judge_file = f"./rationales/{DATASET}/{MODEL}/{RUN_NAME}/r1_judge.json"
            with open(judge_file, "r") as file:
                input_data = json.load(file)
            print("INPUT FILE:", judge_file)

            NUM_SAMPLES = min(5, len(input_data[0]["rationales"]))
            # row 0 = bad rationale, row 1 = good rationale
            # col 0 = bad code, col 1 = good code
            table = [[0,0],[0,0]] 

            for idx, question in enumerate(input_data):
                for i in range(NUM_SAMPLES):
                    if question["rationales"][i] and not has_typo(question["rationales"][i]): # TODO: fix ignore typo
                        verdict = question["judgements"][i]
                        code_result = question["r1_correct"][i]
                        table[verdict][code_result] += 1

            # Normalize table
            total = sum(sum(row) for row in table)
            table = [[x / total for x in row] for row in table]

            print("Rationale/Code Table:")
            for row in table:
                print(' '.join(['{:.3f}'.format(element) for element in row]))


if __name__ == "__main__":

    NUM_JUDGEMENTS = 1
    
    RUN_NAME = "icl"
    main()
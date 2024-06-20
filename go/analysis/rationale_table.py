import json


def main():

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
                    if question["rationales"][i]:
                        verdict = question["judgements"][i]
                        code_result = question["r1_correct"][i]
                        verdict = min(1, verdict)
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
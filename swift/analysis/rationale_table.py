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
            # Tracks frequency for histogram
            freq = [0] * (NUM_JUDGEMENTS+1)
            # row 0 = bad rationale, row 1 = good rationale
            # col 0 = bad code, col 1 = good code
            table = [[0,0],[0,0]] 

            for idx, question in enumerate(input_data):
                for i in range(NUM_SAMPLES):
                    if question["rationales"][i]:
                        score = question["judgements"][i]
                        freq[score] += 1
                        code_result = question["r1_correct"][i]
                        if score > GOOD_THRESHOLD:
                            table[1][code_result] += 1
                        if score < BAD_THRESHOLD:
                            table[0][code_result] += 1

            # Normalize histogram
            total = sum(freq)
            freq = [x / total for x in freq]

            # Normalize table
            total = sum(sum(row) for row in table)
            table = [[x / total for x in row] for row in table]

            print("Histogram:")
            formatted_freq = ', '.join([f'{value:.3f}' for value in freq])
            print("[" + formatted_freq + "]")
            print("Rationale/Code Table:")
            for row in table:
                print(' '.join(['{:.3f}'.format(element) for element in row]))


if __name__ == "__main__":

    NUM_JUDGEMENTS = 20
    GOOD_THRESHOLD = 15     # Need strictly greater for rationale to be "good"
    BAD_THRESHOLD = 5       # Need strictly less for rationale to be "bad"
    
    RUN_NAME = "ft"
    main()
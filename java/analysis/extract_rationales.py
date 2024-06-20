import json
import os


def extract_rationale(answer):
    stops = ["```", "###"]
    for stop in stops:
        idx = answer.find(stop)
        if idx != -1:
            answer = answer[:idx]
    while answer.endswith("\n"):
        answer = answer[:-1]
    return answer


def main():
    
    datasets = ["humaneval"]
    models = ["cl-7b-instruct"]


    for DATASET in datasets:
        for MODEL in models:

            # Load r0 repair to know if initial code was correct
            r0_file = f"../repair/{DATASET}/{MODEL}/r0.json"
            with open(r0_file, "r") as file:
                r0_repairs = json.load(file)
            print("R0 REPAIR FILE:", r0_file)

            # Load r1 output to get the rationale
            r1_file = f"../output/{DATASET}/{MODEL}/{RUN_NAME}/r1.json"
            with open(r1_file, "r") as file:
                r1_output = json.load(file)

            # Load r1 repairs to know if repair code was correct
            r1_file = f"../repair/{DATASET}/{MODEL}/{RUN_NAME}/r1.json"
            with open(r1_file, "r") as file:
                r1_repairs = json.load(file)
            print("R1 REPAIR FILE:", r1_file)

            # Configure output file
            rationale_file = f"./rationales/{DATASET}/{MODEL}/ft/r1.json"
            os.makedirs(os.path.dirname(rationale_file), exist_ok=True)
            print("RATIONALE FILE:", rationale_file)

            # Extract the rationales
            NUM_SAMPLES = min(5, len(r1_repairs[0]["answers"]))
            output = []
            for idx, sample in enumerate(r1_output):
                rationales = []
                r1_correct = []
                for i in range(NUM_SAMPLES):
                    r1_correct.append(r1_repairs[idx]["errors"][i] == None)
                    r0_error = r0_repairs[idx]["errors"][i]
                    if not r0_error:
                        rationales.append(None)
                    else:
                        rationale = extract_rationale(sample["answers"][i])
                        print("-" * 150)
                        print(rationale)
                        rationales.append(rationale)

                output.append({
                    "id": sample["id"],
                    "name": sample["name"],
                    "prompt": sample["prompt"],
                    "answers": r0_repairs[idx]["answers"][:NUM_SAMPLES],
                    "errors": r0_repairs[idx]["errors"][:NUM_SAMPLES],
                    "rationales": rationales,
                    "r1_correct": r1_correct
                })

            with open(rationale_file, 'w') as file:
                json.dump(output, file, indent=4)


if __name__ == "__main__":

    # Either fine-tune run name or baseline run name
    RUN_NAME = "e8-lr2"

    main()
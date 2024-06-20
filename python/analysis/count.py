import json
import os


def is_logical_error(answer):
    return ("assert candidate" in answer) or ("Timeout occurred" in answer)


def count_file(file_name, round):
    with open(file_name, "r") as file:
        repairs = json.load(file)
    num_samples = len(repairs[0]["answers"])
    total_wrong = [0] * num_samples
    syntax_wrong = [0] * num_samples
    logical_wrong = [0] * num_samples
    for sample in repairs:
        for i in range(num_samples):
            if not sample["errors"][i]:
                continue
            total_wrong[i] += 1
            if is_logical_error(sample["errors"][i]):
                logical_wrong[i] += 1
            else:
                syntax_wrong[i] += 1
    # Get the average over each round
    avg_total_wrong = sum(total_wrong) / len(total_wrong)
    avg_syntax_wrong = sum(syntax_wrong) / len(syntax_wrong)
    avg_logical_wrong = sum(logical_wrong) / len(logical_wrong)
    return {
        "round": round,
        "avg_total": avg_total_wrong,
        "avg_syntax": avg_syntax_wrong,
        "avg_logical": avg_logical_wrong,
        "avg_percent_syntax": avg_syntax_wrong / (avg_syntax_wrong + avg_logical_wrong)
    }


# Returns a list of 5 count objects (init and 4 repair rounds)
# Each object stores <total wrong, syntax wrong, logical wrong, percent syntax wrong>
def count_all_rounds(repair_dir):
    res = []
    # Count initial
    initial_repair_file = os.path.join(os.path.dirname(repair_dir), "r0.json")
    res.append(count_file(initial_repair_file, 0))
    # Count repair rounds
    for i in range(1, 5):
        repair_file = os.path.join(repair_dir, f"r{i}.json")
        res.append(count_file(repair_file, i))
    return res


# 2 datasets (Humaneval and MBXP)
# 3 models (cli, clb, mist)
# 4 runs (base, icl, ft, gpt)
# 5 rounds (init and 4 repair rounds)
# Creates one big list with all the json
def main():

    datasets = ["humaneval", "mbxp"]
    models = ["cl-7b-instruct", "cl-7b-base", "mistral-7b"]
    run_names = ["base", "icl", "gpt"]
    
    
    for model in models:
        res = []
        for dataset in datasets:
            for run_name in run_names:
                current_dir = f"../repair/{dataset}/{model}/{run_name}"
                res.append({
                    "dataset": dataset,
                    "model": model,
                    "run_name": run_name,
                    "analysis": count_all_rounds(current_dir)
                })
            # UPDATE your fine-tune run names here
            if model == "mistral-7b":
                current_dir = f"../repair/{dataset}/{model}/e5-lr5"
                res.append({
                    "dataset": dataset,
                    "model": model,
                    "run_name": "fine-tuned",
                    "analysis": count_all_rounds(current_dir)
                })
            else:
                current_dir = f"../repair/{dataset}/{model}/e8-lr2"
                res.append({
                    "dataset": dataset,
                    "model": model,
                    "run_name": "fine-tuned",
                    "analysis": count_all_rounds(current_dir)
                })

        output_file = f"./counts/analysis_{model}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as file:
            json.dump(res, file, indent=4)
                

if __name__ == "__main__":
    
    main()
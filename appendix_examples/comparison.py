import json
import os


# Returns True if repair occurred between r0 and r1
def did_repair(sample):
    return sample["r0"] == False and sample["r1"] == True


# Takes a language name and run name for dir path 
# question object = <pass r0, pass r1>
# returns a dict mapping HumanEval question id to a list of NUM_SAMPLES question objects
def get_question_obj_dict(language, run_name):
    repair_r0_file = f"../{language}/repair/humaneval/{MODEL}/r0.json"
    repair_r1_file = f"../{language}/repair/humaneval/{MODEL}/{run_name}/r1.json"
    with open(repair_r0_file, "r") as file:
        repairs_r0 = json.load(file)
    with open(repair_r1_file, "r") as file:
        repairs_r1 = json.load(file)

    res = {}
    for i in range(len(repairs_r0)):
        id = repairs_r0[i]["name"].split('/')[1]
        question_objs = []
        for j in range(NUM_SAMPLES):
            question_objs.append({
                "r0": (repairs_r0[i]["errors"][j] == None),
                "r1": (repairs_r1[i]["errors"][j] == None)
            })
        res[id] = question_objs
    return res


# Given a run name, loads all the necessary text data
# For r0, we just care about code, so use repair file
# For r1, we need rationale, so use output file
def load_repair_text(language):
    # Stores 4 lists of data: r0, r1 base, r1 icl, r1 ft
    res = {}
    run_names = ["base", "icl", FT_RUN]

    # Load r0 repair
    repair_file = f"../{language}/repair/humaneval/{MODEL}/r0.json"
    with open(repair_file, "r") as file:
        repairs = json.load(file)
    res["r0"] = repairs
    
    # Load base, icl, ft repairs
    for run_name in run_names:
        repair_file = f"../{language}/output/humaneval/{MODEL}/{run_name}/r1.json"
        with open(repair_file, "r") as file:
            repairs = json.load(file)
        res[run_name] = repairs
    
    return res


# Finds a sample in the text file where the HumanEval name matches the question id
def get_sample(output_list, qid, sample_id):
    for result in output_list:
        if result["name"].split('/')[1] == qid:
            return result["answers"][sample_id]


# For each qid, create one subdir for each language
# Each subdir stores 4 files: r0, base r1, icl r1, ft r1
def write_join_files(interesting_set):

    run_names = ["r0", "base", "icl", FT_RUN]

    # Stores 4 lists of data: r0, r1 base, r1 icl, r1 ft
    text = load_repair_text(LANG)

    # Using the question and sample id, get the associated text:
    for (qid, sample_id) in interesting_set.items():
        for run_name in run_names:
            output_file = os.path.join(LANG, qid, f"{run_name}.java")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as file:
                file.write(get_sample(text[run_name], qid, sample_id))
        

# Takes a language name as dir path
# Compares <base, ICL, FT> of a given language
# 1: Map each HumanEval question to the id
# 2: Read the output file (r0) and next output file (r1) in parallel
# 3: From output file, get the (Q,A)
# 4: From the repair file, get the (E, R)
# 5: If FT is correct but base and ICL are incorrect, add to res
def main():

    # Stores 3 lists of question object dictionaires
    all_runs = {}
    run_names = ["base", "icl", FT_RUN]

    # Get data
    for run_name in run_names:
        all_runs[run_name] = get_question_obj_dict(LANG, run_name)

    # For each question id, find if <fail, fail, pass> exists
    # Map each question id to the specific sample id within [0, NUM_SAMPLES)
    interesting_set = {}
    for qid in all_runs["base"]:
        for i in range(NUM_SAMPLES):
            repair_base = did_repair(all_runs["base"][qid][i])
            repair_icl = did_repair(all_runs["icl"][qid][i])
            repair_ft = did_repair(all_runs[FT_RUN][qid][i])        
            if not repair_base and not repair_icl and repair_ft:
                interesting_set[qid] = i
                break

    print("INTERESTING SET:")
    print(interesting_set)
    write_join_files(interesting_set)
                

# Extracts "interesting" question results, interesting = base & ICL repair fails, FT repair works
# Only running on cl-7b-instruct for now
if __name__ == "__main__":
    
    MODEL = "cl-7b-instruct"
    NUM_SAMPLES = 5
    FT_RUN = "e8-lr2"
    LANG = "java"

    main()
    print("COMPARISON COMPLETED!")
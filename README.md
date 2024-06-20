# DistiLRR

Source code for the paper DistiLRR: Transferring Code Repair for Low-Resource Programming Languages<br>
Authors: Kyle Wong, Alfonso Amayuelas, Liangming Pan, William Yang Wang<br>


## Dependencies
The necessary dependencies are included in requirements.txt


## Workflow
Each language folder is set up in the exact same structure. Below are the main steps to reproduce our work.
1. Enter a language folder. Download the MBXP and HumanEval datasets with `python download_data.py`
2. Create the fine-tuning dataset
3. Fine-tune a DistiLRR model
4. Generate and evaluate the initial round
5. Generate and evaluate rounds of code repair
6. Perform the further analysis (from section 5 in our paper)


## Dataset Construction
* Enter the `create_ft_dataset/` directory
* Run `bash create_qae.sh` to obtain the student's incorrect answers
* Run `python create_qaer.py` to obtain the teacher's correct repairs
* Run `python format_qaer.py` to construct the fine-tune ready dataset


## Fine-tuning
* Enter the `finetune/` directory
* Run `bash finetune.sh` for fine-tuning. The hyperparameters we used are listed in Appendix B of our paper.


## Generate/Evaluate Initial Round
* Run `bash gen_init.sh` to obtain the initial answers
* Run `python eval_init.py` to obtain the initial errors


## Generate/Evaluate Code Repair
* Run `bash gen_repair.sh` to repair with DistiLRR
* Run `bash gen_repair_<baseline>.sh` to repair with baselines


## Analysis
* Enter the `analysis/` directory
* Run `python extract_rationales.py` to extract the rationales from a repair run
* Run `python judge_rationales.py` to judge rationale quality with GPT-4. Be mindful of the amount of tokens used, since these queries may be costly.
* Run `python rationale_table.py` to compute the rate of good/bad rationales and correct/incorrect code
* Run `python count.py` to count syntax errors within a repair run


## Execution Setup
To test generated code, we write to a temp file and execute it via `subprocess`. You can execute the code locally or inside a docker container. Our code contains examples for both options. For local execution, you need to install the necessary software. For containerized execution, you need to download a parent image that has the necessary software. A containerized execution example is in `swift/`, while local execution examples are in the other language folders. Executing code locally is probably the simpler option.
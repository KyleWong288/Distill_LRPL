import os
import datasets


DSC_MODEL_DICT = {
    "cl-7b-instruct": "codellama/CodeLlama-7b-Instruct-hf",
    "cl-7b-base": "codellama/CodeLlama-7b-hf",
    "cg2-7b": "Salesforce/codegen2-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1"
}


lora_module_dict = {
    'cl-7b-instruct': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
    'cl-7b-base': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
    'cg2-7b': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
    'mistral-7b': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ]
}


def parse_model_name(name):
    if name in DSC_MODEL_DICT:
        return DSC_MODEL_DICT[name]
    else:
        raise ValueError(f"Undefined base model: {name}")


def load_dataset(dataset_dir, model_name):
    train_file = os.path.join(dataset_dir, model_name, "train.jsonl")
    dev_file = os.path.join(dataset_dir, model_name, "dev.jsonl")
    print("TRAIN DATA:", train_file)
    print("DEV DATA:", dev_file)
    train_data = datasets.load_dataset("json", data_files=train_file, split="train", download_mode="force_redownload")
    dev_data = datasets.load_dataset("json", data_files=dev_file, split="train", download_mode="force_redownload")
    return train_data, dev_data
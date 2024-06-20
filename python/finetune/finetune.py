import argparse
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import TaskType, LoraConfig
from utils_ft import *


def main(args):
    model_name = parse_model_name(args.base_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    print("MODEL LOADED:", model_name)

    # Load data
    train_dataset, dev_dataset = load_dataset(args.data_path, args.base_model)
    print("DATASETS LOADED:", args.data_path)

    # Configure training args
    training_args = TrainingArguments(
        output_dir=f'../finetuned_models/{args.base_model}/{args.run_name}',
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.scheduler,
        weight_decay=args.weight_decay,
        fp16=True,
        report_to='wandb',
        run_name=args.run_name,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.model.config.use_cache = False

    # Configure peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=128,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=lora_module_dict[args.base_model],
        bias="none",
    )

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=args.max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        peft_config=peft_config,
    )

    print("TRAINING BEGIN")
    torch.cuda.empty_cache()
    trainer.train()

    # Save model
    model.save_pretrained(training_args.output_dir)
    print("MODEL SUCCESSFULLY SAVED")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--num_epochs", default=5, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--evaluation_strategy", default='steps', type=str)  
    parser.add_argument("--eval_steps", default=100, type=float) 
    parser.add_argument("--warmup_steps", default=10, type=float)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--from_remote", default=False, type=bool)
    parser.add_argument("--deepspeed", default='./config_new.json', type=str)
    args = parser.parse_args()
    
    wandb.login()
    main(args)
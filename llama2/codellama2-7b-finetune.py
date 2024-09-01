import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from datasets import load_from_disk
import evaluate
from trl import SFTTrainer
from config import ScriptArguments

project_config = {
    "survey-json": {
        "project_name": "survey-json-model-inst",
        "train_dataset_path": "../datasets/survey_json_datasets_instruction_train",
        "test_dataset_path": "../datasets/survey_json_datasets_instruction_test"
    },
    "schema": {
        "project_name": "schema-model-inst",
        "train_dataset_path": "../datasets/schema_datasets/schema_data_train",
        "test_dataset_path": "../datasets/schema_datasets/schema_data_test"
    },
    "paraloq": {
        "project_name": "paraloq-model-inst",
        "train_dataset_path": "../datasets/paraloq/paraloq_data_train",
        "test_dataset_path": "../datasets/paraloq/paraloq_data_test"
    },
    "nous": {
        "project_name": "nous-model-inst",
        "train_dataset_path": "../datasets/nous/nous_data_train",
        "test_dataset_path": "../datasets/nous/nous_data_test"
    }
}

def main(project="schema"):
    assert len(os.environ["HF_TOKEN"]) != 0
    project_name = project_config[project]["project_name"]
    train_dataset = load_from_disk(project_config[project]["train_dataset_path"])
    test_dataset = load_from_disk(project_config[project]["test_dataset_path"])
    os.environ["WANDB_PROJECT"] = project_name
    # os.environ["WANDB_MODE"] = "offline"
    run_name = "codellama2-7b-qlora-survey-json"
    script_args = ScriptArguments()
    output_dir = f"model_output/{project_name}/{run_name}"
    
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def formatting_func(example):
        text = f"### USER:\nConvert the question list to survey json.\n{example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
        return text

    model_id = "NousResearch/CodeLlama-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        torch_dtype=torch.float32,
        attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2"
    )
    model.config.use_cache = False 
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    lora_config = LoraConfig(
        r=script_args.lora_r,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout
    )

    train_dataset = load_from_disk("../datasets/survey_json_datasets_train")
    test_dataset = load_from_disk("../datasets/survey_json_datasets_test")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        max_grad_norm=script_args.max_grad_norm,
        max_steps=script_args.max_steps,
        warmup_ratio=script_args.warmup_ratio,
        lr_scheduler_type=script_args.lr_scheduler_type,
        gradient_checkpointing=script_args.gradient_checkpointing,
        group_by_length=script_args.group_by_length,
        # evaluation_strategy="steps", # only enable if SFTTrainer is configured
        # eval_steps=script_args.eval_steps, # only enable if SFTTrainer is configured
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        report_to="wandb",
        run_name=run_name,
    )

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        peft_config=lora_config,
        packing=script_args.packing,
        dataset_text_field="id",
        tokenizer=tokenizer,
        max_seq_length=script_args.max_seq_length,
        formatting_func=formatting_func,
        # eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics
    )

    trainer.train()
    # trainer.evaluate(eval_dataset=eval_dataset)


if __name__ == "__main__":
    main()
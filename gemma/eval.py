import os
import torch
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
from trl import SFTTrainer
from config import ScriptArguments

os.environ["WANDB_PROJECT"] = "survey-json-model-eval"
run_name = "gemma-7b-qlora-survey-json"
checkpoint_id = "checkpoint-500"
model_id = os.path.join("model_output", run_name, "checkpoint-500")
script_args = ScriptArguments

test_dataset = load_from_disk("../datasets/survey_json_datasets_test")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quantization_config, 
    torch_dtype=torch.float32,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

for example in test_dataset:
    # print(f"### USER:\nConvert the question list to survey json.\n{example['data'][0]}\n### ASSISTANT: {example['data'][1]}")

    input_text = f"### USER:\n Convert the question list to survey json.\n{example['data'][0]}"
    print(input_text + "/n/n")
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=1024*2)
    print(tokenizer.decode(outputs[0]))
    break
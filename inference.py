import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

checkpoint_path = "checkpoint-600"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path, 
    quantization_config=quantization_config, 
    torch_dtype=torch.float16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side='left')
tokenizer.pad_token_id = tokenizer.eos_token_id

instruction = """{"test": "test"}"""

response = ""
prompt = f"<s>[INST] Convert the question list to survey json.\n{instruction} [/INST] {response}"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto")
results = pipe(prompt, do_sample=True, temperature=1, max_new_tokens=2088)
print(results)

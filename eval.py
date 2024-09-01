import os
import torch
from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import Levenshtein
import json
from tqdm import tqdm

test_dataset = load_from_disk("./datasets/survey_json_datasets_instruction_test")

def get_model_id(model_type, run_name, checkpoint_id):
    return os.path.join(model_type, "model_output", run_name, checkpoint_id)

# model_type = "gemma"
# run_name = "gemma-7b-qlora-survey-json"
checkpoint_id = "checkpoint-500"

mistral_7b_model_id = get_model_id("mistral", "mistral-7b-qlora-inst", checkpoint_id)
gemma_7b_model_id = get_model_id("gemma", "gemma-7b-qlora-inst", checkpoint_id)
gemma_2b_model_id = get_model_id("gemma", "gemma-2b-qlora-inst", checkpoint_id)
codellama2_7b_model_id = get_model_id("llama2", "codellama2-7b-qlora-inst", checkpoint_id)
llama3_7b_model_id = get_model_id("llama3", "llama3-7b-qlora-inst", checkpoint_id)
phi_2_model_id = get_model_id("phi2", "phi-2-qlora-inst", checkpoint_id)

model_ids = {
    # "mistral_7b": mistral_7b_model_id,
    # "gemma_7b": gemma_7b_model_id,
    # "gemma_2b": gemma_2b_model_id,
    # "codellama2_7b": codellama2_7b_model_id,
    # "llama3_7b": llama3_7b_model_id,
    "phi_2": phi_2_model_id
}

for model_name in model_ids.keys():
    print("running test on", model_name)
    model_id = model_ids[model_name]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    batch_size = 8

    prompts = []
    ground_truths = []
    preds = []
    ground_truths = []
    for i in tqdm(range(0, len(test_dataset))):
        data = test_dataset[i]
        prompt, ground_truth = data['text'].split("[/INST]")
        prompt += "[/INST]"
        prompts.append(prompt)
        ground_truths.append(ground_truth.strip())

        
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        results = pipe(batch, do_sample=True, temperature=1, max_new_tokens=2088)

        for result in results:
            output = result[0]['generated_text']
            pred = output.split("[/INST]")[1].strip()
            preds.append(pred)
            
    print("Saving Predictions")
    # save preds to file 
    with open(f"predictions/{model_name}_predictions.txt", 'w') as f:
        for pred in preds:
            f.write(pred + "\n")
    
    # save ground truths to file
    with open(f"predictions/{model_name}_ground_truths.txt", 'w') as f:
        for gt in ground_truths:
            f.write(gt + "\n")
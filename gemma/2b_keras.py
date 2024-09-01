import os

os.environ["KAGGLE_USERNAME"] = "dylanli"
os.environ["KAGGLE_KEY"] = ""
os.environ["KERAS_BACKEND"] = "torch"  # Or "torch" or "tensorflow" or "jax"
# Avoid memory fragmentation on JAX backend.
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

import keras
import keras_nlp
import torch

# import wandb
# from wandb.keras import WandbMetricsLogger
max_token_length=1024

import pandas as pd
from sklearn.model_selection import train_test_split

def estimate_token_length(sentence):
    average_token_length = 4  # Adjust this based on the specific model
    return len(sentence) // average_token_length

def build_training_data(df, lang = 'en-us', template=None, max_token_length=max_token_length):
    df = df[df['Survey language(s) (comma-separated language codes)'] == lang]
    instructions = df['Simple question list as text'].tolist()
    responses = df['Simple survey json as text'].tolist()
    
    training_data = []
    
    for instruction, response in zip(instructions, responses):
        instruction_tokens = estimate_token_length(instruction)
        response_tokens = estimate_token_length(response)
        if instruction_tokens > max_token_length or response_tokens > max_token_length:
            continue
        training_data.append(template.format(instruction=instruction, response=response))
    
    return training_data


template = "Instruction:\nConvert the question list to survey json.\n{instruction}\n\nResponse:\n{response}"
df = pd.read_csv('../datasets/questionList_to_simpleJson.csv')
data = build_training_data(df, lang='en-us', template=template)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print("Train data size:", len(train_data), "Test data size:", len(test_data))

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
print(gemma_lm.summary())

torch.cuda.empty_cache()
prompt = template.format(
    instruction="Convert the question list to survey json.\nWhat is the name of this place?\nWhere is this place?\nWhat kind of place is this?\n  - School\n  - Park\n  - Restaurant\nHow do you interact with this space?\nHow important is this place to you?\n  - 1\n  - 2\n  - 3\nHow many times per week do you visit this place?\n",
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=max_token_length))
# There is no specific code to clear touch cache in Python. If you are referring to clearing GPU memory, you can use the following code:

# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
print(gemma_lm.summary())


# Limit the input sequence length to 512 (to control memory usage).
gemma_lm.preprocessor.sequence_length = max_token_length
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
# wandb.init(project="gemma")
gemma_lm.fit(data, epochs=10, batch_size=1)

torch.cuda.empty_cache()
prompt = template.format(
    instruction="Convert the question list to survey json.\nWhat is the name of this place?\nWhere is this place?\nWhat kind of place is this?\n  - School\n  - Park\n  - Restaurant\nHow do you interact with this space?\nHow important is this place to you?\n  - 1\n  - 2\n  - 3\nHow many times per week do you visit this place?\n",
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=max_token_length))
# There is no specific code to clear touch cache in Python. If you are referring to clearing GPU memory, you can use the following code:

# Save the model to disk.
timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
gemma_lm.save("gemma_lm_" + timestamp + ".keras")


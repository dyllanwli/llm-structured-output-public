import pandas as pd

from datasets import Dataset
from datasets import load_dataset


def estimate_token_length(sentence):
    average_token_length = 4  # Adjust this based on the specific model
    return len(sentence) // average_token_length

def build_training_data(df, lang = 'en-us', max_token_length=1024):
    df = df[df['Survey language(s) (comma-separated language codes)'] == lang]
    instructions = df['Simple question list as text'].tolist()
    responses = df['Simple survey json as text'].tolist()
    

    texts = []
    
    for instruction, response in zip(instructions, responses):
        instruction_tokens = estimate_token_length(instruction)
        response_tokens = estimate_token_length(response)
        if instruction_tokens > max_token_length or response_tokens > max_token_length:
            continue
        prompt = f"<s>[INST] Convert the question list to survey json.\n{instruction} [/INST] {response}"
        texts.append(prompt)
    
    return {"text": texts}

max_token_length = 2048
df = pd.read_csv('./datasets/survey_json_metadata/questionList_to_simpleJson.csv')
data = build_training_data(df, lang='en-us', max_token_length=max_token_length)
dataset = Dataset.from_dict(data)
dataset.save_to_disk('./datasets/survey_json_datasets_instruction/')
train_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()
train_dataset.save_to_disk('./datasets/survey_json_datasets_instruction_train/')
test_dataset.save_to_disk('./datasets/survey_json_datasets_instruction_test/')
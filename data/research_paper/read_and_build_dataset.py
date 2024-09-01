import json  


def load_documents_from_spatial_data():
    uuid_list = []
    # Read UUIDs from spatial_data.txt
    with open('spatial_data.txt', 'r') as file:
        uuid_list = [line.strip() for line in file.readlines()]

    documents = []
    # Load documents from the train folder based on UUIDs
    for uuid in uuid_list:
        try:
            with open(f'train/{uuid}.json', 'r') as doc_file:  # Changed to .json
                documents.append(json.load(doc_file))  # Load JSON data
        except FileNotFoundError:
            print(f'Document for UUID {uuid} not found.')

    return documents

def preprocess(documents):
    count_tokens = 0
    max_tokens = 0
    for document in documents:
        for item in document:
            count_tokens += len(item["text"])
            max_tokens = max(max_tokens, len(item["text"]))
    print(count_tokens//4)
    print(max_tokens//4)

documents = load_documents_from_spatial_data()

preprocess(documents)
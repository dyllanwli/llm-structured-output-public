import os
import json
import Levenshtein
import re
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import json

def edit_distance(json_str1, json_str2):
    return Levenshtein.distance(json_str1, json_str2)


def extract_json(sample, project="schema"):
    if sample is None or sample == "nan":
        return None, {"format_error": True}

    prompt = ChatPromptTemplate.from_template(
        "Format and return only JSON array (no comments line) content from: {obj}"
    )
    model = ChatOpenAI(temperature=0)
    chain = prompt | model

    json_str = sample
    if project == "schema":
        match = re.search(
            r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
            sample,
            re.DOTALL,
        )
        json_str = match.group(1) if match else None
    try:
        json_obj = json.loads(json_str)
        return json_obj, {"format_error": False}
    except:
        try:
            json_str = chain.invoke({"obj": sample}).content
            json_obj = json.loads(json_str)
            return json_obj, {"format_error": True}
        except:
            return None, {"format_error": True, "error": True}


def setup_api_key(file_path="../config.json"):
    with open(file_path, "r") as file:
        data = json.load(file)
    api_key = data["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key


def content_score(generated_json_output, actual_json_output):
    # This function is aims to use the LLM to evalute the two json pairs with following cirteria:
    prompt_template = """
    You will be presented with two JSON outputs: 
    1. Generated JSON output from an LLM based on free-form text
    2. Actual JSON output (ground truth)

    Your task is to evaluate the quality of the generated JSON output (Output 1) by comparing it to the actual JSON output (Output 2) based on the following criteria:

    {criteria_text}

    For each criterion, rate the generated JSON output on a scale of 1 to 5, where 5 is the best. Provide a brief explanation for each rating.

    Generated JSON Output (Output 1):
    {output1}

    Actual JSON Output (Output 2 - Ground Truth):
    {output2}
    """
    criteria = {
        "Completeness": "Assess how well the generated JSON captures all the relevant information present in the input text. This aims to check if all the important entities, relationships, and attributes are correctly identified and included in the output.",
        "Accuracy": "Evaluate the correctness of the extracted information in the JSON output. Verify if the values, data types, and structures match the information provided in the input text. This aims to check for any errors, inconsistencies, or misinterpretations.",
        "Granularity": "Consider the level of detail captured in the JSON output. Assess whether the generated structure provides an appropriate level of granularity based on the requirements of the task. This aims to determine if the JSON includes all the necessary fields and sub-structures to represent the information effectively.",
        "Consistency": "If multiple examples or instances are provided, evaluate the consistency of the generated JSON across different inputs. This aims to check if the LLM maintains a consistent structure, naming conventions, and data representation across various examples.",
    }

    class Metric(BaseModel):
        completeness: float = Field(
            description="Completeness score for the generated JSON output",
            default=1,
            ge=1,
            le=5,
        )
        accuracy: float = Field(
            description="Accuracy score for the generated JSON output",
            default=1,
            ge=1,
            le=5,
        )
        granularity: float = Field(
            description="Granularity score for the generated JSON output",
            default=1,
            ge=1,
            le=5,
        )
        consistency: float = Field(
            description="Consistency score for the generated JSON output",
            default=1,
            ge=1,
            le=5,
        )

    def output_parser(query):
        model = ChatOpenAI(temperature=0)
        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=Metric)

        prompt = PromptTemplate(
            template="Format the user query\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser

        return chain.invoke({"query": query})
        

    def evaluate_outputs(generated_output, actual_output):
        criteria_text = "\n".join(f"{key}: {value}" for key, value in criteria.items())

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["criteria_text", "output1", "output2"],
        )
        model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

        chain = prompt | model | StrOutputParser()

        response = chain.invoke(
            {
                "criteria_text": criteria_text,
                "output1": generated_output,
                "output2": actual_output,
            }
        )
        return output_parser(response)

    evaluation_result = evaluate_outputs(generated_json_output, actual_json_output)
    return evaluation_result


def levenshtein_distance(predict, ground_truth):
    epsilon = 1e-10
    distance = Levenshtein.distance(predict, ground_truth)
    max_distance = max(len(predict), len(ground_truth))
    accuracy = (max_distance - distance) / (max_distance + epsilon)
    return accuracy


def json_score(json1, json2, key_weight=0.5, value_weight=0.5):
    # If json1 and json2 are lists, calculate the similarity score for each pair of elements
    if isinstance(json1, list) and isinstance(json2, list):
        scores_list = []
        for i in range(min(len(json1), len(json2))):
            scores_list.append(json_score(json1[i], json2[i], key_weight, value_weight))
        scores = {
            "overall": sum(score["overall"] for score in scores_list)
            / len(scores_list),
            "key": sum(score["key"] for score in scores_list) / len(scores_list),
            "value": sum(score["value"] for score in scores_list) / len(scores_list),
            "json_array": True,
        }
        return scores
    elif isinstance(json1, list) or isinstance(json2, list):
        return {
            "overall": 0,
            "key": 0,
            "value": 0,
            "json_array": True,
        }

    key_scores = []
    value_scores = []

    def extract_pairs(json_obj, prefix=""):
        pairs = set()
        if json_obj is not None:
            for key, value in json_obj.items():
                if isinstance(value, dict):
                    pairs.update(extract_pairs(value, prefix + str(key) + "."))
                else:
                    pairs.add((prefix + str(key), str(value)))
        return pairs

    pairs1 = extract_pairs(json1)
    pairs2 = extract_pairs(json2)

    common_pairs = pairs1 & pairs2

    for pair in common_pairs:
        key, value = pair
        key_scores.append(1.0)
        value_scores.append(1.0)

    for pair in pairs1 - common_pairs:
        key1, value1 = pair
        max_key_score = 0
        max_value_score = 0
        for pair2 in pairs2 - common_pairs:
            key2, value2 = pair2
            key_distance = levenshtein_distance(key1, key2)
            key_similarity = key_distance
            max_key_score = max(max_key_score, key_similarity)

            value_distance = levenshtein_distance(value1, value2)
            value_similarity = value_distance
            max_value_score = max(max_value_score, value_similarity)
        key_scores.append(max_key_score)
        value_scores.append(max_value_score)

    for pair in pairs2 - common_pairs:
        key2, value2 = pair
        max_key_score = 0
        max_value_score = 0
        for pair1 in pairs1 - common_pairs:
            key1, value1 = pair1
            key_distance = levenshtein_distance(key1, key2)
            key_similarity = key_distance
            max_key_score = max(max_key_score, key_similarity)

            value_distance = levenshtein_distance(value1, value2)
            value_similarity = value_distance
            max_value_score = max(max_value_score, value_similarity)
        key_scores.append(max_key_score)
        value_scores.append(max_value_score)

    avg_key_score = sum(key_scores) / len(key_scores) if key_scores else 0
    avg_value_score = sum(value_scores) / len(value_scores) if value_scores else 0

    scores = {
        "overall": key_weight * avg_key_score + value_weight * avg_value_score,
        "key": avg_key_score,
        "value": avg_value_score,
        "json_array": False,
    }
    return scores

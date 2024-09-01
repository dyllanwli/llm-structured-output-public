import sys
import os
from datasets import load_from_disk
import json
from tqdm import tqdm
current_dir = os.getcwd()

sys.path.append(os.path.dirname(current_dir))

from utils import setup_api_key, levenshtein_distance

setup_api_key(file_path="../../config.json")


def get_model_id(model_type, run_name, project_name, checkpoint_id):
    return os.path.join(
        model_type, "model_output", project_name, run_name, checkpoint_id
    )

from utils import json_score, extract_json, content_score
from sklearn.metrics import f1_score, recall_score

run_list = ["schema", "paraloq", "nous"]
for project in run_list:
    print(f"Running for project: {project}")
    # get_result(project=project)

    # load json file
    with open(f"./{project}_instruction_generation.json", "r") as f:
        data = json.load(f)
    overall_score = 0
    overall_key_score = 0
    overall_value_score = 0
    foramt_error_count = 0
    json_array_count = 0
    edit_distance = 0
    n = len(data["generated_responses"])
    y_true = [True] * n
    y_pred = []

    content_scores = {
        "completeness": [],
        "accuracy": [],
        "granularity": [],
        "consistency": [],
    }
    for i in tqdm(range(n)):
        try:
            generated_response = data["generated_responses"][i]
            actual_response = data["actual_responses"][i]
            generated_json, format_error = extract_json(
                generated_response, project=project
            )
            actual_json, _ = extract_json(actual_response, project=project)
            if format_error["format_error"]:
                foramt_error_count += 1
            j_scores = json_score(generated_json, actual_json)
            if j_scores["key"] >= 0.99 and j_scores["value"] >= 0.8:
                y_pred.append(True)
            else:
                y_pred.append(False)
            overall_score += j_scores["overall"]
            overall_key_score += j_scores["key"]
            overall_value_score += j_scores["value"]
            if j_scores["json_array"]:
                json_array_count += 1

            # get content score
            c_score = content_score(generated_response, actual_response)
            # print(f"Content score: {c_score}")
            content_scores["completeness"].append(c_score["completeness"])
            content_scores["accuracy"].append(c_score["accuracy"])
            content_scores["granularity"].append(c_score["granularity"])
            content_scores["consistency"].append(c_score["consistency"])
            edit_distance += levenshtein_distance(generated_response, actual_response)
        except Exception as e:
            print(generated_response)
            raise e

    # get avg content score
    na_count = 0
    for key in content_scores.keys():
        # dealing with na
        for x in range(len(content_scores[key])):
            if (
                x == "NA"
                or x == "N/A"
                or x == "na"
                or x == "n/a"
                or x == "nan"
                or x is None
            ):
                content_scores[key][x] = 5
                na_count += 1
        content_scores[key] = sum(content_scores[key]) / len(content_scores[key])
        print("na count for key: ", key, na_count)

    avg_score = overall_score / n
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    summary = {
        "avg_score": avg_score,
        "avg_key_score": overall_key_score / n,
        "avg_value_score": overall_value_score / n,
        "format_error_count": foramt_error_count,
        "f1_score": f1,
        "recall": recall,
        "content_scores": content_scores,
        "edit_distance": edit_distance / n,
    }
    print(summary)
    print("\n")

    with open(f"./{project}_instruction_generation_summary.json", "w") as f:
        json.dump(summary, f)

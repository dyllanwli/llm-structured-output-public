import Levenshtein
import json

def levenshtein_accuracy(predict, ground_truth):
    epsilon = 1e-10
    # Calculate the Levenshtein distance between the two strings
    distance = Levenshtein.distance(predict, ground_truth)

    # Calculate the maximum possible distance (i.e., the length of the longer string)
    max_distance = max(len(predict), len(ground_truth))

    # Calculate the accuracy as the complement of the normalized distance
    accuracy = (max_distance - distance) / (max_distance + epsilon)

    return accuracy

def survey_json_accuracy(pred_string, true_string):
    true_json = json.loads(true_string)
    pred_json = json.loads(pred_string)

    """
    print(data.keys())
    Usually have three fields:
        header
        questions
        subHeader (optional)
    """
    header_accuracy = 0
    subheader_accuracy = 0
    question_accuracy = []
    if "questions" in pred_json:
        for true_q, pred_q in zip(true_json["questions"], pred_json["questions"]):
            # compare the questions 
            key_accuracy = dict()
            for k in true_q.keys():
                if k in pred_q:
                    acc = levenshtein_accuracy(str(true_q[k]), str(pred_q[k]))
                else:
                    acc = 0
                key_accuracy[k] = acc
            question_accuracy.append(key_accuracy)
    else:
        question_accuracy.append(dict())

    if "header" in pred_json:
        header_accuracy = levenshtein_accuracy(pred_json["header"], true_json["header"])
    else:
        header_accuracy = 0

    if "subHeader" in pred_json:
        subheader_accuracy = levenshtein_accuracy(pred_json["subHeader"], true_json["subHeader"])
    else:
        subheader_accuracy = 0

    question_key_accuracy = {}
    for question in question_accuracy:
        for key, acc in question.items():
            if key not in question_key_accuracy:
                question_key_accuracy[key] = [acc]
            else:
                question_key_accuracy[key].append(acc)

    for key, acc_list in question_key_accuracy.items():
        question_key_accuracy[key] = sum(acc_list) / len(acc_list)
    # print(question_accuracy)

    results = {
        "header_accuracy": header_accuracy,
        "subheader_accuracy": subheader_accuracy,
        "question_key_accuracy": question_key_accuracy
    }
    return results

def main():
    ground_truths_file = "phi_2_ground_truths.txt"
    predictions_file = "phi_2_predictions.txt"
    ground_truths = []
    with open(ground_truths_file, 'r') as f:
        ground_truths = f.readlines()
    ground_truths = [x.strip() for x in ground_truths]
    
    predictions = []
    with open(predictions_file, 'r') as f:
        predictions = f.readlines()
    predictions = [x.strip() for x in predictions]
    print(len(ground_truths), len(predictions))

if __name__ == "__main__":
    main()
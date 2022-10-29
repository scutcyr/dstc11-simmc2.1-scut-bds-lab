#!/usr/bin/env python3

import nltk
import json
import argparse
import numpy as np
from evaluate_dst import evaluate_from_flat_list
from response_evaluation import evaluate_response_generation
import pdb
import re
import sys


SYSTEM_TOK = "System :"


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize."""
    return nltk.tokenize.word_tokenize(sentence.lower())


def parse_response_from_file(input_path):
    """Parses the response from a flattened file.
    Args:
        input_path: Path to read the responses from.
    """
    lines = []
    with open(input_path, "r") as file_id:
        for raw in file_id.readlines():
            items = raw.split("\t")
            # 模型预测回复
            if len(items) == 4:
                text = items[0]
            else:
                text = items[2]

            if SYSTEM_TOK in text:
                split_line = text.split(SYSTEM_TOK, 1)
            else:
                split_line = text

            # pdb.set_trace()
            ## 为了与原始数据格式保持一致
            lines.append((split_line[0].strip("\n"), split_line[1].strip("\n").strip("<EOS>")))
    return lines


def parse_flattened_results_from_file(path):
    results = []
    with open(path, "r") as f_in:
        for line in f_in:
            parsed = parse_flattened_result(line)
            results.append(parsed)

    return results


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    items = to_parse.split("\t")
    if len(items) == 4:
        coref_objs, disamb_objs, dst = items[1], items[2], items[3]
    else:
        coref_objs, disamb_objs, dst = items[5], items[4], items[1]
    # pdb.set_trace()
    coref_objs = [int(o.strip().strip("<").strip(">")) for o in json.loads(coref_objs) if o != ""]
    disamb_objs = [int(o.strip().strip("<").strip(">")) for o in json.loads(disamb_objs) if o != ""]
    

    dst_dict = json.loads(dst)
    belief = {
        "act": dst_dict["act"],
        "slots": [],
        "request_slots": dst_dict["act_attributes"]["request_slots"],
        "objects": coref_objs,
        "disambiguation_candidates": disamb_objs
    }
    for key, value in dst_dict["act_attributes"]["slot_values"].items():
        belief["slots"].append([key, value])

    return [belief]


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ref_path", help="path for target, line-separated format (.txt)")
    parser.add_argument("--input_test_path", help="path for model prediction output, line-separated format (.txt)")
    parser.add_argument("--data_json_path", default="../data/simmc2_dials_dstc10_devtest.json", help="Data with .json format gold responses")
    parser.add_argument("--output_json_response_path", default=None, help="Responses generated by the model")
    parser.add_argument("--output_path_report", help="path for saving evaluation summary (.json)")
    parser.add_argument("--dialog_meta_data", type=str, default='../data_object_special/simmc2_dials_dstc10_devtest_inference_disambiguation.json')
    parser.add_argument("--single_round_evaluation", action="store_true", default=False, help="Single round evaluation for hidden split")
    args = parser.parse_args()

    # pdb.set_trace()
    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(args.input_ref_path)
    list_predicted = parse_flattened_results_from_file(args.input_test_path)

    # pdb.set_trace()

    # Evaluate Subtask 1 ~ Subtask 3
    report = evaluate_from_flat_list(list_target, list_predicted)
    print(report)

    # Evaluate Subtask 4
    if args.single_round_evaluation:
        
        with open(args.data_json_path, "r") as file_id:
            gt_responses = json.load(file_id)

        #print(gt_responses)
        #print(gt_responses[0])

        dialog_meta_data = json.load(open(args.dialog_meta_data)) # List[Dict]

        predicted_response = [] 

        with open(args.input_test_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == len(dialog_meta_data)
            for line, meta in zip(lines, dialog_meta_data):
                response = line.split("<EOB>")[1].split("<EOS>")[0].strip()
                predicted_response.append({
                    "dialog_id" : meta["dialog_id"],
                    "predictions" : [{
                        "turn_id" : meta["turn_id"],
                        "response" : response
                    }]
                })
        if args.output_json_response_path:
            json.dump(predicted_response, open(args.output_json_response_path, "w"), indent=4)


        bleu_score, bleu_std_err = evaluate_response_generation(
            gt_responses, predicted_response, args.single_round_evaluation
        )
        print(f"BLEU Score: {bleu_score} +- {bleu_std_err}")

        report["bleu"] = bleu_score
        report["bleu_stderr"] = bleu_std_err
    else:
        # Convert the data from the model friendly format to JSON
        list_target = parse_response_from_file(args.input_ref_path)
        list_predicted = parse_response_from_file(args.input_test_path)
        # Compute BLEU scores.
        bleu_scores = []
        # Smoothing function.
        chencherry = nltk.translate.bleu_score.SmoothingFunction()

        for response, gt_response in zip(list_predicted, list_target):
            #print("预测回复：", response[0])
            #print("真实回复：", gt_response[0])
            #assert response[0] == gt_response[0], "Input contexts do not match!"
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(gt_response[1])],
                normalize_sentence(response[1]),
                smoothing_function=chencherry.method7,
            )
            bleu_scores.append(bleu_score)
        mean_bleu_scores = np.mean(bleu_scores)
        mean_bleu_scores_std = np.std(bleu_scores) / np.sqrt(len(bleu_scores))

        report["bleu"] = mean_bleu_scores
        report["bleu_stderr"] = mean_bleu_scores_std

        print("BLEU score: {} +- {}".format(mean_bleu_scores, mean_bleu_scores_std))

    # Save report
    with open(args.output_path_report, "w") as f_out:
        f_out.write("%s\n" % json.dumps(report, indent=2))
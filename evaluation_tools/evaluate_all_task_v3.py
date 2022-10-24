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


START_BELIEF_STATE = "<SOB>"
END_BELIEF_STATE = "<EOB>"
SYSTEM_TOK = "<SYSTEM>"


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
            if len(items) == 3:
                text = items[0]
            else:
                text = items[1]
            if SYSTEM_TOK in text:
                split_line = text.split(SYSTEM_TOK, 1)
            else:
                split_line = text.split(END_BELIEF_STATE, 1)
            if len(split_line)==1:
                lines.append((split_line[0].strip("\n"), "Sorry, I don't understand what you mean."))
            else:
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
    if len(items) == 3:
        to_parse, coref_objs, disamb_objs = items
    else:
        to_parse, coref_objs, disamb_objs = items[1], items[4], items[3]
    # pdb.set_trace()
    coref_objs = [int(o.strip().strip("<").strip(">")) for o in json.loads(coref_objs) if o != ""]
    disamb_objs = [int(o.strip().strip("<").strip(">")) for o in json.loads(disamb_objs) if o != ""]
    
    dialog_act_regex = re.compile(
        r'([\w:?.?]*)  *\[(.*)\] *\(([^\]]*)\)'
    )    
    
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= (\[(.*)\]|[^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")

    belief = []

    # Parse
    # pdb.set_trace()
    splits = to_parse.strip().split(START_BELIEF_STATE)
    if len(splits) == 2:
        to_parse = splits[1].strip()
    else:
        to_parse = splits[0].strip()
    splits = to_parse.split(END_BELIEF_STATE)

    if len(splits) == 2:
        # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
        to_parse = splits[0].strip()

        for dialog_act in dialog_act_regex.finditer(to_parse):
            d = {
                "act": dialog_act.group(1),
                "slots": [],
                "request_slots": [],
                "objects": coref_objs,
                "disambiguation_candidates": disamb_objs,
            }

            for slot in slot_regex.finditer(dialog_act.group(2)):
                d["slots"].append([slot.group(1).strip(), slot.group(2).strip()])

            for request_slot in request_regex.finditer(dialog_act.group(3)):
                d["request_slots"].append(request_slot.group(1).strip())

            if d != {}:
                belief.append(d)

    return belief


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
    parser.add_argument("--debug_file", type=str, default="")
    args = parser.parse_args()

    # pdb.set_trace()
    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(args.input_ref_path)
    list_predicted = parse_flattened_results_from_file(args.input_test_path)

    data_refs = []
    with open(args.input_ref_path, "r") as f:
        for line in f.readlines():
            data_refs.append(line.strip())
    data_preds = []
    with open(args.input_test_path, "r") as f:
        for line in f.readlines():
            data_preds.append(line.strip())
    # pdb.set_trace()

    # Evaluate Subtask 1 ~ Subtask 3
    report = evaluate_from_flat_list(list_target, list_predicted)
    print(report)

    # Evaluate Subtask 4
    if args.single_round_evaluation:
        with open(args.data_json_path, "r") as file_id:
            dialogs = json.load(file_id)
        
        total_idx = 0
        last_turn_idxs = []
        for dialog in dialogs["dialogue_data"]:
            num_turns = len(dialog["dialogue"])
            for idx, turn in enumerate(dialog["dialogue"]):
                if idx == num_turns - 1:
                    last_turn_idxs.append(total_idx)
                total_idx += 1

    # Convert the data from the model friendly format to JSON
    list_target = parse_response_from_file(args.input_ref_path)
    list_predicted = parse_response_from_file(args.input_test_path)
    # Compute BLEU scores.
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    if args.debug_file != "":
        debug_fr = open(args.debug_file, "w")

    # pdb.set_trace()
    for idx, (response, gt_response) in enumerate(zip(list_predicted, list_target)):
        #print("预测回复：", response[0])
        #print("真实回复：", gt_response[0])
        if args.single_round_evaluation and idx not in last_turn_idxs:
            continue
        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [normalize_sentence(gt_response[1])],
            normalize_sentence(response[1]),
            smoothing_function=chencherry.method7,
        )
        if args.debug_file != "":
            debug_fr.write("%s\t%s\t%s\t%s\t%s\n" % (str(bleu_score), gt_response[1], response[1], data_refs[idx], data_preds[idx]))
        # if bleu_score <= 1.1:
        #     pdb.set_trace()
        bleu_scores.append(bleu_score)
    # pdb.set_trace()
    mean_bleu_scores = np.mean(bleu_scores)
    mean_bleu_scores_std = np.std(bleu_scores) / np.sqrt(len(bleu_scores))

    report["bleu"] = mean_bleu_scores
    report["bleu_stderr"] = mean_bleu_scores_std

    print("BLEU score: {} +- {}".format(mean_bleu_scores, mean_bleu_scores_std))

    debug_fr.close()

    # Save report
    with open(args.output_path_report, "w") as f_out:
        f_out.write("%s\n" % json.dumps(report, indent=2))
        
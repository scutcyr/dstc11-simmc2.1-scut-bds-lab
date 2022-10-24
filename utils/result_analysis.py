# coding=utf-8
# Copyright 2022 iFLYTEK, The State Key Laboratory of Cognitive Intelligence.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" Results Analysis for DSTC-11 SIMMC 2.1

Updated by Yirong Chen 
Used for [SIMMC 2.1](https://github.com/facebookresearch/simmc2)
Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/09/07

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json

import nltk
import numpy as np


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize.
    """
    return nltk.tokenize.word_tokenize(sentence.lower())


def evaluate_response_generation(
    gt_responses, model_responses, single_round_eval=False
):
    """Evaluates response generation using the raw data and model predictions.
    用于分析不同的样本的预测效果
    Args:
        gt_responses: Ground truth responses.
        model_responses: Generated responses.
        single_round_eval: Evaluate only for the last turn.
    """
    gt_responses_pool = {ii["dialogue_idx"]: ii for ii in gt_responses["dialogue_data"]}
    #print(gt_responses_pool)
    bleu_scores = []
    bleu_scores_with_id = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    num_evaluations = 0
    for model_datum in model_responses:
        dialog_id = model_datum["dialog_id"]
        num_gt_rounds = len(gt_responses_pool[dialog_id]["dialogue"])
        for round_datum in model_datum["predictions"]:
            round_id = round_datum["turn_id"]
            # Skip if single_round_eval and this is not the last round.
            if single_round_eval and round_id != num_gt_rounds - 1:
                continue

            response = round_datum["response"]
            gt_datum = gt_responses_pool[dialog_id]["dialogue"][round_id]
            gt_response = gt_datum["system_transcript"]

            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(gt_response)],
                normalize_sentence(response),
                smoothing_function=chencherry.method7,
            )
            temp_dict = {"dialog_id":dialog_id, "turn_id": round_id, "gt_response": gt_response, "pre_response": response, "bleu_score": bleu_score}
            bleu_scores_with_id.append(temp_dict)
            bleu_scores.append(bleu_score)
    print("#Instances evaluated BLEU: {}".format(len(bleu_scores)))
    return bleu_scores, bleu_scores_with_id, np.mean(bleu_scores), np.std(bleu_scores) / np.sqrt(len(bleu_scores))

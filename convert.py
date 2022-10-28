# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Copyright 2022 iFLYTEK, The State Key Laboratory of Cognitive Intelligence. All rights reserved.

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


# File: convert.py
# Description: The dataset preprocessing code
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/10
# Usage:
"""
Updated by Yirong Chen 
Check for SIMMC 2.0
Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/25

2022/08/10 更新
将歧义候选disambiguation candidates 添加到target文件当中


Output Line-by-line File Format:
--context_before_objects or --revert=0: 对象信息在对话上下文后面
    Save Format: <DISAM> User : Can you please help me find a hoodie with good ratings? <SOO><NOCOREF><OBJ><0>[(-0.1045,-0.4989,0.0037,-0.0780,0.0455,1.0000)]<@1167> ... <OBJ><8>[(0.3595,0.0354,0.4989,0.4967,0.0643,0.3403)]<@1006><EOO> => Belief State : 
    BART Encoder Input Format: <s> <DISAM> User : Can you please help me find a hoodie with good ratings? <SOO><NOCOREF><OBJ><0><@1167> ... <OBJ><8><@1006><EOO> => Belief State : </s>
    BART Decoder Output Format: <s> REQUEST:GET [ type = hoodie, customerReview = good ] () <EOB> No problem. How do you like the brown one in the bottom row, second from the left? <EOS></s>
    T5/Blenderbot Encoder Input Format: <DISAM> User : Can you please help me find a hoodie with good ratings? <SOO><NOCOREF><OBJ><0><@1167> ... <OBJ><8><@1006><EOO> => Belief State : </s>
    BART Decoder Output Format: <pad> REQUEST:GET [ type = hoodie, customerReview = good ] () <EOB> No problem. How do you like the brown one in the bottom row, second from the left? <EOS></s>

--objects_before_context or --revert=1: 对象信息在对话上下文前面
    Save Format: <DISAM><SOO><NOCOREF><OBJ><0>[(-0.1045,-0.4989,0.0037,-0.0780,0.0455,1.0000)]<@1167> ... <OBJ><8>[(0.3595,0.0354,0.4989,0.4967,0.0643,0.3403)]<@1006><EOO> User : Can you please help me find a hoodie with good ratings? => Belief State : 
    BART Encoder Input Format: <s><DISAM><SOO><NOCOREF><OBJ><0><@1167> ... <OBJ><8><@1006><EOO> User : Can you please help me find a hoodie with good ratings? => Belief State :</s> 
    BART Decoder Output Format: <s> REQUEST:GET [ type = hoodie, customerReview = good ] () <EOB> No problem. How do you like the brown one in the bottom row, second from the left? <EOS></s>
    T5/Blenderbot Encoder Input Format: <DISAM><SOO><NOCOREF><OBJ><0><@1167> ... <OBJ><8><@1006><EOO> User : Can you please help me find a hoodie with good ratings? => Belief State :</s> 
    BART Decoder Output Format: <pad> REQUEST:GET [ type = hoodie, customerReview = good ] () <EOB> No problem. How do you like the brown one in the bottom row, second from the left? <EOS></s>

Usage:

python convert.py \
  --input_path_json=../data/simmc2.1_dials_dstc11_{train|dev|devtest|teststd}.json \
  --output_path_predict=../data_object_special_context_before_objects/simmc2.1_dials_dstc11_{train|dev|devtest|teststd}_predict.txt \
  --output_path_target=../data_object_special_context_before_objects/simmc2.1_dials_dstc11_{train|dev|devtest|teststd}_target.txt \
  --object_special_token_item2id=../data/item2id.json \
  --scene_json_folder=../data/simmc2_scene_jsons_dstc10_public \
  --image_folder=../data/simmc2_scene_images_dstc10_public \
  --len_context=2 \
  --context_before_objects \


"""
import os
import re
import json
import copy
import glob
import argparse
import collections

from functools import partial
from itertools import chain

import imagesize
import numpy as np

from utils import api
from utils.metadata import (
    FASHION_SIZES, FASHION_AVAILABLE_SIZES,
    FASHION_BRAND, FASHION_PRICE,
    FASHION_CUSTOMER_REVIEW, FURNITURE_BRAND,
    FURNITURE_PRICE, FURNITURE_CUSTOMER_RATING
)

from utils.image import get_image_name

from evaluation_tools.convert import parse_flattened_results_from_file

# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "=> Belief State :"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"
START_OF_OBJ_TOKEN = "<SOO>"
END_OF_OBJ_TOKEN = "<EOO>"
OBJ_START = "<OBJ>"
OBJ_PREVI = "<PREVIOBJ>"
DET_START = "<DET>"
NO_COREF = "<NOCOREF>"

available_sizes2st = {
    'XS': '<A>',
    'S': '<B>',
    'M': '<C>',
    'L': '<D>',
    'XL': '<E>',
    'XXL': '<F>' 
}

# If we use each object token as special token
NUM_FASHION_ITEMS = 288
NUM_FURNITURE_ITEMS = 57
MAX_NUM_OBJ_IN_SCENE = 200

TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE} "
TEMPLATE_TARGET = ("{context} {START_BELIEF_STATE} {belief_state} "
                   "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")
TEMPLATE_TARGET_FINAL = ("{context} {START_BELIEF_STATE} {belief_state} "
                         "{END_OF_BELIEF} {response} {END_OF_SENTENCE} {disambig_str}")

TEMPLATE_PREDICT_USE_OBJVEC = "{context} {objvec} {START_BELIEF_STATE} "
TEMPLATE_PREDICT_USE_OBJVEC_SYSSTATE = "{context} {objvec} {sys_state} {START_BELIEF_STATE} "

TEMPLATE_PREDICT_USE_OBJVEC_DET = "{context} {det} {objvec} {START_BELIEF_STATE} "
TEMPLATE_PREDICT_OBJVEC_FIRST = "{objvec} {context} {START_BELIEF_STATE} "
TEMPLATE_PREDICT_OBJVEC_FIRST_SYSSTATE = "{objvec} {context} {sys_state} {START_BELIEF_STATE} "
TEMPLATE_PREDICT_OBJVEC_FIRST_DET = "{det} {objvec} {context} {START_BELIEF_STATE} "
TEMPLATE_FINAL = "{context} {START_BELIEF_STATE} {det} {objvec}"  # seg: 2 0 1 0 1, ... 


# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"


def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def represent_visual_objects_special_token(object_ids, for_belief_state=False):
    # Stringify visual objects (JSON)
    str_objects = ", ".join(["<"+str(o)+">" for o in object_ids])
    if for_belief_state:
        return str_objects
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def arrange_det(scene_json_folder, scene_id):
    det_arrange_list = []
    scene_id_for_img = scene_id[2:] if scene_id.startswith('m_') else scene_id 
    if scene_id_for_img in det_info:
        det_scene = det_info[scene_id_for_img]
        img_w = det_scene['width']
        img_h = det_scene['height']
        
        for det in det_scene['det']:
            x1 = det['rect']['x1']
            y1 = det['rect']['y1']
            x2 = det['rect']['x2']
            y2 = det['rect']['y2']
            label = det['label']
            pos_str = '{}{}[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(DET_START, label, x1/img_w -0.5, y1/img_h -0.5, x2/img_w -0.5, y2/img_h -0.5, (x2-x1)*(y2-y1)/(img_w*img_h))
            det_arrange_list.append(pos_str)
        return ''.join(det_arrange_list)
    else:
        return ''
    

def arrange_object_special_tokens(scene_json_folder, image_folder, scene_ids, object_item2id, insert_bbox_coords):
    arrange_list = []
    scene_loaded_list = []
    obj_dict_possibly_duplicated = dict()
    for scene_id_idx, scene_id in enumerate(scene_ids):
        with open(os.path.join(scene_json_folder, f"{scene_id}_scene.json"), 'r') as f_in:
            scene = json.load(f_in)
        scene_loaded_list.append(scene)
        for obj in scene['scenes'][0]['objects']: 
            obj_dict_possibly_duplicated[obj['index']] = scene_id_idx
    
    num_scene = len(scene_ids)
    for scene_id_idx, scene_id in enumerate(scene_ids):
        scene = scene_loaded_list[scene_id_idx]
        bbox_id = scene_id[2:] if scene_id.startswith('m_') else scene_id 
        with open(os.path.join(scene_json_folder, f"{bbox_id}_bbox.json"), 'r') as f_in:
            bbox = json.load(f_in)
        camera_position = []; camera_dir_vec = []
        for bbox_item in bbox['Items']:
            if bbox_item['name'] == 'camera':
                camera_position = np.array(bbox_item['position'])
            if bbox_item['name'] == 'camera_forward':
                camera_dir_vec = np.array(bbox_item['position'])

        if insert_bbox_coords:
            largest_z_value = 0
            for obj in scene['scenes'][0]['objects']:
                position = np.array(obj['position'])
                obj_displacement = position - camera_position
                theta = np.dot(obj_displacement, camera_dir_vec) / (np.linalg.norm(obj_displacement)*np.linalg.norm(camera_dir_vec))
                largest_z_value = max(np.linalg.norm(obj_displacement) * np.cos(theta), largest_z_value)
        for obj in scene['scenes'][0]['objects']:
            assert obj['index'] in obj_dict_possibly_duplicated, "SOMETHING IS MISSING!"
            if scene_id_idx == obj_dict_possibly_duplicated[obj['index']]:
                if insert_bbox_coords:
                    position = np.array(obj['position'])
                    obj_displacement = position - camera_position
                    theta = np.dot(obj_displacement, camera_dir_vec) / (np.linalg.norm(obj_displacement)*np.linalg.norm(camera_dir_vec))
                    z_value = np.linalg.norm(obj_displacement) * np.cos(theta)
                    
                    # image name 
                    image_id = None
                    if "m" in scene_id[0]: image_id = scene_id[2:]
                    else: image_id = scene_id
                    image_file_name = os.path.join(image_folder, image_id+".png")
                    if os.path.exists(image_file_name):
                        img_w, img_h = imagesize.get(image_file_name)
                        x1, y1, h, w = obj['bbox']
                        x2, y2 = x1 + w, y1 + h
                        pos_str = '[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(x1/img_w -0.5, y1/img_h -0.5, x2/img_w -0.5, y2/img_h -0.5, (x2-x1)*(y2-y1)/(img_w*img_h), z_value/largest_z_value)
                    else:
                        print(f'{scene_id} is not present in img_size!!!')
                        pos_str = '[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(0.0, 0.0, 0.0, 0.0, 0.0, z_value/largest_z_value)
                else:
                    pos_str = ''

                if (num_scene != 1) and (scene_id_idx == 0): 
                    arrange_list.append(OBJ_PREVI + "<" + str(obj['index']) + ">" + pos_str + object_item2id[obj['prefab_path']])
                else: 
                    arrange_list.append(OBJ_START + "<" + str(obj['index']) + ">" + pos_str + object_item2id[obj['prefab_path']])
    return ''.join(arrange_list)


def get_scene_id(scene_ids, this_turn, so_far=False):
    """
        scene_ids: dict, whose keys are dialogue turn idx and values are scene_id
        this_turn: int, of current dialogue turn idx
    """
    od = collections.OrderedDict(
        sorted(scene_ids.items(), key=lambda t: int(t[0])))
    od_list = list(od.items())
    idx_scene = [(int(idx), scene_id) for idx, scene_id in od_list]
    
    if so_far:
        return list([x[1] for x in idx_scene if x[0] <= this_turn])

    for i in range(len(idx_scene)):
        if idx_scene[i][0] <= this_turn:
            this_turn_scene_id = idx_scene[i][1]
    return this_turn_scene_id


def format_dialog(dialog,
                  len_context=2,
                  use_multimodal_contexts=True,
                  use_belief_states=True,
                  object_item2id=None,
                  scene_json_folder='',
                  image_folder='',
                  insert_bbox_coords=True,
                  revert=False,
                  with_target=True,
                  use_disambiguation_candidates=True,
                  use_system_transcript_annotated=False):
    scene_ids = dialog["scene_ids"]
    dialog_idx = dialog['dialogue_idx']
    prev_asst_uttr = None
    prev_turn = None
    lst_context = []

    for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):

        user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()        
        
        if with_target:
            user_belief = turn[FIELDNAME_BELIEF_STATE]
        
        if "system_transcript" in turn:
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()
        else:
            # print(f"Diag ID : {dialog_idx}, turn_id :{turn_idx}")
            asst_uttr = ''
        
        this_turn_scene_id = get_scene_id(scene_ids, turn_idx)
        scene_ids_so_far = get_scene_id(scene_ids, turn_idx, so_far=True)
        # Format main input context
        context = ""
        if prev_asst_uttr:
            context += f"System : {prev_asst_uttr} "
            if use_multimodal_contexts:
                # Add multimodal contexts
                visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                    "act_attributes"]["objects"]

                if object_item2id is not None:
                    context += represent_visual_objects_special_token(visual_objects, for_belief_state=False) + " "
                else:
                    context += represent_visual_objects(visual_objects) + " "

        context += f"User : {user_uttr}"
        prev_asst_uttr = asst_uttr
        prev_turn = turn

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-len_context:])
                
        if object_item2id is not None:
            object_token_arranged = arrange_object_special_tokens(scene_json_folder, image_folder, scene_ids_so_far, object_item2id, insert_bbox_coords)
            obj_token_str = START_OF_OBJ_TOKEN + NO_COREF + object_token_arranged + END_OF_OBJ_TOKEN
        
        # Format System belief state as Input
        # Code from Taowang on 2022/10/17
        if use_system_transcript_annotated:
            ## 处理生成task4所需要的系统信息，这是可以用的
            sys_state = turn[FIELDNAME_SYSTEM_STATE]
            if "act" in sys_state:
                sys_act = sys_state["act"]
            else:
                sys_act = ""
            
            if "slot_values" in sys_state["act_attributes"]:
                sys_slot_values = sys_state["act_attributes"]["slot_values"]
            else:
                sys_slot_values = {}

            if "request_slots" in sys_state["act_attributes"]:
                sys_request_slots = sys_state["act_attributes"]["request_slots"]
            else:
                sys_request_slots = {}

            # sys_objects = sys_state["act_attributes"]["objects"]
            sys_request_slots_str = ", ".join(sys_request_slots)

            sys_objects_dict = {}
            sys_slot_values_list = []
            for key, value in sys_slot_values.items():
                if "Object ID" in key:
                    obj_id = "<%s>" % key.replace("Object ID: ", "").strip()
                    sys_objects_dict[obj_id] = ", ".join(f"{k.strip()} = {str(v).strip()}" if k != 'availableSizes' 
                                                    else "{} = {}".format(k.strip(), str(v).replace("'", "").strip())
                                                    for k, v in value.items())
                elif key == "availableSizes":
                    sys_slot_values_list.append( "{} = {}".format(key.strip(), str(value).replace("'", "").strip()) )
                else:
                    sys_slot_values_list.append( f"{key.strip()} = {str(value).strip()}" )
            sys_slot_values_str = ", ".join(sys_slot_values_list)
            sys_objects_str = " ".join("{} {}".format(k, v) for k, v in sys_objects_dict.items())
            sys_state_merge = f"{sys_act} [ {sys_slot_values_str} ] ( {sys_request_slots_str} ) {sys_objects_str}"

        # Format belief state
        if use_belief_states:
            if with_target:
                if object_item2id is not None:
                    belief_state = []
                    act = user_belief["act"].strip()
                    slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}" if k!='availableSizes' else '{} = {}'.format(k.strip(), str([available_sizes2st[x] for x in v]).replace("'", "").strip()) 
                    # slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}" if k!='availableSizes' else f'{k.strip()} = {str([available_sizes2st[x] for x in v]).replace("\'", "").strip()}'
                                            for k, v in user_belief["act_attributes"]
                                            ["slot_values"].items())
                    request_slots = ", ".join(
                        user_belief["act_attributes"]["request_slots"])
                    objects_str = represent_visual_objects_special_token(user_belief["act_attributes"]["objects"], for_belief_state=True)
                    # for bs_per_frame in user_belief:
                    # 在此处增加歧义候选识别任务
                    # Updated by Yirong Chen on 2022/08/10
                    #disambiguation_candidates = user_belief["disambiguation_candidates"] # 一个列表
                    disambiguation_candidates_str = represent_visual_objects_special_token(user_belief["disambiguation_candidates"], for_belief_state=True)
                    if use_disambiguation_candidates:
                        str_belief_state_per_frame = (
                            f"{act} [ {slot_values} ] ({request_slots}) < {objects_str} > | {disambiguation_candidates_str} |")

                    else:
                        str_belief_state_per_frame = (
                            f"{act} [ {slot_values} ] ({request_slots}) < {objects_str} >")
                    belief_state.append(str_belief_state_per_frame)
                else:
                    belief_state = []
                    act = user_belief["act"].strip()
                    slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}"
                                            for k, v in user_belief["act_attributes"]
                                            ["slot_values"].items())
                    request_slots = ", ".join(
                        user_belief["act_attributes"]["request_slots"])
                    objects = ", ".join(
                        map(str, user_belief["act_attributes"]["objects"]))
                    # for bs_per_frame in user_belief:
                    # 在此处增加歧义候选识别任务
                    # Updated by Yirong Chen on 2022/08/10
                    disambiguation_candidates = ", ".join(
                        map(str, user_belief["disambiguation_candidates"]))
                    if use_disambiguation_candidates:
                        str_belief_state_per_frame = (
                            f"{act} [ {slot_values} ] ({request_slots}) < {objects} > | {disambiguation_candidates} |")

                    else:
                        str_belief_state_per_frame = (
                            f"{act} [ {slot_values} ] ({request_slots}) < {objects} >")
                    
                    belief_state.append(str_belief_state_per_frame)

                str_belief_state = " ".join(belief_state)
            
            # Format the main input
            if object_item2id is not None: 

                if not revert:
                    if use_system_transcript_annotated:
                        predict = TEMPLATE_PREDICT_USE_OBJVEC_SYSSTATE.format(
                            context=context,
                            objvec=obj_token_str,
                            sys_state=sys_state_merge,
                            START_BELIEF_STATE=START_BELIEF_STATE
                        )

                    else:
                        predict = TEMPLATE_PREDICT_USE_OBJVEC.format(
                            context=context,
                            objvec=obj_token_str,
                            START_BELIEF_STATE=START_BELIEF_STATE
                        )
                else:
                    if use_system_transcript_annotated:
                        predict = TEMPLATE_PREDICT_OBJVEC_FIRST_SYSSTATE.format(
                            objvec=obj_token_str,
                            context=context,
                            sys_state=sys_state_merge,
                            START_BELIEF_STATE=START_BELIEF_STATE
                        )

                    else:
                        predict = TEMPLATE_PREDICT_OBJVEC_FIRST.format(
                            objvec=obj_token_str,
                            context=context,
                            START_BELIEF_STATE=START_BELIEF_STATE
                        )
            else:
                predict = TEMPLATE_PREDICT.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                )

            if with_target:
                # Format the main output
                target = TEMPLATE_TARGET.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                    belief_state=str_belief_state,
                    END_OF_BELIEF=END_OF_BELIEF,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                )
            else: target=""
        else:
            # Format the main input
            predict = TEMPLATE_PREDICT_NOBELIEF.format(
                context=context, START_OF_RESPONSE=START_OF_RESPONSE)

            if with_target:
                # Format the main output
                target = TEMPLATE_TARGET_NOBELIEF.format(
                    context=context,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                    START_OF_RESPONSE=START_OF_RESPONSE,
                )
            else: target=""
        yield predict, target


def convert_json_to_flattened(
    input_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    object_special_token_item2id="",
    scene_json_folder='',
    image_folder='',
    insert_bbox_coords=True,
    revert=False,
    with_target=True,
    use_system_transcript_annotated=False,
    predict_only_final_turn_system_transcript=False
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """
    if object_special_token_item2id:
        with open(object_special_token_item2id, 'r') as f_in:
            object_item2id = json.load(f_in)
        use_object_special_token = True
    else:
        use_object_special_token = False

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]
    print("-----------------------Start Converting...------------------------")
    _formatter = partial(format_dialog,
                         len_context=len_context,
                         use_multimodal_contexts=use_multimodal_contexts,
                         use_belief_states=use_belief_states,
                         object_item2id=object_item2id,
                         scene_json_folder=scene_json_folder,
                         image_folder=image_folder,
                         insert_bbox_coords=insert_bbox_coords,
                         revert=revert,
                         with_target=with_target,
                         use_disambiguation_candidates=True,
                         use_system_transcript_annotated=use_system_transcript_annotated)
    predicts, targets = zip(*chain.from_iterable(map(_formatter, data)))

    if predict_only_final_turn_system_transcript:
        new_predicts = []
        j = 0
        for i in len(data):
            j = j + len(data[i][FIELDNAME_DIALOG])
            new_predicts.append(predicts[j-1])

        print("-----------------------End Converting...------------------------")
        with open(output_path_predict, "w") as f_predict:
            f_predict.write("\n".join(new_predicts))

        return True



    print("-----------------------End Converting...------------------------")
    # Output into text files
    with open(output_path_predict, "w") as f_predict:
        f_predict.write("\n".join(predicts))

    if with_target:
        with open(output_path_target, "w") as f_target:
            f_target.write("\n".join(targets))


def format_disambiguation_label(dialogue_data):
    # This function is used for SIMMC2.1
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            if "disambiguation_label" in turn["transcript_annotated"]:
                lines.append(turn["transcript_annotated"]["disambiguation_label"])
            else:
                lines.append(-100)
    return lines


def format_disambiguation_label_for_simmc2(dialogue_data):
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            if "disambiguation_label" in turn:
                lines.append(turn["disambiguation_label"])
            else:
                lines.append(-100)
    return lines


def format_system_transcript_to_response(dialogue_data):
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            lines.append("System : "+turn["system_transcript"]+" <EOS>")
    return lines


def format_system_act_from_dialogue_data(dialogue_data):
    # 提取json文件当中的system intent, 用于辅助训练
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            lines.append(turn["system_transcript_annotated"]["act"])
    return lines

def format_user_act_from_dialogue_data(dialogue_data):
    # 提取json文件当中的system intent, 用于辅助训练
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            lines.append(turn["transcript_annotated"]["act"])
    return lines


def format_dialogue_scene_name(dialogue_data):
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            image_name, scene_label = get_image_name(
                        dialog["scene_ids"], turn_idx
                    )
            lines.append(image_name)
    return lines


def format_dialogue_subtask4_inference(dialogue_data, predict_only_final_turn_system_transcript=False):
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        if predict_only_final_turn_system_transcript:
            lines.append(
                    {
                    "dialog_id": dialog["dialogue_idx"],
                    "turn_id": dialog[FIELDNAME_DIALOG][-1]["turn_idx"]
                    }
                )

        else:
            for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
                lines.append(
                        {
                        "dialog_id": dialog["dialogue_idx"],
                        "turn_id": turn["turn_idx"]
                        }
                    )
    return lines


def format_inference_disambiguation(dialogue_data):
    # 适用于SIMMC2.1、SIMMC2.0
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            if "disambiguation_label" in turn["transcript_annotated"]:
                # SIMMC2.1
                lines.append(
                    {
                    "dialog_id": dialog["dialogue_idx"],
                    "turn_id": turn["turn_idx"],
                    "disambiguation_label": turn["transcript_annotated"]["disambiguation_label"]
                    }
                )
            elif "disambiguation_label" in turn:
                # SIMMC2.0
                lines.append(
                    {
                    "dialog_id": dialog["dialogue_idx"],
                    "turn_id": turn["turn_idx"],
                    "disambiguation_label": turn["disambiguation_label"]
                    }
                )
            else:
                lines.append(
                    {
                    "dialog_id": dialog["dialogue_idx"],
                    "turn_id": turn["turn_idx"],
                    "disambiguation_label": -100
                    }
                )
    return lines



def format_multimodal_context(dialogue_data):
    # 适用于SIMMC2.1、SIMMC2.0
    # dialogue_data: [{}, {}, {}, ...]
    lines = []
    for dialog in dialogue_data:
        multimodal_context = []
        for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            lines.append(
                    {
                    "dialog_id": dialog["dialogue_idx"],
                    "turn_id": turn["turn_idx"],
                    "multimodal_context": copy.deepcopy(list(set(multimodal_context)))
                    }
                )

            multimodal_context.extend(turn["system_transcript_annotated"]["act_attributes"]["objects"])

    return lines




def extract_final_turn_predict_from_all_turn_predict(dialogue_data, predict_lines):
    final_turn_predict_lines = []
    line_id = 0
    for dialog in dialogue_data:
        dialog_len = len(dialog[FIELDNAME_DIALOG])
        line_id += dialog_len
        final_turn_predict_lines.append(predict_lines[line_id-1])
    return final_turn_predict_lines


def convert_taowang_result_to_submit_txt_file(input_file, output_file):
    '''
    input_file is a line-by-line file, split the subtask result with tab (	), 
    with format:
        generation result \t coref result \t disamb result
    looks like:
        REQUEST:COMPARE [  ] ( price, customerReview ) <EOB> The one in the cubicle is $179.99 with a 3.8 rating and the one on the rack is $64.99 with a 3.5 stars rating. <EOS>	["<23>", "<16>"]	[]
        ...

    output_file is also a line-by-line file, looks like:
        => Belief State : REQUEST:COMPARE [  ] ( price, customerReview ) < 23, 16 > |  | <EOB> The one in the cubicle is $179.99 with a 3.8 rating and the one on the rack is $64.99 with a 3.5 stars rating. <EOS>
        

    '''

    belief_state = "=> Belief State : "
    object_regex = re.compile(r"([A-Za-z0-9]+)")
    disamb_candidate_regex = re.compile(r"([A-Za-z0-9]+)")

    lines = []
    with open(input_file, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if (len(line) > 0 and not line.isspace()):
                lines.append(line)

    split_lines = [line.split("\t") for line in lines]

    format_lines = []
    for line_list in split_lines:
        text = line_list[0]
        pos_start, pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]
        # 格式化任务2的结果为 "< int, int, ... >"
        coref_str = line_list[1]
        coref_object_list = []
        for object_id in object_regex.finditer(coref_str):
            str_object_id = object_id.group(1).strip()
            int_object_id = int(str_object_id)
            coref_object_list.append(int_object_id)

        coref_object_list = sorted(coref_object_list)
        coref_str = str(coref_object_list).replace('[', '< ').replace(']',' >') if coref_object_list else '<  >'
        # 格式化任务1的结果为 "| int, int, ... |"
        disam_str = line_list[2]
        disam_object_list = []
        for object_id in object_regex.finditer(disam_str):
            str_object_id = object_id.group(1).strip()
            int_object_id = int(str_object_id)
            disam_object_list.append(int_object_id)

        disam_object_list = sorted(disam_object_list)
        disam_str = str(disam_object_list).replace('[', '| ').replace(']',' |') if disam_object_list else '|  |'

        format_line = belief_state + text[:pos_start+1] + ' ' + coref_str + ' ' + disam_str + ' <EOB>' + text[pos_end:]
        format_lines.append(format_line)

    with open(output_file, "w") as f_out:
        f_out.write("\n".join(format_lines))


def convert_taowang_result_to_submit_txt_file_from_dir(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    input_file_path_list = [path for path in sorted(glob.glob(os.path.join(input_dir+"/**/", "*.out.proc"), recursive=True))]

    for input_file in input_file_path_list:
        epoch_name = input_file.split("/")[-2]
        output_file = os.path.join(output_dir, "pred-results-of-"+epoch_name+".txt")
        convert_taowang_result_to_submit_txt_file(input_file, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="the directory of the SIMMC Dataset.")
    parser.add_argument('--input_path_json', type=str, help="the path of simmc2.1_dials_dstc11_train|dev|devtest.json file.")
    parser.add_argument('--output_path_predict', type=str) # .txt file
    parser.add_argument('--output_path_target', type=str) # .txt file
    parser.add_argument('--len_context', default=2, type=int)
    parser.add_argument('--use_multimodal_contexts', type=int, default=1)
    parser.add_argument('--use_belief_states', type=int, default=1)
    parser.add_argument('--object_special_token_item2id', type=str)
    parser.add_argument('--scene_json_folder', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--insert_bbox_coords', type=int, default=1)
    parser.add_argument('--revert', type=int, default=0)
    parser.add_argument("--context_before_objects", action="store_true", help="context_before_objects")
    parser.add_argument("--objects_before_context", action="store_true", help="objects_before_context")
    parser.add_argument("--use_system_transcript_annotated", action="store_true", help="use_system_transcript_annotated")
    parser.add_argument("--predict_only_final_turn_system_transcript", action="store_true", help="predict_only_final_turn_system_transcript")
    parser.add_argument('--with_target', type=int, default=1)
    parser.add_argument('--output_disambiguation_label', type=str, default=None) # .txt file
    parser.add_argument('--output_path_response', type=str, default=None) # .txt file
    parser.add_argument('--output_path_scene_name', type=str, default=None) # .txt file
    parser.add_argument('--output_path_system_act', type=str, default=None) # .txt file
    parser.add_argument('--output_path_user_act', type=str, default=None) # .txt file
    parser.add_argument('--output_inference_json', type=str, default=None) # .json file
    parser.add_argument('--output_inference_disambiguation', type=str, default=None) # .json file
    parser.add_argument('--input_path_all_turn_predict_lines', type=str, default=None) # .txt file
    parser.add_argument('--output_path_final_turn_predict_lines', type=str, default=None) # .txt file
    parser.add_argument('--output_multimodal_context_json', type=str, default=None) # .json file

    parser.add_argument('--input_line_by_line_txt_file', type=str, default=None) # .txt file
    parser.add_argument('--output_parse_flattened_results_json_file', type=str, default=None) # .json file

    parser.add_argument('--input_line_by_line_out_proc_file', type=str, default=None) # 涛哥那边输出的模型预测结果文件.out.proc
    parser.add_argument('--output_line_by_line_txt_file', type=str, default=None) # 涛哥那边输出的模型预测结果文件.out.proc

    parser.add_argument('--input_line_by_line_out_proc_file_dir', type=str, default=None) # 涛哥那边输出的模型预测结果文件.out.proc
    parser.add_argument('--output_line_by_line_txt_file_dir', type=str, default=None) # 涛哥那边输出的模型预测结果文件.out.proc



    args = parser.parse_args()

    if args.context_before_objects and args.objects_before_context:
        # 同时设置为True，则有冲突
        raise ValueError(
                "Used --context_before_objects and --objects_before_context is not allowed!"
        )
    elif args.context_before_objects:
        args.revert = 0
    elif args.objects_before_context:
        args.revert = 1

    print(args)


    # 将涛哥那边的文件转换为标准输出
    if args.input_line_by_line_out_proc_file is not None and args.output_line_by_line_txt_file is not None:
        convert_taowang_result_to_submit_txt_file(input_file=args.input_line_by_line_out_proc_file, output_file=args.output_line_by_line_txt_file)

    if args.input_line_by_line_out_proc_file_dir is not None and args.output_line_by_line_txt_file_dir is not None:
        convert_taowang_result_to_submit_txt_file_from_dir(input_dir=args.input_line_by_line_out_proc_file_dir,
                                                           output_dir=args.output_line_by_line_txt_file_dir)



    if args.input_path_json is not None and args.input_line_by_line_txt_file is not None and args.output_parse_flattened_results_json_file is not None:
        with open(args.input_path_json, "r") as f_in:
            json_predicted = json.load(f_in)
        
        parse_flattened_results = parse_flattened_results_from_file(args.input_line_by_line_txt_file)

        with open(args.input_line_by_line_txt_file, 'r') as f:
            lines = f.readlines()

        i = 0
        for dialog in json_predicted['dialogue_data']:
            for turn_idx, turn in enumerate(dialog['dialogue']):
                #print(i, ' ', parse_flattened_results[i])
                if len(parse_flattened_results[i]) == 0:
                    print("当前异常的行位置：", i+1)
                    print("当前异常的输出：", lines[i])

                    turn["transcript_annotated"] = {
                        "act": "Unknown",
                        "act_attributes": {
                            "slot_values": {},
                            "request_slots": [],
                            "objects": []
                        },
                        "disambiguation_candidates": []
                    }
                    i = i+1

                else:
                    turn["transcript_annotated"] = {
                        "act": parse_flattened_results[i][0]['act'],
                        "act_attributes": {
                            "slot_values": dict(parse_flattened_results[i][0]['slots']),
                            "request_slots": parse_flattened_results[i][0]['request_slots'],
                            "objects": parse_flattened_results[i][0]['objects']
                        },
                        "disambiguation_candidates": parse_flattened_results[i][0]['disambiguation_candidates']
                    }
                    i = i+1

        with open(args.output_parse_flattened_results_json_file, "w") as json_file:
            json.dump(json_predicted, json_file, indent=4)

    if args.output_disambiguation_label is not None:
        # 创建simmc2.1_dials_dstc11_train_disambiguation_label.txt文件
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)
        #$print(data[0])
        print("Data version is: ", json_data["version"])
        if json_data["version"]=="simmc_2.1_dstc11":
            # 对应于SIMMC2.1版本
            # "disambiguation_label"在"transcript_annotated"里面
            lines = format_disambiguation_label(json_data["dialogue_data"])
        elif json_data["version"]=="simmc_2_dstc10":
            # 对应于SIMMC2.0版本
            # "disambiguation_label"在"transcript_annotated"外面
            lines = format_disambiguation_label_for_simmc2(json_data["dialogue_data"])

        print(lines)
        if os.path.exists(args.output_disambiguation_label):
            os.remove(args.output_disambiguation_label)
        with open(args.output_disambiguation_label, 'w') as f:
            for line in lines:
                f.write(str(line)+'\n')

    if args.output_path_response is not None:
        # 创建response文件
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)
        #$print(data[0])
        print("Data version is: ", json_data["version"])
        lines = format_system_transcript_to_response(json_data["dialogue_data"])
        if os.path.exists(args.output_path_response):
            os.remove(args.output_path_response)
        with open(args.output_path_response, 'w') as f:
            for i in range(len(lines)):
                if i < len(lines)-1:
                    f.write(str(lines[i])+'\n')
                else:
                    f.write(str(lines[i]))
    
    if args.output_path_scene_name is not None:
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)
        #$print(data[0])
        print("Data version is: ", json_data["version"])
        lines = format_dialogue_scene_name(json_data["dialogue_data"])
        if os.path.exists(args.output_path_scene_name):
            os.remove(args.output_path_scene_name)
        with open(args.output_path_scene_name, 'w') as f:
            for i in range(len(lines)):
                if i < len(lines)-1:
                    f.write(str(lines[i])+'\n')
                else:
                    f.write(str(lines[i]))

    if args.output_path_system_act is not None:
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)
        #$print(data[0])
        print("Data version is: ", json_data["version"])
        lines = format_system_act_from_dialogue_data(json_data["dialogue_data"])
        if os.path.exists(args.output_path_system_act):
            os.remove(args.output_path_system_act)
        with open(args.output_path_system_act, 'w') as f:
            for i in range(len(lines)):
                if i < len(lines)-1:
                    f.write(str(lines[i])+'\n')
                else:
                    f.write(str(lines[i]))


    if args.output_path_user_act is not None:
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)
        #$print(data[0])
        print("Data version is: ", json_data["version"])
        lines = format_user_act_from_dialogue_data(json_data["dialogue_data"])
        if os.path.exists(args.output_path_user_act):
            os.remove(args.output_path_user_act)
        with open(args.output_path_user_act, 'w') as f:
            for i in range(len(lines)):
                if i < len(lines)-1:
                    f.write(str(lines[i])+'\n')
                else:
                    f.write(str(lines[i]))

    if args.output_multimodal_context_json is not None:
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)

        print("Data version is: ", json_data["version"])
        lines = format_multimodal_context(json_data["dialogue_data"])
        with open(args.output_multimodal_context_json, "w") as json_file:
            json.dump(lines, json_file, indent=4)






    if args.output_inference_json is not None:
        # 创建inference_disambiguation文件用于评估
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)

        print("Data version is: ", json_data["version"])
        lines = format_dialogue_subtask4_inference(json_data["dialogue_data"], predict_only_final_turn_system_transcript=args.predict_only_final_turn_system_transcript)
        with open(args.output_inference_json, "w") as json_file:
            json.dump(lines, json_file, indent=4)

    if args.output_inference_disambiguation is not None:
        # 创建inference_disambiguation文件用于评估
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)

        print("Data version is: ", json_data["version"])
        lines = format_inference_disambiguation(json_data["dialogue_data"])
        with open(args.output_inference_disambiguation, "w") as json_file:
            json.dump(lines, json_file, indent=4)

    if args.input_path_all_turn_predict_lines is not None and args.output_path_final_turn_predict_lines is not None:
        with open(args.input_path_json, "r") as f_in:
            json_data = json.load(f_in)
        
        predict_lines = []
        all_turn_predict_lines =  open(args.input_path_all_turn_predict_lines, encoding="utf-8")
        for line in all_turn_predict_lines.read().splitlines():
            if (len(line) > 0 and not line.isspace()):
                predict_lines.append(line)
        
        final_turn_predict_lines = extract_final_turn_predict_from_all_turn_predict(json_data["dialogue_data"], predict_lines)

        with open(args.output_path_final_turn_predict_lines, "w") as f:
            f.write("\n".join(final_turn_predict_lines))


    if args.output_path_predict is not None and args.output_path_target is not None:
        convert_json_to_flattened(
            args.input_path_json,
            args.output_path_predict,
            args.output_path_target,
            len_context=args.len_context,
            use_multimodal_contexts=args.use_multimodal_contexts,
            use_belief_states=args.use_belief_states,
            object_special_token_item2id=args.object_special_token_item2id,
            scene_json_folder=args.scene_json_folder,
            image_folder=args.image_folder,
            insert_bbox_coords=args.insert_bbox_coords,
            revert=args.revert,
            with_target=args.with_target,
            use_system_transcript_annotated=args.use_system_transcript_annotated,
            predict_only_final_turn_system_transcript=args.predict_only_final_turn_system_transcript)
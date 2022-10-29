#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.
    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).
    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import os
import re
import json
import argparse
import collections
import pdb
import ast
import imagesize
import numpy as np

from utils import api
from utils.simmc21_dataset import available_sizes2st


# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
BELIEF_STATE = "<DST>"
START_OF_OBJ_TOKEN = "<SOO>"
END_OF_OBJ_TOKEN = "<EOO>"
OBJ_CURR = "<OBJ>"
OBJ_PREVI = "<PREVIOBJ>"
NO_COREF = "<NOCOREF>"
USER_TOK = "User :"
SYSTEM_TOK = "System :"
END_OF_SENTENCE = "<EOS>"

# ACT_TOK = "<act>"
REQ_SLOTS_TOK = "<req_slots>"
SLOT_TYPE_TOK = "<type>"
SLOT_PRICE_TOK = "<price>"
SLOT_CUSTOMETR_REVIEW_TOK = "<customerReview>"
SLOT_BRAND_TOK = "<brand>"
SLOT_SIZE_TOK = "<size>"
SLOT_PATTERN_TOK = "<pattern>"
SLOT_COLOR_TOK = "<color>"
SLOT_SLEEVE_LENGTH_TOK = "<sleeveLength>"
SLOT_AVAILABLE_SIZES_TOK = "<availableSizes>"
SLOT_MATERIALS_TOK = "<materials>"
SLOT_CUSTOMETR_RATING_TOK = "<customerRating>"


# TEMPLATE_PREDICT_USE_OBJVEC = "{context} {objvec} {BELIEF_STATE}"
TEMPLATE_PREDICT_USE_OBJVEC = "{context} {objvec} %s" % "".join(
                            [BELIEF_STATE, REQ_SLOTS_TOK, SLOT_TYPE_TOK, SLOT_PRICE_TOK, SLOT_CUSTOMETR_REVIEW_TOK,
                            SLOT_BRAND_TOK, SLOT_SIZE_TOK, SLOT_PATTERN_TOK, SLOT_COLOR_TOK, SLOT_SLEEVE_LENGTH_TOK,
                            SLOT_AVAILABLE_SIZES_TOK, SLOT_MATERIALS_TOK, SLOT_CUSTOMETR_RATING_TOK])
TEMPLATE_PREDICT_OBJVEC_FIRST = "{objvec} {context} {BELIEF_STATE}"
TEMPLATE_TARGET = "{SYSTEM_TOK} {response} {END_OF_SENTENCE}"


prompt_api = api.PromptAPI(dial_split="train",
                           data_dir="/ps2/sli/data/data_taowang49/projects/19_dstc11/simmc2.1_solutions/work/simmc2.1-iflytek/data", 
                           dialogue_name_prefix="simmc2.1_dials_dstc11_",
                           jsons_dir_name="simmc2_scene_jsons_dstc10_public",
                           images_dir_name="simmc2_scene_images_dstc10_public")


def represent_visual_objects_special_token(object_ids):
    special_object_ids = ["<"+str(o)+">" for o in object_ids]
    return special_object_ids

    
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
                    arrange_list.append(OBJ_CURR + "<" + str(obj['index']) + ">" + pos_str + object_item2id[obj['prefab_path']])
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


def parse_objs(str_objs):
    str_objs_proc = re.sub(r"\[\([^\)]*\)\]", "", str_objs)
    obj_boxes = [ast.literal_eval(position.replace('(', '').replace(')', '')) for position in re.findall(r"\[\([^\)]+\)\]", str_objs)]
    return str_objs_proc, obj_boxes


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
                  outfr=None):
    scene_ids = dialog["scene_ids"]
    dialog_idx = dialog['dialogue_idx']
    prev_asst_uttr = None
    prev_turn = None
    lst_context = []

    for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
        user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()        

        if "system_transcript" in turn:
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()
        else:
            # print(f"Diag ID : {dialog_idx}, turn_id : {turn_idx}")
            asst_uttr = ''
        
        this_turn_scene_id = get_scene_id(scene_ids, turn_idx)
        scene_ids_so_far = get_scene_id(scene_ids, turn_idx, so_far=True)
        # Format main input context
        context = ""
        if prev_asst_uttr:
            context += f"{SYSTEM_TOK} {prev_asst_uttr} "
            if use_multimodal_contexts:
                # Add multimodal contexts
                # pdb.set_trace()
                visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE]["act_attributes"]["objects"]
                visual_objects = "".join(represent_visual_objects_special_token(visual_objects))
                context += f"{START_OF_MULTIMODAL_CONTEXTS}{visual_objects}{END_OF_MULTIMODAL_CONTEXTS}" + " "

        context += f"{USER_TOK} {user_uttr}"
        prev_asst_uttr = asst_uttr
        prev_turn = turn

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-len_context:])

        ## 当前对话涉及的 object id
        object_token_arranged = arrange_object_special_tokens(scene_json_folder, image_folder, scene_ids_so_far, object_item2id, insert_bbox_coords)
        obj_token_proc, obj_boxes = parse_objs(object_token_arranged)
        obj_token_str = START_OF_OBJ_TOKEN + NO_COREF + obj_token_proc + END_OF_OBJ_TOKEN

        if with_target:
            ## 当前的对话状态
            user_belief = turn[FIELDNAME_BELIEF_STATE]
            belief_state = []
            act = user_belief["act"].strip()
            slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}" if k!='availableSizes' 
                                    else "{} = {}".format(k.strip(), str([available_sizes2st[x] for x in v]).replace("'", "").strip())
                                    for k, v in user_belief["act_attributes"]["slot_values"].items())
            request_slots = ", ".join(user_belief["act_attributes"]["request_slots"])
            str_belief_state_per_frame = f"{act} [ {slot_values} ] ( {request_slots} )"
            belief_state.append(str_belief_state_per_frame)
            str_belief_state = " ".join(belief_state)

            ## 歧义候选识别和指代消解任务
            disam_cands = represent_visual_objects_special_token(user_belief["disambiguation_candidates"])
            coref_objs = represent_visual_objects_special_token(user_belief["act_attributes"]["objects"])
            if "disambiguation_label" in turn["transcript_annotated"]:
                is_disam = turn["transcript_annotated"]["disambiguation_label"]
            else:
                is_disam = -100
        else:
            disam_cands = []
            coref_objs = []
            is_disam = -100
            user_belief = {}
        
        if not revert:
            predict = TEMPLATE_PREDICT_USE_OBJVEC.format(
                context=context,
                objvec=obj_token_str
            )
        else:
            predict = TEMPLATE_PREDICT_OBJVEC_FIRST.format(
                objvec=obj_token_str,
                context=context,
                BELIEF_STATE=BELIEF_STATE
            )

        if with_target:
            target = TEMPLATE_TARGET.format(
                SYSTEM_TOK=SYSTEM_TOK,
                response=asst_uttr,
                END_OF_SENTENCE=END_OF_SENTENCE,
            )
        else:
            target = ""

        f_out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (predict, json.dumps(user_belief), target, json.dumps(obj_boxes), 
                                                json.dumps(disam_cands), json.dumps(coref_objs), is_disam))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="the directory of the SIMMC Dataset.")
    parser.add_argument('--input_path_json', type=str, help="the path of simmc2.1_dials_dstc11_train|dev|devtest.json file.")
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--len_context', default=2, type=int)
    parser.add_argument('--use_multimodal_contexts', type=int, default=1)
    parser.add_argument('--use_belief_states', type=int, default=1)
    parser.add_argument('--object_special_token_item2id', type=str)
    parser.add_argument('--scene_json_folder', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--insert_bbox_coords', type=int, default=1)
    parser.add_argument('--revert', type=int, default=0)
    parser.add_argument('--with_target', type=int, default=1)
    args = parser.parse_args()
    print(args)

    if args.object_special_token_item2id:
        with open(args.object_special_token_item2id, 'r') as f_in:
            object_item2id = json.load(f_in)
        use_object_special_token = True
    else:
        object_item2id = None
        use_object_special_token = False

    with open(args.input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]
    
    with open(args.output_path, "w") as f_out:
        for dialog in data:
            format_dialog(dialog,
                        len_context=args.len_context,
                        use_multimodal_contexts=args.use_multimodal_contexts,
                        use_belief_states=args.use_belief_states,
                        object_item2id=object_item2id,
                        scene_json_folder=args.scene_json_folder,
                        image_folder=args.image_folder,
                        insert_bbox_coords=args.insert_bbox_coords,
                        revert=args.revert,
                        with_target=args.with_target,
                        use_disambiguation_candidates=True,
                        outfr=f_out)

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

"""
Updated by Yirong Chen 
Check for SIMMC 2.1
Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/01
说明：由于SIMMC2.0或者2.1数据集存放在其他路径，而不是存放在本代码项目下面的路径，因此需要直接指定DATA_DIR为数据集的绝对路径

特别地，在SIMMC2.1中，"transcript_annotated"字典当中增加了"disambiguation_label"、"disambiguation_candidates"、
"disambiguation_candidates_raw"三个键！
"""
import json
from os.path import join
from typing import Dict, List

import attr
from attr.validators import instance_of

#from .util import find_data_dir, DATA_DIR_ROOT_PATH

DIALOGUE_ACTS = ("INFORM", "CONFIRM", "REQUEST", "ASK")
ACTIVITIES = ("GET", "DISAMBIGUATE", "REFINE", "ADD_TO_CART", "COMPARE")

#DATA_DIR = find_data_dir('DSTC10-SIMMC')  # give root folder name of simmc2 as argument. Ex) find_data_dir('DSTC10-SIMMC')
#Updated by Yirong Chen 
#DATA_DIR = DATA_DIR_ROOT_PATH
DATA_DIR = '/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/data'

@attr.s
class Action:
    dialogue_act: str = attr.ib(
        converter=lambda x: str(x).upper()
    )
    activity: str = attr.ib(
        converter=lambda x: str(x).upper()
    )

    @staticmethod
    def check_in(attribute, value, listing):
        """Universal checker that validates if value is in the given list."""
        if value not in listing:
            raise ValueError("{} must be one of {}, but received {}.".format(attribute.name, listing, value))
    
    @dialogue_act.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, DIALOGUE_ACTS)

    @activity.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, ACTIVITIES)

@attr.s
class ActionAttributes:
    slot_values: Dict = attr.ib()
    request_slots: List = attr.ib()
    objects: List[int] = attr.ib()

# Origin for SIMMC 2.0
@attr.s
class TranscriptAnnotation:
    act: str = attr.ib()
    act_attributes: ActionAttributes = attr.ib()

    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
    @classmethod
    def annotation_filler(cls, act: str, act_attributes):
        dialogue_act, activity = act.split(':')
        act_args = {
            'dialogue_act': dialogue_act,
            'activity': activity
        }
        act_filled = Action(**act_args)
        act_attributes_filled = ActionAttributes(**act_attributes)
        args = {
            'act': act_filled,
            'act_attributes': act_attributes_filled
        }
        return cls(**args)


# Add for SIMMC 2.1
@attr.s
class SystemTranscriptAnnotation:
    act: str = attr.ib()
    act_attributes: ActionAttributes = attr.ib()

    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
    @classmethod
    def annotation_filler(cls, act: str, act_attributes):
        dialogue_act, activity = act.split(':')
        act_args = {
            'dialogue_act': dialogue_act,
            'activity': activity
        }
        act_filled = Action(**act_args)
        act_attributes_filled = ActionAttributes(**act_attributes)
        args = {
            'act': act_filled,
            'act_attributes': act_attributes_filled
        }
        return cls(**args)

# Add for SIMMC 2.1
@attr.s
class UserTranscriptAnnotation:
    act: str = attr.ib()
    act_attributes: ActionAttributes = attr.ib()
    disambiguation_label: int = attr.ib()
    disambiguation_candidates: List[int] = attr.ib()
    disambiguation_candidates_raw: List[int] = attr.ib()

    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
    @classmethod
    def annotation_filler(cls, 
                          act: str, 
                          act_attributes, 
                          disambiguation_label, 
                          disambiguation_candidates,
                          disambiguation_candidates_raw):
        dialogue_act, activity = act.split(':')
        act_args = {
            'dialogue_act': dialogue_act,
            'activity': activity
        }
        act_filled = Action(**act_args)
        act_attributes_filled = ActionAttributes(**act_attributes)


        args = {
            'act': act_filled,
            'act_attributes': act_attributes_filled,
            'disambiguation_label': disambiguation_label,
            'disambiguation_candidates': disambiguation_candidates,
            'disambiguation_candidates_raw': disambiguation_candidates_raw
        }

        return cls(**args)



@attr.s
class SingleDialogueTurn:
    turn_idx: int = attr.ib()
    system_transcript: str = attr.ib()
    system_transcript_annotated: SystemTranscriptAnnotation = attr.ib()
    transcript: str = attr.ib()
    transcript_annotated: UserTranscriptAnnotation = attr.ib()
    #disambiguation_label = attr.ib(default=None)

    @classmethod
    def single_dialogue_filler(cls, turn_idx, transcript, transcript_annotated,
                               system_transcript, system_transcript_annotated): # , disambiguation_label=None

        transcript_annotated_filled = UserTranscriptAnnotation.annotation_filler(**transcript_annotated)
        system_transcript_annotated_filled = SystemTranscriptAnnotation.annotation_filler(**system_transcript_annotated)
        args = {
            'turn_idx': turn_idx,
            'transcript': transcript,
            'transcript_annotated': transcript_annotated_filled,
            'system_transcript': system_transcript,
            'system_transcript_annotated': system_transcript_annotated_filled #,
            # 'disambiguation_label': disambiguation_label
        } 
        return cls(**args)

@attr.s
class Dialogue:
    dialogue_idx: int = attr.ib(
        validator=instance_of(int)
    )
    domain: str = attr.ib(
        converter=lambda x: str(x).lower(),
        validator=instance_of(str)
    )
    mentioned_object_ids: List[int] = attr.ib(
        converter=lambda x: [int(_) for _ in x],
        validator=instance_of(list)
    )
    scene_ids: Dict[int, str] = attr.ib(
        converter=lambda x: {int(k): str(v) for k,v in x.items()},
        validator=instance_of(dict)
    )
    single_dialogue_list: List[SingleDialogueTurn] = attr.ib()

    @domain.validator
    def check(self, attribute, value):
        if value not in ("fashion", "furniture"):
            raise ValueError("Domain must either be fashion or furniture.")

    @classmethod
    def dialogue_filler(cls, dialogue, dialogue_idx, domain, mentioned_object_ids, scene_ids):
        single_dialogue_list = list()
        for idx, single_dialogue in enumerate(dialogue):
            single_dialogue_list.append(SingleDialogueTurn.single_dialogue_filler(**single_dialogue))
        args = {
            'single_dialogue_list': single_dialogue_list,
            'dialogue_idx': dialogue_idx,
            'domain': domain,
            'mentioned_object_ids': mentioned_object_ids,
            'scene_ids': scene_ids
        }
        dialogue_idx
        return cls(**args)


@attr.s
class AllDialogues:

    dialogue_list: List[Dialogue] = attr.ib()
    dialogue_split: str = attr.ib()
    dialogue_domain: str = attr.ib()

    @classmethod
    def from_json(cls, dialogue_name: str, data_dir=DATA_DIR):
        dialogue_json = json.load(open(join(data_dir, "{}.json".format(dialogue_name))))
        dialogue_list = list()
        dialogue_split = dialogue_json['split']
        dialogue_domain = dialogue_json['domain']
        for idx, dialogue in enumerate(dialogue_json['dialogue_data']):
            dialogue_args = {
                'dialogue': dialogue['dialogue'],
                'dialogue_idx': dialogue['dialogue_idx'],
                'domain': dialogue['domain'],
                'mentioned_object_ids': dialogue['mentioned_object_ids'],
                'scene_ids': dialogue['scene_ids'],
            }
            dialogue_list.append(Dialogue.dialogue_filler(**dialogue_args))

        args = {
            'dialogue_list': dialogue_list,
            'dialogue_split': dialogue_split,
            'dialogue_domain': dialogue_domain,
        }
        return cls(**args)


def main_function(dial_split="train", 
                  data_dir=DATA_DIR, 
                  dialogue_name_prefix="simmc2_dials_dstc10_"):
    assert dial_split in {"train", "dev", "devtest", "test", "teststd"}, "Give the right split name: should be one of train, dev, devtest, test"
    all_dialogues = AllDialogues.from_json(dialogue_name_prefix+dial_split, data_dir)
    return all_dialogues


if __name__ == "__main__":
    all_dials = main_function(dial_split='dev',
                              data_dir=DATA_DIR, 
                              dialogue_name_prefix="simmc2.1_dials_dstc11_")
    print(all_dials.dialogue_list[:2])

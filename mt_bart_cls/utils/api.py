"""
Updated by Yirong Chen 
Check for SIMMC 2.1
Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/01
"""
import numpy as np

from .dialogue import main_function as dialogue_main_function
from .metadata import main_function as metadata_main_function
from .scene import Scene
import pdb


# If we use each object token as special token
NUM_FASHION_ITEMS = 288
NUM_FURNITURE_ITEMS = 57
MAX_NUM_OBJ_IN_SCENE = 200


class PromptAPI:
    def __init__(self, 
                 dial_split=None, 
                 data_dir=None, 
                 dialogue_name_prefix="simmc2.1_dials_dstc11_",
                 jsons_dir_name="simmc2_scene_jsons_dstc10_public",
                 images_dir_name="simmc2_scene_images_dstc10_public"):
        """Updated by Yirong Chen 
           for SIMMC2.1: 
           dialogue_name_prefix="simmc2.1_dials_dstc11_"
           for SIMMC2.0:
           dialogue_name_prefix="simmc2_dials_dstc10_" 

        """
        assert dial_split in {'train', 'dev', 'devtest', 'test', 'teststd', None}
        self.data_dir=data_dir # 数据集的目录
        self.dialogue_name_prefix=dialogue_name_prefix # 存储对话数据的json文件的前缀
        self.jsons_dir_name=jsons_dir_name # bbox与scene的json存储文件夹名称，test和public分开
        self.images_dir_name=images_dir_name # 场景图片存储文件夹名称，test和public分开

        if dial_split:
            self.dial_split = dial_split
            self.all_dialogues = dialogue_main_function(dial_split=self.dial_split,
                                                        data_dir=self.data_dir,
                                                        dialogue_name_prefix=self.dialogue_name_prefix)

        self.fashion_meta, self.furniture_meta = metadata_main_function(data_dir=self.data_dir)

    def given_scene_objid_get_meta(self, 
                                   scene_name: str, 
                                   obj_unique_id=None, 
                                   obj_index=None):
        assert (obj_unique_id is None or obj_index is None) and (not (obj_unique_id is None and obj_index is None)), \
            "either only one of obj_unique_id and obj_index should have value"
        scene = Scene.from_json(scene_name,
                                data_dir=self.data_dir, 
                                jsons_dir_name=self.jsons_dir_name, 
                                images_dir_name=self.images_dir_name)
        # print('scene', scene)
        if 'cloth' in scene_name:
            domain_metadata = self.fashion_meta
        elif 'wayfair' in scene_name:
            domain_metadata = self.furniture_meta
        else:
            raise ValueError("scene_name should contain either word 'cloth' or 'wayfair'")

        if obj_unique_id is not None:
            for obj in scene.scene_object:
                if obj.unique_id == int(obj_unique_id):
                    for meta in domain_metadata:
                        if meta.name == obj.prefab_path:
                            return meta  # instance of {Fashion|Furniture}Metadata

        if obj_index is not None:
            for obj in scene.scene_object:
                if obj.index == int(obj_index):
                    for meta in domain_metadata:
                        if meta.name == obj.prefab_path:
                            return meta  # instance of {Fashion|Furniture}Metadata
    
    def given_scene_get_all_obj_info(self, scene_name: str):
        scene = Scene.from_json(scene_name,
                                data_dir=self.data_dir, 
                                jsons_dir_name=self.jsons_dir_name, 
                                images_dir_name=self.images_dir_name)
        if 'cloth' in scene_name:
            domain_metadata = self.fashion_meta
        elif 'wayfair' in scene_name:
            domain_metadata = self.furniture_meta
        else:
            raise ValueError("scene_name should contain either word 'cloth' or 'wayfair'")

        scene_obj_meta_list = []

        for obj in scene.scene_object:
            for meta in domain_metadata:
                if meta.name == obj.prefab_path:
                    scene_obj_meta_list.append({'obj': obj, 'meta': meta})

        return scene_obj_meta_list

    def given_belief_get_obj(self, scene_name: str, belief_state: str):
        """
        aim to get proper object, given (generated) belief state
        belief_state: string between "=> Belief State : " and "<EOB>"
        """
        scene_obj_meta_list = self.given_scene_get_all_meta(scene_name)
        belief_state = belief_state.strip()
        act = belief_state.split('[')[0]
        slot_string = belief_state.split('[')[1].split(']')[0]
        slot_list = [s.replace(' ','') for s in slot_string.split(',')]
        slot = dict()
        for k_v in slot_list:
            if k_v != '' and k_v != ' ':
                print("k_v",k_v)
                k, v = k_v.split('=')
                slot[k] = v
        request_slot = belief_state.split('(')[1].split(')')[0].replace(' ', '').split(',')
        objects = list(map(int, belief_state.split('<')[1].split('>')[0].replace(' ', '').split(',')))

    def dial_data_returner(self, len_history=2):
        """
        returns dial data including coref and belief state. this does not make any kind of a file, but just returns a list, because it's more flexible.
        len_history: length of history (1 user uttr + 1 system uttr = 1 history). 
        Total utterance for each time is therefore 2*len_history + 1 (current user utter)

        [
            {  // start of a dialogue
                domain: dialogue's domain
                dialogues: [
                    {'context': history_1 + user_uttr_1 as list, 'context_with_obj': context with system's mentioned objs, 'belief': belief_1 dict},
                    {'context': history_2 + user_uttr_2 as list, 'context_with_obj': context with system's mentioned objs, 'belief': belief_2 dict},
                    ...
                ]
                scene_objects: {
                    scene_idx_1: {
                    0: 0-th object's meta info, in dictionary format
                    1: 1-th object's meta info, in dictionary format
                    ...
                    },
                    scene_idx_2: {
                    0: 0-th object's meta info, in dictionary format
                    1: 1-th object's meta info, in dictionary format
                    ...
                    },
                    ...
                }
            }  // end of a dialogue
            ...
        ]
        """
        dialogue_data = []
        for dialogue in self.all_dialogues.dialogue_list:
            dialogue_dict = {'domain': dialogue.domain, 'scene_objects': dict(), 'dialogues': []}
            scene_ids = dialogue.scene_ids
            for k, scene_id in scene_ids.items():
                dialogue_dict['scene_objects'][int(k)] = dict()
                scene = Scene.from_json(scene_id,
                                        data_dir=self.data_dir, 
                                        jsons_dir_name=self.jsons_dir_name, 
                                        images_dir_name=self.images_dir_name)
                scene_objs_info = self.given_scene_get_all_obj_info(scene_id)
                for obj_info in scene_objs_info:
                    # TODO: convert world to camera-relative position, considering camera's position and orientation (direction vec -> ouler angle -> subtract displacement then apply rotation matrix)
                    obj_info_dict = {**vars(obj_info['meta']), **vars(obj_info['obj'])}  # order of 'meta' and 'obj' matters
                    # convert object's world position to camera-relative position
                    # stackoverflow.com/questions/21622956/how-to-convert-direction-vector-to-euler-angles
                    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
                    
                    # Actually, only position from camera matters... for directional utterance, using bbox is better
                    obj_world_pos = obj_info['obj'].position
                    camera_pos = scene.camera_object.camera
                    distance_from_camera = np.linalg.norm(np.array(obj_world_pos) - np.array(camera_pos))
                    obj_info_dict['distance'] = distance_from_camera
                    dialogue_dict['scene_objects'][int(k)][obj_info['obj'].index] = obj_info_dict
            
            for idx, single_turn in enumerate(dialogue.single_dialogue_list):
                single_turn_dict = dict()
                belief = single_turn.transcript_annotated
                belief_dict = {'act': ":".join([belief.act.dialogue_act, belief.act.activity]), 'slot_values': belief.act_attributes.slot_values,
                               'request_slots': belief.act_attributes.request_slots, 'objects': belief.act_attributes.objects}
                single_turn_dict['belief'] = belief_dict
                context_list = []
                context_with_obj_list = []
                context_list.insert(0, 'USER : ' + single_turn.transcript)
                context_with_obj_list.insert(0, 'USER : ' + single_turn.transcript)
                for i in range(1, len_history+1):
                    if idx - i >= 0:
                        one_history = 'USER : ' + dialogue.single_dialogue_list[idx-i].transcript + ' SYSTEM : ' + dialogue.single_dialogue_list[idx-i].system_transcript
                        context_list.insert(0, one_history)
                        obj = ', '.join(list(map(str,dialogue.single_dialogue_list[idx-i].system_transcript_annotated.act_attributes.objects)))
                        one_history_with_obj = 'USER : ' + dialogue.single_dialogue_list[idx-i].transcript + ' SYSTEM : ' + dialogue.single_dialogue_list[idx-i].system_transcript + ' <SOM> ' + obj + ' <EOM>'
                        context_with_obj_list.insert(0, one_history_with_obj)
                    else:
                        break
                single_turn_dict['context'] = context_list
                single_turn_dict['context_with_obj'] = context_with_obj_list
                dialogue_dict['dialogues'].append(single_turn_dict)

            dialogue_data.append(dialogue_dict)

        return dialogue_data
    
    # def item_tokens_attrs(item2id, ) 


if __name__ == "__main__":
    prompt_api = PromptAPI(dial_split="dev", 
                           data_dir="/ps2/sli/data/data_taowang49/projects/19_dstc11/simmc2.1_solutions/work/simmc2.1-iflytek/data", 
                           dialogue_name_prefix="simmc2.1_dials_dstc11_",
                           jsons_dir_name="simmc2_scene_jsons_dstc10_public",
                           images_dir_name="simmc2_scene_images_dstc10_public")
    metas = prompt_api.given_scene_get_all_obj_info('m_cloth_store_1416238_woman_3_8')
    meta_dict = {**vars(metas[0]['meta']), **vars(metas[0]['obj'])}
    # pdb.set_trace()
    print(metas)

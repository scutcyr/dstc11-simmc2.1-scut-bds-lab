# coding=utf-8
# Copyright 2022 Research Center of Body Data Science from South China University of Technology. All rights reserved.

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
Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/01
说明：由于SIMMC2.0或者2.1数据集存放在其他路径，而不是存放在本代码项目下面的路径，因此需要直接指定DATA_DIR为数据集的绝对路径
"""
import glob
import json
from os.path import isfile, join
from typing import List, Optional

import attr
from attr import converters, validators

#from .util import find_data_dir, DATA_DIR_ROOT_PATH

#DATA_DIR = find_data_dir('DSTC10-SIMMC')  # give root folder name of simmc2 as argument. Ex) find_data_dir('DSTC10-SIMMC')
#Updated by Yirong Chen
#DATA_DIR = DATA_DIR_ROOT_PATH
DATA_DIR = '~/dstc11_simmc2.1_scut-bds-lab/data'

@attr.s
class SceneObject:
    name: str = attr.ib(
        converter=str,
        validator=validators.instance_of(str)
    )
    prefab_path: str = attr.ib(
        converter=str,
        validator=validators.instance_of(str)
    )
    unique_id: int = attr.ib(
        converter=int,
        validator=validators.instance_of(int)
    )
    index: int = attr.ib(
        converter=int,
        validator=validators.instance_of(int)
    )
    bbox: List[int] = attr.ib(
        converter=lambda x: [int(_) for _ in x],
        validator=validators.instance_of(list)
    )
    position: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    up: Optional[List[int]] = attr.ib(  #  should't be dictionary?
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )
    down: Optional[List[int]] = attr.ib(
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )
    left: Optional[List[int]] = attr.ib(
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )
    right: Optional[List[int]] = attr.ib(
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )

    @bbox.validator
    def check(self, attribute, value):
        if len(value) != 4:
            raise ValueError("bbox must have 4 elements.")
        
    @position.validator
    def check(self, attribute, value):
        if len(value) != 3:
            raise ValueError("position must have 3 elements.")


@attr.s
class CameraObject:
    """
        List because each contains x, y, z position coordinates
    """
    camera: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    right: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    forward: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    up: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )


@attr.s
class Scene:
    scene_name: str = attr.ib(
        converter=str,
        validator=validators.instance_of(str)
    )
    # scene_image: np.ndarray = attr.ib()
    scene_object: SceneObject = attr.ib()
    camera_object: CameraObject = attr.ib()

    @scene_name.validator
    def check(self, 
              attribute, 
              value, 
              data_dir=DATA_DIR, 
              jsons_dir_name="simmc2_scene_jsons_dstc10_public", 
              images_dir_name="simmc2_scene_images_dstc10_public"):
        if value.startswith('m_'):  # "m_ ... " has no bbox, but without "m_" there is bbox file 
            bbox_json = join(data_dir, jsons_dir_name, "{}_bbox.json".format(value[2:]))
        else:
            bbox_json = join(data_dir, jsons_dir_name, "{}_bbox.json".format(value))
            
        scene_json = join(data_dir, jsons_dir_name, "{}_scene.json".format(value))
        # if value.startswith('m_'): 
        #     scene_image = join(data_dir, "images", "{}.png".format(value[2:]))
        # else:
        #     scene_image = join(data_dir, "images", "{}.png".format(value))

        if not (
            isfile(bbox_json)\
                and isfile(scene_json)
                    #and isfile(scene_image)
        ):
            raise ValueError("All required scene files do not exist!")

    @classmethod
    def from_json(cls, 
                  scene_name: str, 
                  data_dir=DATA_DIR, 
                  jsons_dir_name="simmc2_scene_jsons_dstc10_public", 
                  images_dir_name="simmc2_scene_images_dstc10_public"):
        """
        Input dictionary has the following format 

        *_bbox.json (includes camera):

            {
                "Items": [
                    {
                        "name": <str>,
                        "prefabPath": <str>,
                        "bbox": [<int>, <int>, <int>, <int>],
                        "position": [<float>, <float>, <float>]
                    },
                    ...,
                    {
                        "name": "camera_{}".format(<str>),
                        "prefabPath": "camera",
                        "bbox": [-1, -1, -1, -1],
                        "position": [<float>, <float>, <float>]
                    }
                ]
            }
        
        *_scene.json:

            {
                "scenes": [
                    {
                        "objects": [
                            {
                                "prefab_path": <str>,
                                "unique_id": <int>,
                                "index": <int>,
                                "bbox": [<int>, <int>, <int>, <int>],
                                "position": [<float>, <float>, <float>]
                            },
                            ...
                        ]
                    }
                ],
                "relationships": {
                    "right": {
                        "0": <list<int>>, # Each key is stringified int index
                        ...
                    },
                    "left": ...,
                    "up": ...,
                    "down": ...
                }
            }
        """
        if scene_name.startswith('m_'):  # "m_ ... " has no bbox, but without "m_" there is bbox file 
            bbox_json = json.load(open(join(data_dir, jsons_dir_name, "{}_bbox.json".format(scene_name[2:]))))
        else:
            bbox_json = json.load(open(join(data_dir, jsons_dir_name, "{}_bbox.json".format(scene_name))))
        scene_json = json.load(
            open(
                join(data_dir, "jsons", "{}_scene.json".format(scene_name))
            )
        )
        # I guess we don't need image
        # image = cv2.imread(join(data_dir, images_dir_name, "{}.png".format(scene_name)))

        # Define camera object
        camera_args = dict()
        for item in bbox_json['Items']:
            if item['prefabPath'] == "camera":
                if item['name'] == "camera":
                    camera_args['camera'] = item['position']
                elif item['name'] == "camera_right":
                    camera_args['right'] = item['position']
                elif item['name'] == "camera_forward":
                    camera_args['forward'] = item['position']
                elif item['name'] == "camera_up":
                    camera_args['up'] = item['position']
                else:
                    pass
        camera_obj = CameraObject(**camera_args)

        scene_obj = list()
        # Define scene object
        relation = scene_json['scenes'][0]['relationships']
        for item in scene_json['scenes'][0]['objects']:
            scene_obj_args = dict()
            for k, v in item.items():
                scene_obj_args[k] = v

                if k == "index":
                    # right, left, up, down
                    
                    _right = relation['right'].get(str(v), None) if 'right' in relation else None
                    scene_obj_args['right'] = _right
                    _left = relation['left'].get(str(v), None) if 'left' in relation else None
                    scene_obj_args['left'] = _left
                    _up = relation['up'].get(str(v), None) if 'up' in relation else None
                    scene_obj_args['up'] = _up
                    _down = relation['down'].get(str(v), None) if 'down' in relation else None
                    scene_obj_args['down'] = _down

            # Find name from *_bbox.jsons file.
            for bbox_item in bbox_json['Items']:
                if (
                    bbox_item['bbox'] == scene_obj_args['bbox']\
                        and bbox_item['prefabPath'] == scene_obj_args['prefab_path']\
                            and bbox_item['position'] == scene_obj_args['position']
                ):
                    scene_obj_args['name'] = bbox_item['name']            
            scene_obj.append(SceneObject(**scene_obj_args))

        scene_args = {
            'scene_name': scene_name,
            # 'scene_image': image,
            'scene_object': scene_obj,
            'camera_object': camera_obj
        }
        return cls(**scene_args)
        

@attr.s
class Store:
    scenes: List[Scene] = attr.ib()

    @classmethod
    def from_name(cls, 
                  name: str,
                  data_dir=DATA_DIR, 
                  jsons_dir_name="simmc2_scene_jsons_dstc10_public", 
                  images_dir_name="simmc2_scene_images_dstc10_public"):
        files = glob.glob(
            join(
                # "..", "data", "jsons", "{}_*".format(name)  duplicated Scene instance will be made cause both scene jsonfile and bbox jsonfile are catched
                data_dir, jsons_dir_name, "{}_scene*".format(name)
            )
        )
        
        # print('files:', files)

        # Caution that this may return more than 1 scene depending on name (Ex. cloth_store_1)
        return cls([
            Scene.from_json('_'.join(f.split('/')[-1].split('_')[:-1]).replace(".json", '')) for f in files
        ])

if __name__ == "__main__":
    print(Store.from_name("cloth_store_1_1_1"))

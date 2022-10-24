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


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import json
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # 解决IOError: broken data stream when reading image file

def get_image_name(scene_ids, turn_ind):
    """Given scene ids and turn index, get the image name.
    """
    sorted_scene_ids = sorted(
        ((int(key), val) for key, val in scene_ids.items()),
        key=lambda x: x[0],
        reverse=True
    )
    # NOTE: Hardcoded to only two scenes.
    if turn_ind >= sorted_scene_ids[0][0]:
        scene_label = sorted_scene_ids[0][1]
    else:
        scene_label = sorted_scene_ids[1][1]
    image_label = scene_label
    if "m_" in scene_label:
        image_label = image_label.replace("m_", "")
    return f"{image_label}.png", scene_label

def read_image(image_dir, image_name):
    '''如果返回False则说明图像不存在

    '''
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        image = Image.open(image_path)
        if "png" in image_name:
            image = image.convert("RGB")
        return image
    else:
        print(image_name, " is not exists in ", image_dir)
        return False

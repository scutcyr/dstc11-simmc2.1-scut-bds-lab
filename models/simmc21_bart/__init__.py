# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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


from typing import TYPE_CHECKING
from transformers.file_utils import is_torch_available

if is_torch_available():
    from .modeling_bart import (
        MultiTaskBartForConditionalGeneration,
        BoxEmbedding,
        NoCorefHead,
        DisambiguationHead,
        FashionEncoderHead,
        FurnitureEncoderHead
    )
    from .modeling_bart_v2 import (
        MultiTaskBartForConditionalGenerationWithDisamb
    )
    from .modeling_bart_joint_disam_coref import (
        MultiTaskBartForConditionalGenerationJointDisambCoref
    )

    from .modeling_bart_joint_intent import (
        MultiTaskBartForConditionalGenerationWithDisambAndIntent
    )



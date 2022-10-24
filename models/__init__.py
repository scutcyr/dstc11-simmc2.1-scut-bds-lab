# coding=utf-8
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

__version__ = "1.0.0"

from . import (
    simmc21_bart,
    simmc21_blenderbot,
    simmc21_t5,
    simmc21_ul2,
    simmc21_flava,
    simmc21_ofa,
)

from .modeling_simmc21_outputs import Seq2SeqLMOutputForSIMMC

# Model parameters calculating and freezing
from .model_parameters import (count_trainable_parameters, count_total_parameters, show_trainable_parameters, 
                                set_freeze_by_names, freeze_by_model_name, unfreeze_by_model_name)

from .kl_loss import compute_kl_loss
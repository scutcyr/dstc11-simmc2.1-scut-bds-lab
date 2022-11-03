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


""" PyTorch model output for DSTC-11 SIMMC 2.1

Updated by Yirong Chen 
Used for [SIMMC 2.1](https://github.com/facebookresearch/simmc2)
Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/04
Refered from the modeling_outputs.py of transformers package

<YOUR_ANACONDA_INSTALL_PATH>/anaconda3/envs/py38/lib/python3.8/site-packages/transformers/modeling_outputs.py

Usage: 


"""
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List

from transformers.utils import ModelOutput


@dataclass
class Seq2SeqLMOutputForSIMMC(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    nocoref_loss: Optional[torch.FloatTensor] = None
    misc_loss: Optional[torch.FloatTensor] = None
    disam_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    disambiguation_logits: Optional[torch.FloatTensor] = None
    nocoref_logits: Optional[torch.FloatTensor] = None
    enc_head_results: Optional[List] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

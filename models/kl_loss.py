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


# File: kl_loss.py
# Description: The kl_loss calculation function
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/12
# Usage:
#    from models import compute_kl_loss


import torch.nn.functional as F

def compute_kl_loss(p, q, pad_mask=None, attention_mask=None, reduction="batchmean"):
    '''
    p/q: torch.Tensor类型，Size为：[batch_size, *], 例如：[batch_size, seq_length, hidden_size], [batch_size, hidden_size]

    reduction="batchmean" or "mean" or "sum"

    pad_mask: [batch_size, seq_length]
    attention_mask: [batch_size, seq_length]

    '''
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        if len(p.size()) != len(pad_mask.size()):
            pad_mask = pad_mask.unsqueeze(-1).expand(-1,-1,p.size()[-1]) # [batch_size, seq_length] -> [batch_size, seq_length, 1] -> [batch_size, seq_length, hidden_size]
        pad_mask = pad_mask > 0 # 转为布尔张量
        if pad_mask[0,0,0] == True:
            # 有可能直接使用attention_mask作为pad_mask输入，此时的pad部分为0，非pad部分为1
            # 对其进行取反，即可获得真正的pad_mask
            pad_mask = ~pad_mask

        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    elif attention_mask is not None:
        pad_mask = attention_mask.unsqueeze(-1).expand(-1,-1,p.size()[-1]) # [batch_size, seq_length] -> [batch_size, seq_length, 1] -> [batch_size, seq_length, hidden_size]
        pad_mask = pad_mask > 0 # 转为布尔张量
        if pad_mask[0,0,0] == True:
            # 有可能直接使用attention_mask作为pad_mask输入，此时的pad部分为0，非pad部分为1
            # 对其进行取反，即可获得真正的pad_mask
            pad_mask = ~pad_mask

        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    if reduction=="mean":
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
    else:
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    
    if reduction=="batchmean":
        return loss/(p.size()[0])
    return loss
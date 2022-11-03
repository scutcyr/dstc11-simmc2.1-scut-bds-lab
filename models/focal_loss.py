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
Reference: https://github.com/alexmeredith8299/focal_loss_pytorch/blob/main/focal_loss_pytorch/focal_loss.py
This function implements binary focal loss for tensors of arbitrary size/shape.

Usage:
loss_fn = BinaryFocalLoss(gamma=5)
logits = torch.tensor([[0.2,0.3,0.5],
                       [0.5,0.4,0.1]])
labels = torch.tensor([[2],[0]],dtype=torch.long)
loss = loss_fn(logits, labels.view(-1))
print(loss)
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

        Example:
            focal_loss = FocalLoss(class_num=2, gamma=2)
            logits = torch.tensor([[-0.0453, -0.1713],
                                   [-0.0453, -0.1713],
                                   [-0.0073, -0.0737]])
            labels = torch.tensor([[1], [1], [1]],dtype=torch.long)
            loss = focal_loss(logits, labels.view(-1))
            print(loss)


    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        '''对于targets = tensor([1, 1, 0])
               class_mask = 
                    tensor([[0., 1.],
                            [0., 1.],
                            [1., 0.]])
        '''


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


























class BinaryFocalLoss(torch.nn.modules.loss._Loss):
    """
    Inherits from torch.nn.modules.loss._Loss. Finds the binary focal loss between each element
    in the input and target tensors.
    Parameters
    -----------
        gamma: float (optional)
            power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean
    Attributes
    -----------
        gamma: float (optional)
            focusing parameter -- power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean
    """
    def __init__(self, gamma=2, reduction='mean'):
        if reduction not in ("sum", "mean", "none", "batchmean"):
            raise AttributeError("Invalid reduction type. Please use 'mean', 'sum', or 'none'.")
        super().__init__(None, None, reduction)
        self.gamma = gamma
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input_tensor, target):
        """
        Compute binary focal loss for an input prediction map and target mask.
        Arguments
        ----------
            input_tensor: torch.Tensor
                input prediction map
            target: torch.Tensor
                target mask
        Returns
        --------
            loss_tensor: torch.Tensor
                binary focal loss, summed, averaged, or raw depending on self.reduction
        """
        #Warn that if sizes don't match errors may occur
        if not target.size() == input_tensor.size():
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size ({input_tensor.size()}). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.",
                stacklevel=2,
            )
            print("input_tensor=", input_tensor)
            print("input_tensor=", target)

        #Broadcast to get sizes/shapes to match
        input_tensor = F.softmax(input_tensor,dim=1)
        input_tensor, target = torch.broadcast_tensors(input_tensor, target)
        assert input_tensor.shape == target.shape, "Input and target tensor shapes don't match"

        #Vectorized computation of binary focal loss
        pt_tensor = (target == 0)*(1-input_tensor) + target*input_tensor
        pt_tensor = torch.clamp(pt_tensor, min=self.eps, max=1.0) #Avoid vanishing gradient
        loss_tensor = -(1-pt_tensor)**self.gamma*torch.log(pt_tensor)

        #Apply reduction
        if self.reduction =='none':
            return loss_tensor
        if self.reduction=='mean':
            return torch.mean(loss_tensor)
        if self.reduction=='batchmean':
            return torch.sum(loss_tensor)/loss_tensor.size()[0]
        #If not none or mean, sum
        return torch.sum(loss_tensor)
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class Focalloss(nn.Module):

    def __init__(self,gamma,num_class,aver = True):
        super(Focalloss,self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.aver = aver
    def forward(self, input,target,weights = None):
        n, c, h, w = input.size()
        inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.contiguous().view(-1)

        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1,1)
        class_mask.scatter_(1, ids.data, 1.)
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        #-weights * (torch.pow((1 - probs), gamma)) * log_p
        loss = (-weights * (torch.pow((1 - probs), self.gamma)) * log_p) if weights is not None else \
            (- (torch.pow((1 - probs), self.gamma)) * log_p)

        if self.aver:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        # self.criterion = nn.CrossEntropyLoss(weight=weight,
        #                                      ignore_index=ignore_label,
        #                                      reduction='none')

    def forward(self, score, target,weights, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = F.cross_entropy(score, target, weight=weights, ignore_index=self.ignore_label,
                                       reduction='none').contiguous().view(-1)
        # pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class OHEM_Focal_Loss(nn.Module):


    def __init__(self,ignore_label=-1, thres=0.7, min_kept=100000,gamma = 2):
        super(OHEM_Focal_Loss,self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.gamma = gamma

    def get_loss(self, input,target,weight = None):
        pred = F.softmax(input,dim=1)
        logp = F.cross_entropy(input,target,weight=weight,ignore_index=self.ignore_label,reduction='none').contiguous().view(-1)
        p = torch.exp(-logp)
        pixel_losses = (1 - p) ** self.gamma * logp
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, outs,target,weight = None):

        x1,x2,x3,x4,x5 = outs
        loss1 = self.get_loss(x1,target,weight)
        loss2 = self.get_loss(x2,target,weight)
        loss3 = self.get_loss(x3,target,weight)
        loss4 = self.get_loss(x4,target,weight)
        loss5 = self.get_loss(x5,target,weight)

        losses = loss1 + loss2 + loss3 + loss4 + loss5
        return losses


if __name__ == '__main__':
    x  = torch.ones(1,16,128,128)
    y = torch.zeros(1,128,128).long()
    loss = Focalloss(2,16,aver=True,is_weight=True,alpha=None)
    # loss2 = Focalloss(2,16,aver=False)
    l1 = loss(x,y)
    # l2 = loss2(x,y)
    print(l1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')
 
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1) 
        return loss

    def forward(self, output_s, output_e, target_s, target_e):
        num_joints = output_s.size(1)
        loss = 0

        for idx in range(num_joints):
            start_pred = output_s[:,idx].squeeze()
            end_pred = output_e[:,idx].squeeze()
            start_gt = target_s[:,idx].squeeze()
            end_gt = target_e[:,idx].squeeze()
            loss += (self.criterion(start_pred,start_gt).mean()) 
            loss += (self.criterion(end_pred,end_gt).mean())
        return loss / num_joints 

class NMTNORMCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTNORMCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
 
        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing
 
    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot
 
    def _bottle(self, v):
        return v.view(-1, v.size(2))
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
 
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = torch.mean(self.criterion_(scores, gtruth), dim=1)
        return loss

    def forward(self, output_s, output_e, target_s, target_e):
        batch_size = output_s.size(0)
        num_joints = output_s.size(1)
        loss = 0

        for idx in range(num_joints):
            start_pred = output_s[:,idx].squeeze()
            end_pred = output_e[:,idx].squeeze()
            start_gt = target_s[:,idx].squeeze()
            end_gt = target_e[:,idx].squeeze()
            
            loss += self.criterion(start_pred,start_gt[:,0]).mean()
            loss += self.criterion(end_pred,end_gt[:,1]).mean()
        return loss / num_joints

class NMTCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
 
        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing
 
    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot
 
    def _bottle(self, v):
        return v.view(-1, v.size(2))
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
 
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = torch.sum(self.criterion_(scores, gtruth), dim=1)
        return loss

    def forward(self, output_s, output_e, target_s, target_e):
        batch_size = output_s.size(0)
        num_joints = output_s.size(1)
        loss = 0

        for idx in range(num_joints):
            start_pred = output_s[:,idx].squeeze()
            end_pred = output_e[:,idx].squeeze()
            start_gt = target_s[:,idx].squeeze()
            end_gt = target_e[:,idx].squeeze()
            
            loss += self.criterion(start_pred,start_gt[:,0]).sum()
            loss += self.criterion(end_pred,end_gt[:,1]).sum()
        return loss / batch_size


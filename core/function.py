import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        input = data['image']
        target_s = data['target_start']
        target_e = data['target_end']

        data_time.update(time.time() - end)

        # compute output
        output_s, output_e = model(input)

        target_s = target_s.cuda(non_blocking=True)
        target_e = target_e.cuda(non_blocking=True)
        loss = criterion(output_s, output_e, target_s, target_e)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input[0].size(0))
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i >= 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input[0].size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)


def validate(args, val_loader, val_dataset, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, 1, 2),
        dtype=np.float32
    )
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            input = data['image']
            target_s = data['target_start']
            target_e = data['target_end']
            loc_start = data['start']
            loc_end = data['end']
            z_start = data['z_start']
            z_end = data['z_end']
            spacing = data['spacing']
            # compute output
            output_s, output_e = model(input)
            
            output_s = F.softmax(output_s, dim=2)
            output_e = F.softmax(output_e, dim=2)                     

            target_s = target_s.cuda(non_blocking=True)
            target_e = target_e.cuda(non_blocking=True)
            loss = torch.sum((output_s - target_s) ** 2) + torch.sum((output_e - target_e)**2) 
            num_images = input[0].size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            output_s = F.softmax(output_s, dim=2)
            output_e = F.softmax(output_e, dim=2)
            
            _, preds_s = output_s.max(2,keepdim=True)
            _, preds_e = output_e.max(2,keepdim=True)
           
            output = torch.ones([input[0].size(0), 2])
            output[:, 0] = torch.floor(preds_s / args.split_ratio).squeeze() * (spacing[:,-1].to(preds_s.device))
            output[:, 1] = torch.ceil(preds_e / args.split_ratio).squeeze() * (spacing[:,-1].to(preds_e.device))
            
            target = np.ones([input[0].size(0),2])
            target[:, 0] = loc_start
            target[:, 1] = loc_end

            output = output.cpu().numpy()
            preds = output.copy()
            gts = target.copy()
            
            start_acc = float(np.mean(np.abs(preds[:,0]-gts[:,0])))
            end_acc = float(np.mean(np.abs(preds[:,1]-gts[:,1])))
            acc = float(np.mean(np.abs(preds-gts)))

            z_start_acc = np.mean(np.abs((preds_s.squeeze() - z_start.to(preds_s.device).squeeze()).cpu().numpy()))
            z_end_acc = np.mean(np.abs((preds_e.squeeze()-z_end.to(preds_e.device).squeeze()).cpu().numpy()))

            idx += num_images

            if i >= 0:
                msg = 'Sample ID :{sample_id}\t'\
                      'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {s_acc:.3f}({s:.3f}) | {e_acc:.3f}({e:.3f})'.format(
                          i, len(val_loader), sample_id = str(data['id'].cpu().numpy()), batch_time=batch_time,
                          loss=losses, s_acc=start_acc, s=z_start_acc, e_acc=end_acc,e=z_end_acc)
                logger.info(msg)

                  
    return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
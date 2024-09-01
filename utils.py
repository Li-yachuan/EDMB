import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

def norm_loss(mean,std):
    var = torch.pow(std, 2)
    kl_reg_loss = 0.5 * torch.sum(torch.pow(mean, 2)
                                  + var - 1.0 - torch.log(var))
    return kl_reg_loss

def cross_entropy_loss(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(
        prediction, labelf, weight=mask, reduction='sum')

    return cost

# def cross_entropy_loss(prediction, labelf, beta):
#     mask = labelf.clone()
#     num_positive = torch.sum((labelf >= 0.5) & (labelf <= 1)).float()
#     num_negative = torch.sum(labelf < 0.5).float()
#
#     mask[(labelf >= 0.5) & (labelf <= 1.5)] = 1.0 * num_negative / (num_positive + num_negative)
#     mask[labelf < 0.5] = beta * num_positive / (num_positive + num_negative)
#     mask[labelf > 1.5] = 0
#     cost = F.binary_cross_entropy(
#         prediction, labelf, weight=mask, reduction='sum')
#
#     return cost


def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
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
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def step_lr_scheduler(optimizer, epoch, lr_decay_epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch in lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']

    return optimizer


class Metrics(object):
    def __init__(self):
        self.auc = Averagvalue()
        self.f1 = Averagvalue()
        self.acc = Averagvalue()
        self.sen = Averagvalue()
        self.spe = Averagvalue()
        self.pre = Averagvalue()
        self.iou = Averagvalue()

    def __call__(self, predict, target):
        predict = predict.flatten()
        threshold = 0.5
        predict_b = np.where(predict >= threshold, 1, 0)
        target = target.numpy().flatten()

        tp = (predict_b * target).sum()
        tn = ((1 - predict_b) * (1 - target)).sum()
        fp = ((1 - target) * predict_b).sum()
        fn = ((1 - predict_b) * target).sum()
        auc = roc_auc_score(target, predict)
        acc = (tp + tn) / (tp + fp + fn + tn)
        pre = tp / (tp + fp)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        iou = tp / (tp + fp + fn)
        f1 = 2 * pre * sen / (pre + sen)

        self._metrics_update(auc, f1, acc, sen, spe, pre, iou)

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):
        return {
            "AUC": self.auc.avg,
            "F1": self.f1.avg,
            "Acc": self.acc.avg,
            "Sen": self.sen.avg,
            "Spe": self.spe.avg,
            "pre": self.pre.avg,
            "IOU": self.iou.avg
        }

    def show(self):
        print(
            "Acc:{:.4f}".format(self.acc.avg),
            "Sen:{:.4f}".format(self.sen.avg),
            "Spe:{:.4f}".format(self.spe.avg),
            "AUC:{:.4f}".format(self.auc.avg),
            "F1:{:.4f}".format(self.f1.avg),
            "IOU:{:.4f}".format(self.iou.avg),
            "pre:{:.4f}".format(self.pre.avg),
        )

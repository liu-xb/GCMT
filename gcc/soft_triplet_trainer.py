from __future__ import print_function, absolute_import
import time
from torch.nn import functional as F
import random
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter
import torch
from torch import nn

class SoftTripletTrainer(object):
    def __init__(self, model_list, model_ema_list, num_cluster=500, alpha=0.999):
        super().__init__()
        self.models = model_list
        self.num_cluster = num_cluster
        self.model_num = len(self.models)
        self.model_emas = model_ema_list
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,optimizer, print_freq=1, train_iters=200, 
              loss_weight=0.6, k=12, beta=0.05):
        for model in self.models:
            model.train()
        for model_ema in self.model_emas:
            model_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_soft_triplet = AverageMeter()
        precisions = [AverageMeter() for i in range(self.model_num)]
        losses = AverageMeter()

        end = time.time()
        for iter_idx in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t = []
            p_out_t = []
            for i in range(self.model_num):
                f_out_t_i, p_out_t_i = self.models[i](inputs[i])
                f_out_t.append(f_out_t_i)
                p_out_t.append(p_out_t_i)

            f_out_t_ema = []
            p_out_t_ema = []
            for i in range(self.model_num):
                f_out_t_ema_i, p_out_t_ema_i = self.model_emas[i](inputs[i])
                f_out_t_ema.append(f_out_t_ema_i)
                p_out_t_ema.append(p_out_t_ema_i)

            loss_ce = loss_tri = 0
            for i in range(self.model_num):
                loss_ce += self.criterion_ce(p_out_t[i], targets)

            # entropy loss
            target_p = F.softmax(p_out_t_ema[0], dim=1).detach()
            target_entropy = - (target_p * torch.log(target_p)).mean(0).sum()
            for i in range(1, self.model_num):
                temp = F.softmax(p_out_t_ema[i], dim=1).detach()
                target_p += temp
                target_entropy -= (temp * torch.log(temp)).mean(0).sum()
            target_p /= 1.0 * self.model_num

            lp = self.logsoftmax(p_out_t[0])
            for i in range(1, self.model_num):
                lp += self.logsoftmax(p_out_t[i])
            loss_ce_soft = - ( target_p * lp ).mean(0).sum()

            # soft triplet loss
            loss_tri_soft = self.criterion_tri_soft(f_out_t[0], f_out_t_ema[0], targets)

            loss = loss_ce*(1- loss_weight) + \
                     loss_ce_soft * (1- loss_weight) \
                     + loss_tri_soft * loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for i in range(self.model_num):
                self._update_ema_variables(self.models[i], self.model_emas[i], self.alpha, epoch*len(data_loader_target)+iter_idx)

            prec_1, = accuracy(p_out_t[0].data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_ce_soft.update((loss_ce_soft - target_entropy).item())
            losses_soft_triplet.update((loss_tri_soft).item())
            losses.update(loss.item())
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if ((iter_idx + 1) % print_freq == 0) or (iter_idx == 0):
                print('[Epoch:{:03d}] [{:03d}/{:03d}] | '
                      'Loss: {:2.3f} | '
                      'Ce: {:2.3f} | '
                      'Entropy: {:2.3f} | '
                      'SoftTriplet: {:2.3f} | '
                      'Acc: {:2.1%}'
                      .format(epoch, iter_idx + 1, len(data_loader_target),
                              losses.avg,
                              losses_ce.avg,
                              losses_ce_soft.avg, 
                              losses_soft_triplet.avg,
                              precisions[0].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        targets = pids.cuda()
        inputs_list = [inputs_1]
        return inputs_list, targets

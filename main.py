import argparse
import logging
import math
import os
import random
import time
import numpy as np
import torch
import torchvision
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
from fast_pytorch_kmeans import KMeans
from sklearn.cluster import KMeans as SklearnKMeans
from data import load_dataset, load_eval_dataset
from model import ModelEMA, get_model
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict, CopyUpdate,
                   ContinuousDataloader, LrScheduler, EMA_update, SelfAdaptiveTrainingCE, ClassBasesSelection)
import csv
from gensim.models import KeyedVectors
from ot import emd2, sinkhorn2 
from geomloss import  SamplesLoss
import ot

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--total-steps', default=10000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=250, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=65, type=int, help='number of classes')
parser.add_argument('--resize', default=224, type=int, help='resize image')
parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='train weight decay')
parser.add_argument('--ema', default=0., type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='steps that student waits for taecher')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=625, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=3e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0.9, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=8, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--consistency_threshold', default=0.0, type=float, help='source pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--tar_weight', default=1.0, type=float, help='coefficient of target unlabeled loss')
parser.add_argument('--consistency_weight', default=0.1, type=float, help='coefficient of target consistency loss')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", default=False, help="use 16-bit (mixed) precision", action='store_true')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--warmup_path', default='./checkpoint/warm', type=str)
parser.add_argument('--src', default=['Clipart', 'Art', 'Product'], nargs='+')
parser.add_argument('--tar', default=['RealWorld'], nargs='+')
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--bottleneck', default='conv', type=str)
parser.add_argument('--backbone', default='resnet50', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--bottle_dim', default=256, type=int)
parser.add_argument('--port', default=18888, type=int)
parser.add_argument('--eval_batch_size', default=128, type=int)
parser.add_argument('--consis_weight', default=1., type=float)
parser.add_argument('--tar_alignment_weight', default=0.01, type=float)
parser.add_argument('--src_alignment_weight', default=1.0, type=float)
parser.add_argument('--ext_weight', default=0.0, type=float)
parser.add_argument('--res_weight', default=0.01, type=float)
parser.add_argument('--wait_step', default=0, type=int)
parser.add_argument('--use_bypass', default=False, action='store_true')
parser.add_argument('--src_pass', default=False, action='store_true')
parser.add_argument('--use_proxy', default=False, action='store_true')
parser.add_argument('--is_drop', default=False, action='store_true')
parser.add_argument('--if_detach', default=False, action='store_true')
parser.add_argument('--if_zero', default=False, action='store_true')
parser.add_argument('--consis_tar', default=False, action='store_true')
parser.add_argument('--consis_src', default=False, action='store_true')
parser.add_argument('--consis_training_mode', default=0, type=int)
parser.add_argument('--align_mode', default=0, type=int)
parser.add_argument('--ext_mode', default=0, type=int)
parser.add_argument('--if_consis_align', default=False, action='store_true')
parser.add_argument('--tar_threshold', default=0.9, type=float, help='target pseudo label threshold')
parser.add_argument('--further_balance', default=False, action='store_true')
parser.add_argument('--T', default=3.0, type=float)
parser.add_argument('--bt', default=0.1, type=float)
parser.add_argument('--src_tar_weight', default=0.5, type=float)
parser.add_argument('--box_warm_iter', default=2, type=int)
parser.add_argument('--weight_tau', default=1, type=float)
parser.add_argument('--use_weight_pred', default=False, action='store_true')
parser.add_argument('--use_prediction_alignment', default=False, action='store_true')
parser.add_argument('--cls_var', default=0, type=int)
parser.add_argument('--weight_var', default=0, type=int)
# === OT/Word2Vec arguments ===
parser.add_argument('--ot_alpha', type=float, default=1.0, help='OT Wasserstein distance weight')
parser.add_argument('--ot_beta', type=float, default=0.5, help='Confidence weight')
parser.add_argument('--ot_gamma', type=float, default=0.5, help='Diversity weight')
parser.add_argument('--word2vec_path', type=str, default='GoogleNews-vectors-negative300.bin', help='Path to Word2Vec model')
# Curriculum gamma scheduling
parser.add_argument('--gamma_explore', type=float, default=1.0, help='Early stage diversity reward')
parser.add_argument('--gamma_mid', type=float, default=0.0, help='Middle stage gamma')
parser.add_argument('--gamma_refine', type=float, default=-0.5, help='Late stage diversity penalty')
# Sinkhorn OT options
parser.add_argument('--use_sinkhorn', action='store_true', help='Use Sinkhorn distance instead of EMD')
parser.add_argument('--sinkhorn_reg', type=float, default=0.05, help='Entropy regularization (ε) for Sinkhorn')
parser.add_argument('--log_file_name', type=str, default='train_ot.log', help='Filename for logging output')
parser.add_argument('--unbalanced_mass_reg', type=float, default=0.1, help='mass regularization for unbalanced OT (reg_m)')

parser.add_argument('--use_align_cost_reg', action='store_true', default=True,
                    help='Enable OT-based semantic regularizer for alignment loss')

parser.add_argument('--align_cost_weight', type=float, default=0.5,
                    help='Weight of semantic alignment regularizer added to L_align')

def shift_log(x, offset=1e-6):
    return torch.log(torch.clamp(x + offset, max=1.))

def get_data(dataset_loaders, target_loader, args):
    src_datas_w = []
    src_datas_s = []
    src_labels = []
    src_indexs = []

    for i in range(0, len(dataset_loaders)):
        loader = dataset_loaders[i]
        data_w, label, src_inds = loader.use_next()
        src_datas_w.append(data_w.cuda())  # to(args.device))
        src_labels.append(label.cuda())  # .to(args.device))
        src_indexs.append(src_inds.cuda())
    (tar_data_w, tar_data_s, tar_data_m), tar_label, tar_index = target_loader.use_next()
    tar_data_w = tar_data_w.cuda()  # .to(args.device)
    tar_data_s = tar_data_s.cuda()  # .to(args.device)
    tar_data_m = tar_data_m.cuda()
    tar_label = tar_label.cuda()  # .to(args.device)
    tar_index = tar_index.cuda()
    return src_datas_w, src_labels, tar_data_w, tar_data_s, tar_data_m, tar_label, src_indexs, tar_index


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_score(pred):
    return pred.data.max(1)[0]


def balance_fix_loss(pred1, pred2, mask=None):
    mid = (pred1 + pred2) / 2
    eps = 1e-7
    loss1 = (torch.sum(-torch.log(mid) * F.one_hot(pred1.data.max(1)[1], pred1.shape[1]), dim=1) * mask).mean()
    kl_criterion = nn.KLDivLoss(reduce=False)
    kl1 = (torch.sum(kl_criterion(torch.log(mid + eps), pred1), dim=1) * mask).mean()
    kl2 = (torch.sum(kl_criterion(torch.log(mid + eps), pred2), dim=1) * mask).mean()
    loss2 = (kl1 + kl2) / 2
    return loss1 + loss2


def pJS(pred1, pred2, p=0.5, mask=None):
    kl_criterion = nn.KLDivLoss(reduce=False)
    eps = 1e-7
    mid = p * pred1 + (1 - p) * pred2
    if mask is None:
        weight = torch.ones(len(pred1)).cuda()
    else:
        weight = mask
    kl1 = (torch.sum(kl_criterion(torch.log(mid + eps), pred1), dim=1) * weight).mean()
    kl2 = (torch.sum(kl_criterion(torch.log(mid + eps), pred2), dim=1) * weight).mean()

    return p * kl1 + (1 - p) * kl2


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']



def ot_reg_cls(pred1, pred2, avg_preds, class_heads, w2v_embeddings, threshold=0.9, alpha=0.5):
    """
    OT-based semantic pseudo-labeling loss (OTMatch-style replacement for L_psd).
    
    Args:
        pred1: [B, K] - teacher softmax prediction (e.g., esm_tar_preds)
        pred2: [B, K] - student softmax prediction (e.g., esm_aug_tar_preds)
        avg_preds: [B, K] - ensemble logits for pseudo-labels + confidence filtering
        class_heads: [K, D1] - classifier head weights (learned)
        w2v_embeddings: [K, D2] - Word2Vec embeddings (fixed)
        threshold: float - confidence threshold
        alpha: float - blend weight between classifier heads and W2V

    Returns:
        Scalar OT-based consistency loss
    """
    B, K = pred1.size()
    device = pred1.device

    # Compute blended semantic cost matrix
    fc_heads = F.normalize(class_heads, dim=1)
    w2v_heads = F.normalize(w2v_embeddings, dim=1)
    sim_fc = torch.matmul(fc_heads, fc_heads.T)
    sim_w2v = torch.matmul(w2v_heads, w2v_heads.T)
    cost_matrix = 1.0 - (alpha * sim_fc + (1 - alpha) * sim_w2v)

    # Confidence filtering
    conf, pseudo_labels = avg_preds.max(dim=1)
    mask = conf >= threshold
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device), cost_matrix

    selected = mask.nonzero(as_tuple=True)[0]
   
    # Compute semantic penalty for selected samples
    loss = 0.0
    for i in selected:
        y_hat = pseudo_labels[i].item()
        cost_vec = cost_matrix[y_hat].to(device)
        loss += torch.dot(pred2[i], cost_vec)

    return loss / len(selected) , cost_matrix


def align_cost_regularizer(pred_tar, pseudo_labels, cost_matrix):
    """
    Semantic regularizer for alignment loss (penalizes target predictions far from pseudo-labels).

    Args:
        pred_tar: [B, K] - softmax output of domain classifier (GRL) on target
        pseudo_labels: [B] - hard pseudo-labels from teacher
        cost_matrix: [K, K] - precomputed class-aware cost matrix

    Returns:
        Scalar regularization loss
    """
    B, K = pred_tar.size()
    device = pred_tar.device

    loss = 0.0
    cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32).to(device)
    for i in range(B):
        y_hat = pseudo_labels[i].item()
        loss += torch.dot(pred_tar[i], cost_matrix[y_hat])

    return loss / B
    

def train_loop(args, train_src_dataset, train_tar_dataset, test_loader, box_src_loader, box_tar_loader,
               teacher_model, criterion, t_optimizer, t_scheduler, t_scaler):
    logger = logging.getLogger(__name__)
    logger.info("***** Running Training *****")
    logger.info(time.time())
    # Load Word2Vec and cost matrix ONCE
    w2v_model = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
    class_names = [str(i) for i in range(args.num_classes)]
    cost_matrix = build_cost_matrix(class_names, w2v_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w2v_tensor = torch.tensor(get_class_embeddings(class_names, w2v_model), dtype=torch.float32).to(device)


    src_data_loaders = []
    for i in range(0, len(train_src_dataset)):
        src_data_loaders.append(ContinuousDataloader(train_src_dataset[i], args))
    tar_data_loader = ContinuousDataloader(train_tar_dataset, args, is_tar=True)


    label_record = torch.zeros(len(args.src) + 1, len(train_tar_dataset), args.num_classes).cuda()
    label_record_num = torch.zeros(len(args.src) + 1, len(train_tar_dataset)).cuda()

    mix_source_feature_bank = []
    mix_source_score_bank = []
    base_position = [0]
    for i in range(0, len(args.src)):
        mix_source_feature_bank.append(torch.zeros([len(train_src_dataset[i]), args.bottleneck_dim]))
        mix_source_score_bank.append(torch.zeros(len(train_src_dataset[i])))
        base_position.append(base_position[i] + len(train_src_dataset[i]))

    ema_similar = [1 for _ in range(0, len(args.src))]
    args.branch_weights = torch.ones(len(train_tar_dataset), len(args.src)).cuda() / len(args.src)
    for step in range(args.start_step, args.total_steps):
        if step == args.start_step:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            mean_acc = AverageMeter()
            esm_acc = AverageMeter()
            proxy1_acc = AverageMeter()
            t_losses_alighment = AverageMeter()
            t_losses_broadcast = AverageMeter()
            record_acc = [AverageMeter() for _ in range(len(args.src) + 1)]

        teacher_model.train()
        end = time.time()

        src_datas_w, src_labels, tar_datas_w, tar_datas_s, tar_datas_m, tar_l, src_indexs, tar_index = get_data(
            src_data_loaders, tar_data_loader, args)

        data_time.update(time.time() - end)
        branch_weights = args.branch_weights

        with amp.autocast(enabled=args.amp):
            if step > args.wait_step:
                main_src_preds, esm_src_preds, main_src_features, \
                main_tar_preds, esm_tar_preds, main_tar_features, esm_tar_feas, tar_esm_weights, \
                main_aug_tar_preds, esm_aug_tar_preds, \
                src_dis, tar_dis, specific_src_advs, specific_tar_advs, specific_tar_advs_aug, \
                specific_src_proxy_outs, specific_tar_proxy_outs, specific_aug_tar_proxy_outs, \
                esm_src_proxy_outs, esm_tar_proxy_outs, esm_aug_tar_proxy_outs = teacher_model(src_datas_w, tar_datas_w, tar_datas_s)
            else:
                main_src_preds, esm_src_preds, main_src_features, \
                main_tar_preds, esm_tar_preds, main_tar_features, esm_tar_feas, tar_esm_weights, \
                main_aug_tar_preds, esm_aug_tar_preds, \
                src_dis, tar_dis, specific_src_advs, specific_tar_advs, specific_tar_advs_aug, \
                specific_src_proxy_outs, specific_tar_proxy_outs, specific_aug_tar_proxy_outs, \
                esm_src_proxy_outs, esm_tar_proxy_outs, esm_aug_tar_proxy_outs = teacher_model(src_datas_w, tar_datas_w, tar_datas_s, if_grl_step=False)

            # === Stable OT-based ensemble weights for novelty ===
            # === Curriculum adjustment of gamma based on training progress ===
            if step < args.total_steps * 0.3:
                gamma = args.gamma_explore
            elif step < args.total_steps * 0.6:
                gamma = args.gamma_mid
            else:
                gamma = args.gamma_refine

            src_weight = None
            avg_preds = 0
            aug_avg_preds = 0
            weight_preds = 0
            if src_weight is None:
                sim_weight = torch.ones(len(args.src)).float().cuda()
            else:
                sim_weight = src_weight.reshape(len(args.src), -1)
                sim_weight = sim_weight.sum(1)
            sim_weight /= sum(sim_weight)
            for ii in range(0, len(main_tar_preds)):
                avg_preds += nn.Softmax(dim=1)(main_tar_preds[ii])
                aug_avg_preds += nn.Softmax(dim=1)(main_aug_tar_preds[ii])
                weight_preds += branch_weights[tar_index][:, ii].unsqueeze(1) * nn.Softmax(dim=1)(main_tar_preds[ii]).detach()
            avg_preds /= len(main_tar_preds)
            aug_avg_preds /= len(main_tar_preds)

            mean_acc.update(sum((avg_preds.data.max(1)[1] == tar_l).float()).item() / len(avg_preds))
            esm_acc.update(sum((esm_tar_preds.data.max(1)[1] == tar_l).float()).item() / len(avg_preds))
            proxy1_acc.update(sum((weight_preds.data.max(1)[1] == tar_l).float()).item() / len(avg_preds))

            tar_training_loss, align_cost_matrix = ot_reg_cls(
                pred1=nn.Softmax(dim=1)(esm_tar_preds),
                pred2=nn.Softmax(dim=1)(esm_aug_tar_preds),
                avg_preds=weight_preds.detach(),
                class_heads=teacher_model.module.esm_head[-1].weight,     # Classifier heads from the model
                w2v_embeddings=w2v_tensor,               # Word2Vec class embeddings
                threshold=args.tar_threshold,
                alpha=args.ot_alpha                      # Blend between class heads and w2v
            )

            tar_alignment_loss = torch.zeros(1).cuda()
            for ii in range(0, len(args.src)):
                # Original CSR source-side alignment
                t_align_loss1 = 3 * F.cross_entropy(
                    specific_src_advs[ii],
                    main_src_preds[ii][ii].data.max(1)[1],
                    reduction='none').mean()

                # Original CSR target-side alignment (with GRL)
                t_align_loss2 = 0.0
                if args.consis_tar:
                    t_align_loss2 = F.nll_loss(
                        shift_log(1. - F.softmax(specific_tar_advs[ii], dim=1)),
                        esm_tar_preds.data.max(1)[1], reduction='none').mean()

                tar_alignment_loss += t_align_loss1 + t_align_loss2
                
                # OT-based semantic regularization on target alignment
                if args.use_align_cost_reg:
                    align_probs = F.softmax(specific_tar_advs[ii], dim=1)
                    pseudo_labels = esm_tar_preds.data.max(1)[1]
                    align_cost = align_cost_regularizer(
                        align_probs, pseudo_labels, align_cost_matrix
                    )
                    tar_alignment_loss += args.align_cost_weight * align_cost
                    logger.info('Step: {}\tAlign Cost: {:.4f}\tTarget_Align_loss: {:.4f}'.format(
                        step,
                        align_cost.item(),
                        tar_alignment_loss.item()
                    ))

                    
            t_src_cls_loss = 0
            for ii in range(0, len(main_src_preds)):
                t_src_cls_loss += criterion(main_src_preds[ii][ii], src_labels[ii])
            t_src_cls_loss += criterion(torch.cat(esm_src_preds), torch.cat(src_labels))

            if step > args.wait_step:
                t_loss = t_src_cls_loss + args.tar_alignment_weight*tar_alignment_loss + args.tar_weight * tar_training_loss
            else:
                t_loss = t_src_cls_loss

        t_scaler.scale(t_loss).backward()
        t_scheduler.step(step)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        teacher_model.zero_grad()
        torch.cuda.empty_cache()

        if args.world_size > 1:
            src_cls_loss = reduce_tensor(t_src_cls_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            align_loss = reduce_tensor(tar_alignment_loss.detach(), args.world_size)
            tar_cls_loss = reduce_tensor(tar_training_loss.detach(), args.world_size)
        else:
            src_cls_loss = t_src_cls_loss.detach()
            t_loss = t_loss.detach()
            align_loss = tar_alignment_loss.detach()
            tar_cls_loss = tar_training_loss.detach()

        s_losses.update(src_cls_loss.item())
        t_losses.update(t_loss.item())
        t_losses_broadcast.update(tar_cls_loss.item())
        t_losses_alighment.update(align_loss.item())
        batch_time.update(time.time() - end)

        args.num_eval = step // args.eval_step
        if step % 250 == 0 or step == args.total_steps - 1:
            new_branch_weights = update_weight(args, test_loader=box_tar_loader, model=teacher_model, cost_matrix=cost_matrix)
            args.branch_weights = new_branch_weights
            if args.local_rank in [-1, 0]:
                
                logger.info("***********************")
                eval_loss, top1, top5, _, label_record, label_record_num = evaluate(args, test_loader, teacher_model,
                                                                                    criterion, label_record,
                                                                                    label_record_num)
                logger.info('steps: {step}\t'
                            'all_loss: {all_loss.avg:.4f}\t'
                            'src_cls_loss: {src_cls_loss.avg:.4f}\t'
                            'align_loss: {align_loss.avg:.4f}\t'
                            'tar_cls_loss: {tar_cls_loss.avg:.4f}\t'
                            'eval_loss: {eval_loss:.4f}\t'
                            'Prec@1 {top1:.4f}'.format(
                    step=step, all_loss=t_losses, src_cls_loss=s_losses, align_loss=t_losses_alighment,
                    tar_cls_loss=t_losses_broadcast, eval_loss=eval_loss, top1=top1))
                print('steps: {step}\t'
                            'all_loss: {all_loss.avg:.4f}\t'
                            'src_cls_loss: {src_cls_loss.avg:.4f}\t'
                            'align_loss: {align_loss.avg:.4f}\t'
                            'tar_cls_loss: {tar_cls_loss.avg:.4f}\t'
                            'eval_loss: {eval_loss:.4f}\t'
                            'Prec@1 {top1:.4f}'.format(
                    step=step, all_loss=t_losses, src_cls_loss=s_losses, align_loss=t_losses_alighment,
                    tar_cls_loss=t_losses_broadcast, eval_loss=eval_loss, top1=top1))
                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5
                logger.info(f"Best top-1 acc: {args.best_top1:.4f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'teacher_scaler': t_scaler.state_dict()
                }, is_best)

                batch_time = AverageMeter()
                data_time = AverageMeter()
                s_losses = AverageMeter()
                t_losses = AverageMeter()
                mean_acc = AverageMeter()
                proxy1_acc = AverageMeter()
                esm_acc = AverageMeter()
                t_losses_alighment = AverageMeter()
                t_losses_broadcast = AverageMeter()
                record_acc = [AverageMeter() for _ in range(len(args.src) + 1)]

    return


def update_weight(args, test_loader, model, cost_matrix):
    all_branch_outs = [[] for _ in range(len(args.src))]
    all_esm_feas = []
    test_iter = test_loader
    model.eval()
    with torch.no_grad():
        for step, (images, targets, index) in enumerate(test_iter):
            images = images.cuda()
            with amp.autocast(enabled=args.amp):
                tar_outs, esm_outs, _, esm_fea = model(images, is_eval=True)
                all_esm_feas.append(esm_fea.detach())
                for ii in range(0, len(args.src)):
                    all_branch_outs[ii].append(nn.Softmax(dim=1)(tar_outs[ii]).detach())

    avg_preds = sum(torch.cat(out) for out in all_branch_outs) / len(all_branch_outs)
    assert 0.99 < avg_preds[0].sum() < 1.01
    avg_preds_np = avg_preds.cpu().numpy()

    all_branch_instance_scores = []

    for i in range(len(args.src)):
        all_outputs = torch.cat(all_branch_outs[i])  # (N, C)
        outputs_np = all_outputs.cpu().numpy()

        # Confidence = max prob
        confidence = np.max(outputs_np, axis=1)

        # Diversity = entropy
        pred_clamped = np.clip(outputs_np, 1e-8, 1.0)
        diversity = -(pred_clamped * np.log(pred_clamped)).sum(axis=1)

        scores = []

        for idx, (pred, conf, div) in enumerate(zip(outputs_np, confidence, diversity)):
            try:
                if args.use_sinkhorn:
                    # Ensure inputs are torch tensors and normalized
                    p1 = torch.tensor(pred, dtype=torch.float32).cuda()
                    p2 = torch.tensor(avg_preds_np.mean(axis=0), dtype=torch.float32).cuda()
                    p1 = p1 / (p1.sum() + 1e-8)
                    p2 = p2 / (p2.sum() + 1e-8)

                    if not torch.is_tensor(cost_matrix):
                        C = torch.tensor(cost_matrix, dtype=torch.float32).cuda()
                    else:
                        C = cost_matrix.cuda() if not cost_matrix.is_cuda else cost_matrix

                       # Use unbalanced Sinkhorn
                    # with open("sinkhorn_device_log.txt", "a") as f:
                    #     f.write(f"[Step {step}] Sinkhorn devices:\n")
                    #     f.write(f"  p1.device = {p1.device}, p2.device = {p2.device}, cost_matrix = {C.device}\n")
                    T = ot.unbalanced.sinkhorn_knopp_unbalanced(
                        p1, p2, C, 
                        reg=args.sinkhorn_reg, 
                        reg_m=args.unbalanced_mass_reg,  # e.g., 0.1 or 0.01
                        reg_type='kl',  # 'kl' or 'l2' (KL is typical)
                        numItermax=1000,
                        stopThr=1e-6,
                        log=False
                    )

                    # Compute OT cost: <T, C> = sum(T * C)
                    wdist = torch.sum(T * C).item()
                    #logger.info(f"OT distance for sample {idx}: {wdist}")
                    # print("IDX, Wdist", idx, wdist)
                else:
                    wdist = ot.emd2(pred, avg_preds_np.mean(axis=0), cost_matrix)
                    #logger.info(f"OT distance for sample {idx}: {wdist}")
            except Exception as e:
                logger.warning(f"OT failed for sample {idx}, fallback to wdist=1.0. Error: {e}")
                wdist = 1.0

            score = -args.ot_alpha * wdist + args.ot_beta * conf + args.ot_gamma * div
            scores.append(score)

        score_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)
        all_branch_instance_scores.append(score_tensor)

    all_branch_instance_scores = torch.cat(all_branch_instance_scores, dim=1)  # [N, num_sources]
    all_branch_instance_scores = nn.Softmax(dim=1)(all_branch_instance_scores / args.weight_tau)
    #print randomly 100 samples instance scores
    
    print("OT-weighted branch scores (mean per source): " + str(all_branch_instance_scores.mean(0)))
    return all_branch_instance_scores.cuda()


# === OT/Word2Vec utilities ===
def get_class_embeddings(class_names, w2v_model):
    embeddings = []
    for cls in class_names:
        if cls in w2v_model:
            embeddings.append(w2v_model[cls])
        else:
            embeddings.append(np.random.randn(w2v_model.vector_size))
    return np.stack(embeddings)

def build_cost_matrix(class_names, w2v_model):
    emb = get_class_embeddings(class_names, w2v_model)
    sim = 1 - np.dot(emb, emb.T) / (np.linalg.norm(emb, axis=1)[:,None] * np.linalg.norm(emb, axis=1)[None,:])
    return sim



def evaluate(args, test_loader, model, criterion, label_record=None, label_record_num=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    ftop1 = AverageMeter()
    top5 = AverageMeter()
    brach_top1 = []
    for i in range(0, len(args.src) + 1):
        brach_top1.append(AverageMeter())
    model.eval()
    test_iter = test_loader
    tar_predict = torch.zeros(len(test_iter.dataset)).long().cuda()
    all_branch_outs = []
    for i in range(0, len(args.src)):
        all_branch_outs.append([])

    with torch.no_grad():
        end = time.time()
        for step, (images, targets, index) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.cuda()  # .to(args.device)
            targets = targets.cuda()  # to(args.device)
            with amp.autocast(enabled=args.amp):
                tar_outs, esm_outs, _, esm_fea = model(images, is_eval=True)
                tar_outs.append(esm_outs)
                if label_record is not None:
                    for ii in range(0, len(tar_outs)):
                        label_record[ii][index] = label_record[ii][index] + nn.Softmax(dim=1)(tar_outs[ii])
                        label_record_num[ii][index] = 1 + label_record_num[ii][index]
                loss = 0
                outputs = 0

                for ii in range(0, len(tar_outs)):
                    loss += criterion(tar_outs[ii], targets)
                    tpred = nn.Softmax(dim=1)(tar_outs[ii])
                    outputs += tpred

                foutputs = nn.Softmax(dim=1)(tar_outs[-1]) * len(args.src)
                for ii in range(0, len(tar_outs) - 1):
                    tpred = nn.Softmax(dim=1)(tar_outs[ii])
                    foutputs += tpred

            for ii in range(0, len(tar_outs)):
                tacc1, tacc5 = accuracy(tar_outs[ii], targets, (1, 5))
                brach_top1[ii].update(tacc1[0], batch_size)
            # tacc1, tacc5 = accuracy(tar_esamble, targets, (1, 5))
            # brach_top1[-1].update(tacc1[0], batch_size)

            outputs /= (len(tar_outs))
            foutputs /= (len(tar_outs) - 1) * 2
            tar_predict[index] = outputs.data.max(1)[1]
            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            facc1, facc5 = accuracy(foutputs, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            ftop1.update(facc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
        all_accs = []
        for i in range(0, len(brach_top1)):
            all_accs.append(brach_top1[i].avg)
        logger.info("brach avg: " + str(all_accs))
        logger.info("weight avg: " + str(ftop1.avg))


        return losses.avg, top1.avg, top5.avg, tar_predict, label_record, label_record_num


def main(rank, world_size, args):
    print('==> Start rank:', rank)

    local_rank = rank % 8
    args.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}', world_size=world_size,
                            rank=rank)
    args.batch_size = int(args.batch_size / world_size)

    # args.device = torch.device('cuda', args.gpu)

    logging.basicConfig(
        filename=args.log_file_name,  # Logs to file
        filemode='w',          # Overwrites previous logs (use 'a' to append)
    
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    # if args.local_rank not in [-1, 0]:
    #    torch.distributed.barrier()

    train_src_dataset, train_tar_dataset, test_dataset, finetune_dataset, box_src_dataset, box_tar_dataset = load_dataset(args)
    # if args.local_rank == 0:
    #    torch.distributed.barrier()
    # Log dataset sizes to confirm split
    logger.info(f"Train src datasets: {len(train_src_dataset)} domains, sizes: {[len(ds) for ds in train_src_dataset]}")
    logger.info(f"Train tar dataset size: {len(train_tar_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Total amazon images (train + test): {len(train_tar_dataset) + len(test_dataset)}")

    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.eval_batch_size,
                             num_workers=args.workers)
    box_src_loader = []
    for ii in range(0, len(args.src)):
        box_src_loader.append(DataLoader(box_src_dataset[ii],
                                    sampler=SequentialSampler(box_src_dataset[ii]),
                                    batch_size=args.eval_batch_size // 2,
                                    num_workers=args.workers))
    box_tar_loader = DataLoader(box_tar_dataset,
                             sampler=SequentialSampler(box_tar_dataset),
                             batch_size=args.eval_batch_size // 2,
                             num_workers=args.workers)

    # if args.local_rank not in [-1, 0]:
    #    torch.distributed.barrier()

    teacher_model = get_model(args)

    # if args.local_rank == 0:
    #    torch.distributed.barrier()

    logger.info(f"Model: args.model")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M")

    teacher_model.cuda()  # to(args.device)

    criterion = create_loss_fn(args)

    no_decay = ['bn']
    teacher_parameters = teacher_model.get_params(args, no_decay)
    t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)

    t_scheduler = LrScheduler(t_optimizer, args.teacher_lr, args.total_steps, 0, args)
    t_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    teacher_model = nn.parallel.DistributedDataParallel(
        teacher_model, device_ids=[local_rank], find_unused_parameters=True)

    teacher_model.zero_grad()

    # print(torch.cuda.memory_allocated()/1024/1024/1024)
    
    train_loop(args, train_src_dataset, train_tar_dataset, test_loader, box_src_loader, box_tar_loader,
               teacher_model, criterion, t_optimizer, t_scheduler, t_scaler)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.

    world_size = 1
    args.world_size = world_size
    mp.spawn(main, nprocs=world_size, args=(world_size, args))

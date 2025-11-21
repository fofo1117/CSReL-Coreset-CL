# -*-coding:utf8-*-

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle
import copy

from dataset import single_task_dataset
from functions import loss_functions


def random_select(id2cnt, select_size):
    selected_id2prob = {}
    all_ids = list(id2cnt.keys())
    selected_ids = set(random.sample(all_ids, select_size))
    for d_id in selected_ids:
        selected_id2prob[d_id] = 1.0
    return selected_ids, selected_id2prob


def add_new_data(data_file, new_data):
    ori_data = []
    ori_ids = set()
    if os.path.exists(data_file):
        with open(data_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    d_id = di[0]
                    ori_ids.add(int(d_id))
                    ori_data.append(di)
                except EOFError:
                    break
    all_data = ori_data
    for di in new_data:
        d_id = int(di[0])
        if d_id not in ori_ids:
            all_data.append(di)
    random.shuffle(all_data)
    with open(data_file, 'wb') as fw:
        for di in all_data:
            pickle.dump(di, fw)


class FeatureSpaceAttacker(nn.Module):
    """Small network that learns feature-space perturbations."""

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        hidden_dim = max(hidden_dim, 32)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, features):
        # Bound perturbations between -1 and 1 to keep updates small.
        return torch.tanh(self.net(features))


def _make_linear_classifier_forward(model):
    """Return a function that maps features to logits using the model head."""

    head = None
    for attr in ['classifier', 'linear', 'fc', 'head']:
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if isinstance(candidate, nn.Linear):
                head = candidate
                break
    if head is None:
        return None
    weight = head.weight.detach().clone()
    bias = head.bias.detach().clone() if head.bias is not None else None

    def forward_fn(features):
        local_w = weight.to(features.device)
        local_b = bias.to(features.device) if bias is not None else None
        return F.linear(features, local_w, local_b)

    return forward_fn


def _extract_features(model, inputs):
    """Try different hooks to read the last hidden representation."""

    with torch.no_grad():
        if hasattr(model, 'features'):
            try:
                feats = model.features(inputs)
                return feats.detach()
            except TypeError:
                pass
        if hasattr(model, 'embed'):
            return model.embed(inputs).detach()
        try:
            outputs = model(inputs, returnt='all')
            if isinstance(outputs, tuple) and len(outputs) == 2:
                return outputs[1].detach()
        except TypeError:
            pass
    raise RuntimeError('Cannot extract features from the current model')


def _setup_feature_attacker(state, feature_dim, device, attack_params):
    if state['attacker'] is None:
        hidden_dim = attack_params.get('hidden_dim', min(512, feature_dim * 2))
        attacker = FeatureSpaceAttacker(feature_dim=feature_dim, hidden_dim=hidden_dim)
        attacker.to(device)
        lr = attack_params.get('lr', 1e-3)
        optimizer = torch.optim.Adam(attacker.parameters(), lr=lr)
        state['attacker'] = attacker
        state['optimizer'] = optimizer


def _compute_adversarial_losses(model, inputs, labels, loss_fn, attack_state, attack_params):
    if attack_state is None:
        return None
    try:
        features = _extract_features(model, inputs)
    except RuntimeError:
        return None
    device = features.device
    _setup_feature_attacker(attack_state, features.shape[1], device, attack_params)
    attacker = attack_state['attacker']
    optimizer = attack_state['optimizer']
    classifier_forward = attack_state['classifier_forward']
    epsilon = attack_params.get('epsilon', 0.05)
    attack_steps = attack_params.get('train_steps', 3)
    attacker.train()
    with torch.enable_grad():
        for _ in range(attack_steps):
            delta = attacker(features)
            adv_features = features + epsilon * delta
            logits = classifier_forward(adv_features)
            adv_loss = loss_fn(x=logits, y=labels, logits=None).mean()
            objective = -adv_loss
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
    attacker.eval()
    with torch.no_grad():
        base_delta = attacker(features)
    eps_scales = attack_params.get('epsilon_scales', [1.0])
    all_losses = []
    for scale in eps_scales:
        adv_feats = features + (epsilon * scale) * base_delta
        logits = classifier_forward(adv_feats)
        adv_loss = loss_fn(x=logits, y=labels, logits=None)
        all_losses.append(adv_loss)
    stacked = torch.stack(all_losses, dim=0)
    worst_loss, _ = torch.max(stacked, dim=0)
    return worst_loss


def select_by_loss_diff(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda, loss_params,
                        class_sizes=None, attack_params=None):
    """Select samples whose loss increases the most under clean or adversarial views.

    中文说明：在默认情况下，函数会比较样本当前的预测损失与历史基准损失，
    按照增长幅度排序挑选最容易遗忘的样本；当 `attack_params` 被提供时，
    会先训练一个小型特征空间攻击器在模型最后一层表征上制造有界扰动，
    计算攻击后损失与干净损失的差值，并以该“脆弱度”作为排序依据。换言之，
    在有攻击的模式下，越是对特征扰动敏感、攻击后损失上升越大的样本就越优先被选入核心集，
    以便后续增量学习时重点巩固这些易受干扰的知识点。
    """
    status = model.training
    model.eval()
    if on_cuda:
        model.cuda()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'], mse_factor=loss_params['mse_factor'], reduction='none')
    attack_state = None
    if attack_params is not None:
        classifier_forward = _make_linear_classifier_forward(model)
        if classifier_forward is None:
            print('Warning: attack params provided but model head is not supported; fall back to vanilla selection.')
        else:
            attack_state = {
                'attacker': None,
                'optimizer': None,
                'classifier_forward': classifier_forward
            }
    loss_diffs = {}
    id2pos = {}
    id2logits = {}
    batch_ids = []
    batch_sps = []
    batch_labs = []
    batch_logits = []
    with torch.no_grad():
        for i, di in enumerate(rand_data):
            if len(di) == 4:
                d_id, sp, lab, logit = di
            else:
                d_id, sp, lab = di
                logit = None
            id2pos[d_id] = i
            if transforms is not None:
                aug_sp = torch.unsqueeze(transforms(sp), dim=0)
            else:
                aug_sp = torch.unsqueeze(sp, dim=0)
            batch_ids.append(d_id)
            batch_sps.append(aug_sp)
            batch_labs.append(int(lab))
            if logit is not None:
                batch_logits.append(
                    torch.unsqueeze(torch.tensor(logit, dtype=torch.float32), dim=0)
                )
            if i % 32 == 0 or i == len(rand_data) - 1:
                sps = torch.cat(batch_sps, dim=0)
                labs = torch.tensor(batch_labs, dtype=torch.long)
                if len(batch_logits) > 0:
                    lab_logits = torch.cat(batch_logits, dim=0)
                else:
                    lab_logits = None
                if on_cuda:
                    sps = sps.cuda()
                    labs = labs.cuda()
                    if lab_logits is not None:
                        lab_logits = lab_logits.cuda()
                clean_loss = loss_fn(x=model(sps), y=labs, logits=lab_logits)
                adv_loss = None
                if attack_state is not None:
                    adv_loss = _compute_adversarial_losses(
                        model=model,
                        inputs=sps,
                        labels=labs,
                        loss_fn=loss_fn,
                        attack_state=attack_state,
                        attack_params=attack_params
                    )
                loss = clean_loss.clone().detach()
                if on_cuda:
                    loss = loss.cpu()
                loss = loss.numpy()
                if adv_loss is not None:
                    adv_loss = adv_loss.clone().detach()
                    if on_cuda:
                        adv_loss = adv_loss.cpu()
                    adv_loss = adv_loss.numpy()
                if lab_logits is not None:
                    if on_cuda:
                        lab_logits = lab_logits.cpu()
                    lab_logits = lab_logits.clone().detach().numpy()
                for j in range(len(batch_labs)):
                    did = batch_ids[j]
                    if adv_loss is not None:
                        loss_dif = float(adv_loss[j] - loss[j])
                    else:
                        loss_dif = float(loss[j] - ref_loss_dic[did])
                    loss_diffs[did] = loss_dif
                    if lab_logits is not None:
                        id2logits[did] = lab_logits[j, :]
                batch_ids.clear()
                batch_sps.clear()
                batch_labs.clear()
                batch_logits.clear()
                del lab_logits
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    selected_data = []
    id2loss_dif = {}
    class_cnt = {}
    if class_sizes is not None:
        for ci in class_sizes.keys():
            class_cnt[ci] = 0
    for i in range(len(sorted_loss_diffs)):
        d_id = sorted_loss_diffs[i][0]
        pos = id2pos[d_id]
        di = rand_data[pos]
        if class_sizes is not None:
            lab = int(di[2])
            if class_cnt[lab] == class_sizes[lab]:
                continue
            else:
                class_cnt[lab] += 1
        new_di = copy.deepcopy(di)
        if loss_params['mse_factor'] > 0 and len(di) < 4:
            new_di.append(id2logits[d_id])
        selected_data.append(new_di)
        id2loss_dif[d_id] = sorted_loss_diffs[i][1]
        if len(selected_data) == incremental_size:
            break
    if on_cuda:
        model.cpu()
    model.train(status)
    return selected_data, id2loss_dif

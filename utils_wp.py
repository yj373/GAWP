import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20

def mart_loss(model,
              x_natural,
              x_adv,
              y,
              beta=6.0):

    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(x_natural)
    logits = model(x_natural)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss


def diff_in_weights(proxy_1, proxy_2):
    diff_dict = OrderedDict()
    proxy_1_state_dict = proxy_1.state_dict()
    proxy_2_state_dict = proxy_2.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(proxy_1_state_dict.items(), proxy_2_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = diff_w
    return diff_dict


def add_into_diff(model, diff_step, diff):
    diff_scale = OrderedDict()
    if not diff:
        diff = diff_step
        names_in_diff = diff_step.keys()
        diff_squeue = []
        w_squeue = []
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_squeue.append(diff[name].view(-1))
                w_squeue.append(param.view(-1))
        diff_squeue_all = torch.cat(diff_squeue)
        w_squeue_all = torch.cat(w_squeue)
        scale_value = w_squeue_all.norm() / (diff_squeue_all.norm() + EPS)
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_scale[name] = scale_value * diff[name]
    else:
        names_in_diff = diff_step.keys()
        for name in names_in_diff:
            diff[name] = diff[name] + diff_step[name]
        diff_squeue = []
        w_squeue = []
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_squeue.append(diff[name].view(-1))
                w_squeue.append(param.view(-1))
        diff_squeue_all = torch.cat(diff_squeue)
        w_squeue_all = torch.cat(w_squeue)
        scale_value = w_squeue_all.norm() / (diff_squeue_all.norm() + EPS)
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_scale[name] = scale_value * diff[name]
    return diff, diff_scale


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    # print(coeff)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def set_lr(optim, lr=0.1):
    for g in optim.param_groups:
        g['lr'] = lr


class WeightPerturb(object):
    def __init__(self, model, proxy_1, proxy_2, proxy_2_optim, gamma, K2):
        super(WeightPerturb, self).__init__()
        self.model = model
        self.proxy_1 = proxy_1
        self.proxy_2 = proxy_2
        self.proxy_2_optim = proxy_2_optim
        self.gamma = gamma
        self.K2 = K2

    def calc_wp(self, inputs_adv, targets, threshold=-1, weight=None, func='ce', inputs_clean=None, beta=6.0, 
                kappa=None, kappa_threshold=-1, filter_kappa_larger=False, ortho=False):
        diff = OrderedDict()
        diff_scale = OrderedDict()
        diff2 = OrderedDict()
        diff_scale2 = OrderedDict()

        # weight perturbation attack
        for ii in range(self.K2):
            self.proxy_1.load_state_dict(self.model.state_dict())
            self.proxy_2.load_state_dict(self.model.state_dict())
            add_into_weights(self.proxy_1, diff_scale, coeff=1.0 * self.gamma)
            add_into_weights(self.proxy_2, diff_scale, coeff=1.0 * self.gamma)
            self.proxy_2.train()
            output = self.proxy_2(inputs_adv)
            if func == 'ce':
                loss = nn.CrossEntropyLoss(reduce=False)(output, targets)
            elif inputs_clean is not None:
                if func == 'kl':
                    loss_natural = nn.CrossEntropyLoss(reduce=False)(self.proxy_2(inputs_clean), targets)
                    loss_robust = F.kl_div(F.log_softmax(self.proxy_2(inputs_adv), dim=1), F.softmax(self.proxy_2(inputs_clean), dim=1), reduction='none').sum(dim=1)
                    # print(loss_natural.shape)
                    # print(loss_robust.shape)
                    loss = loss_natural + loss_robust * beta
                else:
                    loss = mart_loss(self.proxy_2, inputs_clean, inputs_adv, targets, beta)

            Indicator = None
            if threshold > 0:
                Indicator = (loss < threshold).cuda().type(torch.cuda.FloatTensor)
            Indicator_kappa = None
            if kappa is not None and kappa_threshold > 0 & kappa_threshold < 10:
                if filter_kappa_larger:
                    Indicator_kappa = (kappa < kappa_threshold).cuda().type(torch.cuda.FloatTensor)
                else:
                    Indicator_kappa = (kappa >= kappa_threshold).cuda().type(torch.cuda.FloatTensor)
            if Indicator is not None and Indicator_kappa is not None:
                Indicator = Indicator * Indicator_kappa
                loss = loss.mul(Indicator).mean()
            elif Indicator is not None or Indicator_kappa is not None:
                Indicator = Indicator_kapp if Indicator is None else Indicator
                loss = loss.mul(Indicator).mean()
            else:
                if weight is None:
                    loss = loss.mean()
                else:
                    loss = (loss * weight).mean()
            loss = -1 * loss
                
            self.proxy_2_optim.zero_grad()
            loss.backward()
            self.proxy_2_optim.step()

            diff_step = diff_in_weights(self.proxy_1, self.proxy_2)
            diff, diff_scale = add_into_diff(self.model, diff_step, diff)

        if ortho:
            names_in_diff = diff_scale.keys()
            for name, param in model.named_parameters():
                if name in names_in_diff:
                    delta = (torch.randint(0, 2, size=diff_scale[name], dtype=torch.float) * 2 - 1).cuda()
                    delta = delta.view(-1)
                    delta -= torch.dot(delta, diff_scale[name]) * diff_scale[name].view(-1)
                    delta = delta.reshape(diff_scale[name].shape)
                    delta = delta / torch.norm(delta) * torch.norm(diff_scale[name])
                    diff_scale[name] = delta
        return diff_scale

    def perturb(self, diff, c=1.0):
        add_into_weights(self.model, diff, coeff= c * self.gamma)

    def restore(self, diff, coeff=None):
        if coeff is None:
            c = -1.0 * self.gamma
        else:
            c = coeff
        add_into_weights(self.model, diff, coeff=c)
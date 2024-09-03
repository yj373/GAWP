import argparse
from cgi import test
import inspect
import os
from re import A
import numpy as np
import torch
import torchvision.models as torch_models
import time

from utils.helper import get_dataset
from models.preact_resnet import PreActResNet18
from models.wide_resnet import WideResNet_34_10, WideResNet_28_10, WideResNet_28_4
from models.resnet import resnet50
# from utils.attacks import *
from utils.helper import *
# from utils.robust_eval import *
from torchsummary import summary
from utils_wp import WeightPerturb

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False, func='ce', ds='cifar10'):
    
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    kappa = torch.zeros(len(X))
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        # print('>>>>>>>')
        for i in range(attack_iters):
            output = model(normalize(X + delta, ds=ds))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            correct_index = torch.where(output.max(1)[1] == y)[0]
            kappa[correct_index] += 1
            if func == 'ce':
                loss = F.cross_entropy(output, y, reduction='none')
            else:
                raise ValueError
            if func == 'ce':
                loss = loss.mean()
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            step_size = alpha
            if norm == "l_inf":
                d = torch.clamp(d + step_size * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
            
        all_loss = F.cross_entropy(model(normalize(X+delta, ds=ds)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    
    return max_delta, kappa


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--data', default='cifar10')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save-dir', default='results', type=str)
    parser.add_argument('--ckpt-iters', default=10, type=int)
    parser.add_argument('--load-init', type=str, default='')
    parser.add_argument('--batch-size', default=128, type=int)

    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--lr-schedule', default='piecewise', type=str)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--lr-init', default=0.1, type=float)
    parser.add_argument('--lr-step', default=50, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=2e-4, type=float)

    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--epsilon', default=8, type=float)
    parser.add_argument('--eval-attack-iters', default=10, type=int)
    parser.add_argument('--eval-pgd-alpha', default=2, type=float)
    parser.add_argument('--eval-epsilon', default=8, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--restarts', default=1, type=int)

    parser.add_argument('--wp', action='store_true')
    parser.add_argument('--gawp', action='store_true')
    parser.add_argument('--ga-lambda', default=-1.0, type=float)
    parser.add_argument('--wp-gamma', default=0.01, type=float)
    parser.add_argument('--wp-warmup', default=0, type=int)
    parser.add_argument('--wp-threshold', default=1.5, type=float)
    parser.add_argument('--wp-K2', default=1, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    best_test_robust_acc = 0
    epsilon = args.epsilon / 255.
    eval_epsilon = args.eval_epsilon / 255.
    alpha = args.pgd_alpha / 255.
    eval_alpha = args.eval_pgd_alpha / 255.
    attack_iters = args.attack_iters
    eval_attack_iters = args.eval_attack_iters
    save_metric_name = 'pgd{}'.format(args.eval_attack_iters)

    train_loader, test_loader = get_dataset(name=args.data, root=args.data_dir, batch_size=args.batch_size)
    num_classes = 10
    in_size=32
    if args.data == 'cifar100':
        num_classes = 100
    elif args.data == 'tinyimagenet':
        num_classes = 200
        in_size = 64
    elif args.data == 'mnist':
        in_size = 28

    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_classes, in_size=in_size)
        if args.wp:
            proxy = PreActResNet18(num_classes=num_classes, in_size=in_size)
            proxy_proxy = PreActResNet18(num_classes=num_classes, in_size=in_size)
    elif args.model == 'WideResNet34-10':
        model = WideResNet_34_10(num_classes=num_classes)
        if args.wp:
            proxy = WideResNet_34_10(num_classes=num_classes)
            proxy_proxy = WideResNet_34_10(num_classes=num_classes)
    elif args.model == 'WideResNet28-10':
        model = WideResNet_28_10(num_classes=num_classes).cuda()
    elif args.model == 'WideResNet28-4':
        model = WideResNet_28_4(num_classes=num_classes).cuda()
    elif args.model == 'CNN':
        model = LeNet(dataset=args.data).cuda()
    elif args.model == 'resnet50':
        model = resnet50(pretrained=True).cuda()
    else:
        raise ValueError('{} model is invalid'.format(args.model))

    model = nn.DataParallel(model).cuda()
    if args.wp:
        proxy = nn.DataParallel(proxy).cuda()
        proxy_proxy = nn.DataParallel(proxy_proxy).cuda()

    if os.path.isfile(args.load_init):
        model_dict = torch.load(args.load_init, map_location='cuda:0')
        try:
            # model.load_state_dict(model_dict['state_dict'])
            model.load_state_dict(model_dict)
            print('loaded checkpoint: {}'.format(args.load_init))
        except:
            # model.load_state_dict(model_dict)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_init.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_init.pth'))

    params = model.parameters()
    opt = torch.optim.SGD(params, lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.wp:
        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=args.wp_gamma)
        wp_adversary = WeightPerturb(model=model, proxy_1=proxy_proxy, proxy_2=proxy, proxy_2_optim=proxy_opt, gamma=args.wp_gamma, K2=args.wp_K2)

    epochs = args.epochs
    if args.lr_schedule == 'piecewise':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=args.lr_gamma)
    elif args.lr_schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, args.epochs, eta_min=args.lr_init * args.lr_gamma * args.lr_gamma
        )
    elif args.lr_schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=args.lr_step, gamma=args.lr_gamma
        )
    else:
        lr_scheduler = None

    for epoch in range(1, args.epochs+1):
        start = time.time()
        model.train()
        train_acc_meter = AverageMeter(name='clean_train_acc')
        train_robust_loss_meter = AverageMeter(name='adversarial_train_loss')
        train_robust_acc_meter = AverageMeter(name='adversarial_train_acc')

        for i, (data, label) in enumerate(train_loader):
            data, label = data.cuda(), label.cuda()
            # print(self.alpha)
            delta, kappa = attack_pgd(model, data, label, epsilon, alpha, attack_iters, 
                    args.restarts, args.norm, ds=args.data)
            delta = delta.detach()
            adv_data = normalize(torch.clamp(data + delta[:data.size(0)], min=lower_limit, max=upper_limit), ds=args.data)

            model.train()
            if args.wp and epoch > args.wp_warmup:
                if args.gawp:
                    with torch.no_grad():
                        robust_output_sat_before_wp = model(adv_data)
                        robust_loss_ce_before_wp = F.cross_entropy(robust_output_sat_before_wp, label, reduction='none')
                    wp_weight_1 = (1.0 - F.softmax(robust_loss_ce_before_wp, dim=0)) * data.size(0) # small loss
                    kappa = kappa.cuda()
                    weight = ((args.ga_lambda + 5 * (1 - 2 * kappa / attack_iters)).tanh() + 1) / 2 
                    wp_weight_2 =  weight * data.size(0) / weight.sum() # closer 
                    wp_weight = (wp_weight_1 + wp_weight_2) / 2.0
                    wp_threshold = -1
                else:
                    # RWP (AWP when wp_threshold = -1)
                    wp_weight = None
                    wp_threshold = args.wp_threshold

                wp = wp_adversary.calc_wp(inputs_adv=adv_data, targets=label, threshold=wp_threshold, weight=wp_weight, func='ce')
                wp_adversary.perturb(wp)
                    
            robust_output = model(adv_data)
            robust_loss = F.cross_entropy(robust_output, label)
            robust_acc = accuracy(robust_output.data, label)[0]

            # inspect clean acc
            model.eval()
            with torch.no_grad():
                output = model(normalize(data, ds=args.data))
                clean_acc = accuracy(output.data, label)[0]
            model.train()

            train_acc_meter.update(clean_acc.item(), data.size(0))
            train_robust_loss_meter.update(robust_loss.item(), data.size(0))
            train_robust_acc_meter.update(robust_acc.item(), data.size(0))

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            if args.wp and epoch > args.wp_warmup:
                wp_adversary.restore(wp)

            if i % 100 == 0:
                log_info = 'step: {}/{}, train_acc: {:.4f}, train_robust_loss: {:.6f}, train_robust_acc: {:.4f}'.format(
                    i, len(train_loader), train_acc_meter.avg, train_robust_loss_meter.avg, train_robust_acc_meter.avg
                )
                print(log_info)


        if lr_scheduler is not None:
            lr_scheduler.step()
        else:
            if epoch == 50:
                manual_lr = args.lr_init * 0.1
                for param_group in opt.param_groups:
                    param_group['lr'] = manual_lr
            if epoch == 100:
                manual_lr = args.lr_init * 0.01
                for param_group in opt.param_groups:
                    param_group['lr'] = manual_lr

        train_time = time.time() - start
        _, test_std_acc, _, test_robust_acc, test_time = validate(model, test_loader, eval_epsilon, eval_alpha, eval_attack_iters, args.norm, ds=args.data)
        log_info = 'epoch: {}, train_time: {:.1f}s, test_time: {:.1f}s, lr: {:.6f}, alpha: {:.6f}, epsilon: {:.6f}, attack_iters: {:.1f}, train_acc: {:.4f}, train_robust_loss: {:.6f}, train_robust_acc: {:.4f}, test_acc: {:.4f}, test_{}_acc: {:.4f}'.format(
                epoch, train_time, test_time, opt.param_groups[0]['lr'], alpha, epsilon, attack_iters, train_acc_meter.avg, train_robust_loss_meter.avg, train_robust_acc_meter.avg,
                test_std_acc.item(), save_metric_name, test_robust_acc.item()
        )
        log_info += '\n'
        print(log_info)
        check_acc = test_robust_acc
        if check_acc > best_test_robust_acc:
            print(">>>>>>>>>>>>>>>>>>> update best model at {} epochs, test_{}_acc: {:.4f}".format(epoch, save_metric_name, check_acc.item()))
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_best.pth'))
            best_test_robust_acc = check_acc

        if epoch % args.ckpt_iters == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_{epoch}.pth'))       

            
def validate(model, test_loader, eval_epsilon, eval_alpha, eval_attack_iters, norm, func='ce', ds='cifar10'):
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    model.eval()
    start_time = time.time()
    for i, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.cuda()
        delta, _ = attack_pgd(model, data, label, eval_epsilon, eval_alpha, eval_attack_iters, 
                    1, norm, early_stop=False, func=func, ds=ds)
        delta = delta.detach()
        adv_data = normalize(torch.clamp(data + delta[:data.size(0)], min=lower_limit, max=upper_limit), ds=ds)
        
        with torch.no_grad():
            robust_output = model(adv_data)
            robust_loss = F.cross_entropy(robust_output, label)
            robust_accuracy = accuracy(robust_output.data, label)[0]

            output = model(normalize(data, ds=ds))
            std_loss = F.cross_entropy(output, label)
            std_accuracy = accuracy(output.data, label)[0]

        test_loss += std_loss * label.size(0)
        test_acc += std_accuracy * label.size(0)
        test_robust_loss += robust_loss * label.size(0)
        test_robust_acc += robust_accuracy * label.size(0)
        test_n += label.size(0)
        
    test_time = time.time() - start_time
    return test_loss / test_n, test_acc / test_n, test_robust_loss / test_n, test_robust_acc / test_n, test_time


if __name__ == '__main__':
    main()


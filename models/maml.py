from typing import Dict, List
from torch import optim
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .pure_layers import PureModule
from copy import deepcopy
from .learner import Learner


class MAMLModel(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config, target_model: PureModule):
        """
        :param args:
        """
        super(MAMLModel, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.config = config
        neru = [
            ('conv2d', [64, 1, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 2, 2, 1, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('flatten', []),
            ('linear', [args.n_way, 64])
        ]
        self.net = Learner(neru, 1, 64)
        # self.net = target_model(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    # def __init__(self, config: Dict, target_model: PureModule):
    #     self.config = config
    #     self.target_model = target_model
    #
    #     # TODO(maml): initialize parameters.
    #     # Hint: check .get_initial_params() method
    #     self.params = target_model.get_initial_params()

    # def train(self, mode: bool = True):
    #     self.target_model.train(mode)
    #
    # def eval(self):
    #     self.target_model.eval()
    #
    # def to(self, *args, **kwargs):
    #     self.params = nn.Parameter(self.params.to(*args, **kwargs))
    #
    #     return self
    #
    # def parameters(self) -> List[nn.Parameter]:
    #     return [self.params]

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    # cite from https://github.com/dragen1860/MAML-Pytorch
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        num_classes = self.config['training']['num_classes_per_task']
        batch_size = len(x_spt) // num_classes
        batch_qry = len(x_qry) // num_classes
        y_spt = y_spt.view(batch_size, num_classes)
        y_qry = y_qry.view(batch_qry, num_classes)
        c, h, w = x_spt.shape[1:]
        # torch.Size([32, 5, 1, 28, 28])
        # torch.Size([1, 5, 1, 64, 64])
        # torch.Size([4, 5, 1, 64, 64])
        # torch.Size([1, 5])
        # torch.Size([4, 5])
        x_spt = x_spt.view(batch_size, num_classes, c, h, w)
        x_qry = x_qry.view(batch_qry, num_classes, c, h, w)

        task_num, setsz, c_, h, w = x_spt.size()

        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # [i]
            # 1. run the i-th task and compute loss for k=0
            x_spt[i] = torch.autograd.Variable(x_spt[i], requires_grad=True)
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            # here is the code for this part
            # TODO(maml): compute the logits, outer-loop loss and outer-loop accuracy
            # logits = "TODO"
            # outer_loss = "TODO"
            # outer_acc = "TODO"

            # optim.zero_grad()
            # outer_loss.backward()
            # TODO(maml): you may like to add gradient clipping here
            # optim.step()
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs, loss_q

    # cite from https://github.com/dragen1860/MAML-Pytorch
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        # print(x_spt)
        x_spt.requires_grad = True
        net.parameters().requires_grad = True
        # 1. run   the i-th task and compute loss for k=0
        logits = net(x_spt)
        # print(logits)
        # print(logits.grad_fn)

        loss = F.cross_entropy(logits, y_spt)
        # print(loss)
        # print(loss.grad_fn)
        # loss = torch.autograd.Variable(loss, requires_grad=True)
        # net.parameters = torch.autograd.Variable(net.parameters, requires_grad=True)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)

            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs, loss_q

    # def __call__(self, x: Tensor) -> Tensor:
    #     # TODO: perform a forward pass
    #     # Hint: you should call target_model here with your params
    #     return "TODO"

from typing import Dict, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import MAMLModel, PureLeNet
from utils.data import FSLDataLoader
from .trainer import Trainer



class MAMLTrainer:

    def init_model(self, arg, config: Dict) -> Callable:
        return MAMLModel(arg, config, PureLeNet(config))


    def __init__(self, arg, config: Dict, source_dl: FSLDataLoader, target_dl: FSLDataLoader):
        self.arg = arg
        self.config = config
        self.rnd = np.random.RandomState(self.config['training']['random_seed'])

        self.model = self.init_model(arg, config).to(config['device'])
        self.optim = torch.optim.Adam(self.model.parameters(), **self.config['training']['optim_kwargs'])

        self.source_dataloader = source_dl
        self.target_dataloader = target_dl

    def train_on_episode(self, model, optim, ds_train, ds_test):
        losses = []
        accs = []

        model.train()

        # fast_w = model.params

        for it in range(self.config['model']['num_inner_steps']):
            x, y = self.sample_batch(ds_train)
            xx, yy = self.sample_batch(ds_test)
            x = x.to(self.config['device'])
            y = y.to(self.config['device'])
            xx = xx.to(self.config['device'])
            yy = yy.to(self.config['device'])

            # logits = model(x, y, xx, yy)  # [batch_size, num_classes_per_task]
            acc, loss = model(x, y, xx, yy)
            # loss = F.cross_entropy(logits, y)

            # TODO(maml): perform forward pass, compute logits, loss and accuracy
            # logits = "TODO"
            # loss = "TODO"
            # acc = "TODO"

            # TODO(maml): compute the gradient and update the fast weights
            # Hint: think hard about it. This is maybe the hardest part of the assignment
            # You will likely need to check open-source implementations to get the idea of how things work
            # grad = "TODO"
            # fast_w = "TODO"
            acc[-1] = max(acc)
            losses.append(loss.item())
            accs.append(acc[-1].item())

        # x = torch.stack([s[0] for s in ds_test]).to(self.config['device'])
        # y = torch.stack([s[1] for s in ds_test]).to(self.config['device'])



        return losses[-1], accs[-1]

    def train(self):
        episodes = tqdm(range(self.config['training']['num_train_episodes']))

        for ep in episodes:
            ds_train, ds_test = self.source_dataloader.sample_random_task()
            ep_train_loss, ep_train_acc = self.train_on_episode(self.model, self.optim, ds_train, ds_test)

            episodes.set_description(f'[Episode {ep: 03d}] Loss: {ep_train_loss: .3f}. Acc: {ep_train_acc: .03f}')

    def sample_batch(self, dataset):
        batch_size = min(self.config['training']['batch_size'], len(dataset))
        idx = self.rnd.choice(np.arange(len(dataset)), size=batch_size, replace=False)
        x = torch.stack([dataset[i][0] for i in idx])
        y = torch.stack([dataset[i][1] for i in idx])

        return x, y

    def evaluate(self):
        """
        For evaluation, we should
        """
        scores = {'train': [], 'test': []}
        episodes = tqdm(self.target_dataloader, desc='Evaluating')

        for ep, (ds_train, ds_test) in enumerate(episodes):
            train_scores, test_scores = self.fine_tune(ds_train, ds_test)
            scores['train'].append(train_scores)
            scores['test'].append(test_scores)

            episodes.set_description(
                f'[Test episode {ep: 03d}] Loss: {test_scores[0]: .3f}. Acc: {test_scores[1]: .03f}')

        for split in scores:
            split_scores = np.array(scores[split])
            print(f'[EVAL] Mean {split} loss: {split_scores[:, 0].mean(): .03f}.')
            print(f'[EVAL] Mean {split} acc: {split_scores[:, 1].mean(): .03f}.')


    def compute_scores(self, model, ds_train, ds_test) -> Tuple[np.float, np.float]:
        """
        Computes loss/acc for the dataloader
        """
        model.eval()
        # model.train()
        x, y = self.sample_batch(ds_train)
        xx, yy = self.sample_batch(ds_test)

        x = x.to(self.config['device'])
        y = y.to(self.config['device'])
        xx = xx.to(self.config['device'])
        yy = yy.to(self.config['device'])

        # x = torch.stack([s[0] for s in ds_train]).to(self.config['device'])
        # y = torch.stack([s[1] for s in ds_train]).to(self.config['device'])
        # xx = torch.stack([s[0] for s in ds_test]).to(self.config['device'])
        # yy = torch.stack([s[1] for s in ds_test]).to(self.config['device'])
        acc, loss = model.finetunning(x, y, xx, yy)
        loss=loss.item()
        acc = max(acc).item()
        # logits = model(x, y, xx, yy)
        # loss = F.cross_entropy(logits, y).item()
        # acc = (logits.argmax(dim=1) == y).float().mean().item()

        return loss, acc

    def fine_tune(self, ds_train, ds_test) -> Tuple[float, float]:
        # curr_model = init_model(self.arg, self.config).to(self.config['device'])
        # curr_model.params.data.copy_(self.model.params.data)

        curr_model = self.model
        # curr_optim = torch.optim.Adam(curr_model.parameters(), **self.config['model']['ft_optim_kwargs'])
        curr_optim = self.optim
        train_scores = self.train_on_episode(curr_model, curr_optim, ds_train, ds_test)
        test_scores = self.compute_scores(curr_model, ds_train, ds_test)

        return train_scores, test_scores

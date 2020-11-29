from typing import Tuple, Dict
import numpy as np
import torch

from utils.data import FSLDataLoader
from .trainer import Trainer
from tqdm import tqdm
import torch.nn.functional as F
from models import init_model


class ProtoNetTrainer(Trainer):
    def __init__(self, config: Dict, source_dl: FSLDataLoader, target_dl: FSLDataLoader):

        self.config = config
        self.rnd = np.random.RandomState(self.config['training']['random_seed'])
        self.model = init_model(config).to(config['device'])

        self.optim = torch.optim.Adam(self.model.parameters(), **self.config['training']['optim_kwargs'])
        self.source_dataloader = source_dl
        self.target_dataloader = target_dl




    def sample_batch(self, dataset):
        """
        In ProtoNet we require that the batch contains equal number of examples
        per each class. We do this so it is simpler to compute prototypes
        """
        k = self.config['training']['num_classes_per_task']  # k ways
        num_shots = len(dataset) // k
        batch_size = min(self.config['training']['batch_size'], num_shots)

        idx = [(c * num_shots + i) for c in range(k) for i in self.rnd.permutation(num_shots)[:batch_size]]
        x = torch.stack([dataset[i][0] for i in idx])
        y = torch.stack([dataset[i][1] for i in idx])

        return x, y

    def train_on_episode(self, model, optim, ds_train, ds_test=None):
        losses = []
        accs = []
        model.train()
        for it in range(self.config['training']['num_train_steps_per_episode']):
            x, y = self.sample_batch(ds_train)
            xx, yy = self.sample_batch(ds_test)

            x = x.to(self.config['device'])
            y = y.to(self.config['device'])
            xx = xx.to(self.config['device'])
            yy = yy.to(self.config['device'])

            logits = model(x, y, xx, yy)  # [batch_size, num_classes_per_task]     train set

            yy = yy.view(-1, self.config['training']['num_classes_per_task'])
            loss = F.cross_entropy(logits, yy)
            acc = (logits.argmax(dim=1) == yy).float().mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            accs.append(acc.item())

        return losses[-1], accs[-1]

    def fine_tune(self, ds_train, ds_test) -> Tuple[float, float]:
        # TODO(protonet): your code goes here
        # How does ProtoNet operate in the inference stage?

        curr_model = init_model(self.config).to(self.config['device'])
        curr_model.load_state_dict(self.model.state_dict())
        curr_optim = torch.optim.Adam(curr_model.parameters(), **self.config['training']['optim_kwargs'])

        train_scores = self.train_on_episode(curr_model, curr_optim, ds_train, ds_test)
        test_scores = self.compute_scores(curr_model, ds_train, ds_test)

        return train_scores, test_scores

    def compute_scores(self, model, ds_train, ds_test) -> Tuple[np.float, np.float]:
        """
        Computes loss/acc for the dataloader
        """
        model.eval()
        x = torch.stack([s[0] for s in ds_train]).to(self.config['device'])
        y = torch.stack([s[1] for s in ds_train]).to(self.config['device'])
        xx = torch.stack([s[0] for s in ds_test]).to(self.config['device'])
        yy = torch.stack([s[1] for s in ds_test]).to(self.config['device'])
        yy = yy.view(-1, self.config['training']['num_classes_per_task'])
        logits = model(x, y, xx, yy)
        loss = F.cross_entropy(logits, yy).item()
        acc = (logits.argmax(dim=1) == yy).float().mean().item()

        return loss, acc

    def train(self):
        episodes = tqdm(range(self.config['training']['num_train_episodes']))

        for ep in episodes:
            ds_train, ds_test = self.source_dataloader.sample_random_task()
            ep_train_loss, ep_train_acc = self.train_on_episode(self.model, self.optim, ds_train, ds_test)

            episodes.set_description(f'[Episode {ep: 03d}] Loss: {ep_train_loss: .3f}. Acc: {ep_train_acc: .03f}')

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

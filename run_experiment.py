#!/usr/bin/env python

import argparse

from trainers.trainer import Trainer
from trainers.protonet_trainer import ProtoNetTrainer
from trainers.maml_trainer import MAMLTrainer
from trainers.pointnet2_trainer import PointNetTrainer
from utils.config import construct_config
from utils.data import get_datasets, FSLDataLoader


# cite from https://github.com/dragen1860/MAML-Pytorch

def parse_args():
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('-m', '--method', default='unpretrained_baseline', type=str, help='Which method to run?')
    parser.add_argument('-k', '--shot', default=1, type=int, help='which k shots?')
    parser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    parser.add_argument('--imgc', type=int, help='imgc', default=1)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    return parser.parse_args()


def fix_random_seed(seed: int):
    import random
    import torch
    import numpy

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(method: str):
    # config = construct_config(method)
    # fix_random_seed(config['training']['random_seed'])
    # ds_train, ds_test = get_datasets(config)

    # source_dl = target_dl

    shots = [5, 1]
    parser = parse_args()
    for s in shots:
        config = construct_config(method)
        fix_random_seed(config['training']['random_seed'])
        # if method == 'maml':
        #     config['data']['target_img_size'] = 28
        ds_train, ds_test = get_datasets(config)
        config['training']['num_shots'] = int(s)
        print("now is %d shots, method is %s" % (config['training']['num_shots'], method))
        source_dl = FSLDataLoader(config, ds_train, 25)
        target_dl = FSLDataLoader(config, ds_test, 15)

        print(len(ds_test))
        if method in ['unpretrained_baseline', 'pretrained_baseline']:
            trainer = Trainer(config, source_dl, target_dl)
        elif method == 'protonet':
            trainer = ProtoNetTrainer(config, source_dl, target_dl)
        elif method == 'maml':
            trainer = MAMLTrainer(parser, config, source_dl, target_dl)
        elif method == 'pointnet':
            trainer = PointNetTrainer(config, source_dl, target_dl)
        else:
            raise NotImplementedError(f'Unknown method: {method}')

        trainer.train()
        trainer.evaluate()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args.method)

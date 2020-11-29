import torch
import numpy as np
from typing import Dict
import random

DATA_CONFIG = {
    'root_dir': './data',
    'target_img_size': 64,
    'batch_size': 24
}

METHODS_CONFIGS = {
    'unpretrained_baseline': {
        'name': 'unpretrained_baseline'
    },
    'pretrained_baseline': {
        'name': 'pretrained_baseline'
    },
    'protonet': {
        'name': 'protonet',
        # TODO(protonet): your protonet hyperparams
    },
    'maml': {
        'name': 'maml',
        'inner_loop_lr': 0.4,
        'num_inner_steps': 5,
        'ft_optim_kwargs': {'lr': 0.001},
        # TODO(maml): your maml hyperparams
    },
    'pointnet': {
        'name': 'pointnet'
    }
}

TRAINING_CONFIG = {
    'unpretrained_baseline': {
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 0,
        'optim_kwargs': {'lr': 0.001},
    },
    'pretrained_baseline': {
        # TODO(pretrained_baseline): your pretrained_baseline hyperparams
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 20,
        'optim_kwargs': {'lr': 0.001},
    },
    'protonet': {
        # TODO(protonet): your pretrained_baseline hyperparams
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 20,
        'optim_kwargs': {'lr': 0.001},
        # ... other ProtoNet hyperparameters?
    },
    'maml': {
        # TODO(maml): your pretrained_baseline hyperparams
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 20,
        'optim_kwargs': {'lr': 0.001},
    },
    'pointnet':{
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 20,
        'optim_kwargs': {'lr': 0.001},
    }
}

COMMON_TRAINING_CONFIG = {
    'random_seed': 42,
    'num_shots': 3,  # TODO: this is what you may vary to have different K values
    'num_classes_per_task': 5,
}


def construct_config(method_name: str, **overwrite_kwargs) -> Dict:
    default_config = {
        'data': DATA_CONFIG,
        'model': METHODS_CONFIGS[method_name],
        'training': {**COMMON_TRAINING_CONFIG, **TRAINING_CONFIG[method_name]},
        'device': ('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_point': 1024,
        'normal': True
    }
    final_config = {**default_config, **overwrite_kwargs}

    return final_config

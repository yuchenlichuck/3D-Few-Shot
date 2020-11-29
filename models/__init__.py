from typing import Dict, Callable

from .lenet import LeNet, PureLeNet
#from .protonet import ProtoNet
from .prointnet import ProtoNet
from .maml import MAMLModel
from .pointnet2_cls_ssg import PointNet

def init_model(config: Dict) -> Callable:
    if config['model']['name'] == 'unpretrained_baseline':
        return LeNet(config)
    elif config['model']['name'] == 'pretrained_baseline':
        return LeNet(config)
    elif config['model']['name'] == 'protonet':
        return ProtoNet(config)
    elif config['model']['name'] == 'pointnet':
        return PointNet(config)
    elif config['model']['name'] == 'maml':
        return MAMLModel(config, PureLeNet(config))
    else:
        raise NotImplementedError


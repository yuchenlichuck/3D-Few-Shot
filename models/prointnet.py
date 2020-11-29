from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from models import LeNet
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstraction


def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def prototypical_loss(prototypes, embeddings, targets, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(batch_size, num_examples)`.
    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                   - embeddings.unsqueeze(1)) ** 2, dim=-1)
    return F.cross_entropy(-squared_distances, targets, **kwargs)


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.
    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size*num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size*num_examples)`.
    num_classes : int
        Number of classes in the task.
    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size_num_classes, embedding_size = embeddings.size(0), embeddings.size(-1)
    batch_size = int(batch_size_num_classes / num_classes)

    embeddings = embeddings.view(batch_size, num_classes, embedding_size)
    targets = targets.view(batch_size, num_classes)
    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    def __init__(self, config, normal_channel=True):
        super().__init__()
        # def __init__(self, config: Dict, in_channels, out_channels, hidden_size=64):
        #     super(ProtoNet, self).__init__()
        self.config = config

        # in_channels = 1
        # out_channels = 4
        # hidden_size = 64

        # self.embedder = nn.Sequential(
        #     conv3x3(in_channels, hidden_size),
        #     conv3x3(hidden_size, hidden_size),
        #     conv3x3(hidden_size, hidden_size),
        #     conv3x3(hidden_size, out_channels)
        # )
        in_channel = 6 if normal_channel else 3

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 64)

        # self.embedder = LeNet(config)

        # TODO(protonet): your code here
        # Use the same embedder as in LeNet
        # self.embedder = "TODO"

    # def forward(self, inputs):
    #     embeddings = self.embedder(inputs.view(-1, *inputs.shape[2:]))
    #     return embeddings.view(*inputs.shape[:2], -1)

    def forward(self, xyz: Tensor, targets: Tensor, xx: Tensor, targets1: Tensor) -> Tensor:
        """
        Arguments:
            - x: images of shape [num_classes * batch_size, c, h, w]
            - targets: label of images [num_class*batch_size]
        """

        # Aggregating across batch-size
        num_classes = self.config['training']['num_classes_per_task']
        batch_size = len(xx) // num_classes

        # x = x.view(-1, *x.shape[2:])
        # x = x.permute(1, 2, 0)
        B, h, w = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        #   print(xyz.shape)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        embeddings = self.fc3(x)
        # x = F.log_softmax(x, -1)

        # embeddings = self.embedder(x)  # [num_classes * batch_size, dim]
        embeddings = embeddings.view(*x.shape[:1], -1)
        # TODO(protonet): compute prototypes given the embeddings
        # embedding_size = embeddings.size(-1)
        prototypes = get_prototypes(embeddings, targets, num_classes)

        # embeddings.new_zeros(batch_size, num_classes, embedding_size)
        # targets = targets.view(batch_size * num_classes)
        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity
        # logits = prototypical_loss(prototypes, embeddings, )
        B, h, w = xx.shape
        xx = xx.permute(0, 2, 1)
        #   print(xyz.shape)
        if self.normal_channel:
            norm = xx[:, 3:, :]
            xx = xx[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xx, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        embeddings1 = self.fc3(x)
        # embeddings = embeddings.view(batch_size, num_classes, -1)
        
        embeddings1 = embeddings1.view(batch_size, num_classes, -1)
        prototypes = torch.mean(prototypes, dim=0)
        prototypes = prototypes.unsqueeze(0)
        logits = -torch.sum((prototypes.unsqueeze(2) - embeddings1.unsqueeze(1)) ** 2, dim=-1)

        return logits

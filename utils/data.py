import random
from typing import List, Tuple, Dict, Callable, Iterable
from tqdm import tqdm

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from utils.ModelNetDataLoader import ModelNetDataLoader
from torchvision.datasets import Omniglot
# import data.data_utils as d_utils
from torchvision import transforms as transforms

from utils.config import construct_config


class FSLDataLoader:
    def __init__(self, config: Dict, dataset: Omniglot, num_classes: int):
        self.config = config
        self.rnd = np.random.RandomState(self.config['training']['random_seed'])
        images, labels = zip(*[(x, y) for (x, y) in tqdm(dataset, desc='Loading images in memory')])
        k = self.config['training']['num_classes_per_task']
        # way
        n = self.config['training']['num_shots']
        # n shots

        num_unique_classes = num_classes  # 40

        sorted_images = [[] for _ in range(num_unique_classes)]

        for img, c in zip(images, labels):
            for i in range(config['data']['batch_size']):

                if i >= int(c.shape[0]):
                    break
                if num_unique_classes == 25:
                    if int(c[i]) >= 25:
                        continue
                    if len(sorted_images[int(c[i])]) < 60:
                        sorted_images[int(c[i])].append(img[i].numpy())
                elif num_unique_classes == 15:
                    if int(c[i]) < 25:
                        continue
                    if len(sorted_images[int(c[i])-25]) < 20:
                        sorted_images[int(c[i])-25].append(img[i].numpy())
        print(len(sorted_images))



        self.images = torch.from_numpy(np.array(sorted_images))  # [num_unique_classes, num_imgs_per_class, c, h, w]

    '''
    def __init__(self, config: Dict, dataset: Dataset):


        self.config = config
        self.rnd = np.random.RandomState(self.config['training']['random_seed'])
        # images, labels = zip([(x,y) for (x,y) in dataset])
        # print("finish loading")
        images, labels = zip(*[(x, y) for (x, y) in tqdm(dataset, desc='Loading images in memory')])

        # num_unique_classes = len(set(labels))
        num_unique_classes = 40
        sorted_images = [[] for _ in range(num_unique_classes)]
        # for img, c in zip(images, labels):
        #     sorted_images[c].append(img.numpy())

        for img, c in zip(images, labels):
            for i in range(config['data']['batch_size']):
                # print(c.shape)
                # print(img.shape)

                if i >= int(c.shape[0]):
                    break
                # img1= np.numpy(img[i])
                # print(img1)
                
                sorted_images[int(c[i])].append(img[i])

        # for j in sorted_images[i]:
        # arr = np.array(sorted_images).astype(float)
        print(type(sorted_images))

        self.images = np.array(sorted_images)
        print(self.images.shape)
        # self.images = torch.from_numpy(arr)  # [num_unique_classes, num_imgs_per_class, c, h, w]
    '''

    def sample_random_task(self) -> Tuple[DataLoader, DataLoader]:
        # few shot part in each episode take one task as a
        k = self.config['training']['num_classes_per_task']
        task_classes = self.rnd.choice(range(len(self.images)), replace=False, size=k)
        # which one will be tests
        ds_train, ds_test = self.construct_datasets_for_classes(task_classes, shuffle_splits=True)

        return ds_train, ds_test

    def construct_datasets_for_classes(self, classes_to_use: List[int], shuffle_splits: bool = True) -> Tuple[
        Dataset, Dataset]:
        """
        It is guaranteed that examples are sorted by class order
        """
        k = self.config['training']['num_classes_per_task']
        n = self.config['training']['num_shots']
        num_imgs_per_class = len(self.images[0])
        h, w = self.images[0][0].shape

        task_imgs = self.images[classes_to_use]  # [num_classes_per_task, num_imgs_per_class, c, h, w]
        task_labels = torch.arange(k).unsqueeze(1).repeat(1,
                                                          num_imgs_per_class)  # [num_classes_per_task, num_imgs_per_class]

        if shuffle_splits:
            task_imgs = task_imgs[:, self.rnd.permutation(task_imgs.shape[1])]

        task_imgs_train = task_imgs[:, :n].reshape(-1, h, w)  # [num_classes_per_task * num_shots, c, h, w]
        task_imgs_test = task_imgs[:, n:].reshape(-1, h,
                                                  w)  # [num_classes_per_task * (num_imgs_per_class - num_shots), c, h, w]
        task_labels_train = task_labels[:, :n].reshape(-1)  # [num_classes_per_task * num_shots]
        task_labels_test = task_labels[:, n:].reshape(-1)  # [num_classes_per_task * (num_imgs_per_class - num_shots)]

        assert len(task_imgs_train) == len(
            task_labels_train), f"Wrong sizes: {len(task_imgs_train)} != {len(task_labels_train)}"
        assert len(task_imgs_test) == len(
            task_labels_test), f"Wrong sizes: {len(task_imgs_test)} != {len(task_labels_test)}"

        task_dataset_train = list(zip(task_imgs_train, task_labels_train))
        task_dataset_test = list(zip(task_imgs_test, task_labels_test))

        # task_dataset_train = Dataset(task_dataset_train)
        # task_dataset_test = Dataset(task_dataset_test)

        return task_dataset_train, task_dataset_test

    '''
    def construct_datasets_for_classes(self, classes_to_use: List[int], shuffle_splits: bool = False) -> Tuple[
        Dataset, Dataset]:
        """
        It is guaranteed that examples are sorted by class order
        """
        k = self.config['training']['num_classes_per_task']
        # way
        n = self.config['training']['num_shots']
        # n shots
        num_imgs_per_class = len(self.images[0])
        # How many images in the class, it suppose all the images has same sizes
        h, w = self.images[0][0].shape

        print(h, w)
        # [1024,6]

        task_imgs = self.images[classes_to_use]  # [num_classes_per_task, num_imgs_per_class, c, h, w]
        task_labels = torch.arange(k).unsqueeze(1).repeat(1,
                                                          num_imgs_per_class)  # [num_classes_per_task, num_imgs_per_class]

        # Don't random
        # if shuffle_splits:
        #     task_imgs = task_imgs[:, self.rnd.permutation(task_imgs.shape[1])]
        # evuladtion samples 15 per class
        task_imgs_train = []
        task_imgs_test = []
        task_labels_train = []
        task_labels_test = []

        for i in range(k):
            print(len(task_imgs[i]))
            # print(task_imgs[i])
            #
            # print(type(task_imgs[i]))
            # print(task_imgs[i].shape)
            task_imgs_train.append((task_imgs[i][:n]))
            task_imgs_test.append((task_imgs[i][n:n + 15]))
            task_labels_train.append((task_labels[i][:n]))
            task_labels_test.append((task_labels[i][n:n + 15]))
            # task_imgs_train.append((task_imgs[i][:n]).reshape(-1, h, w))
            # task_imgs_test.append((task_imgs[i][n:n + 15]).reshape(-1, h, w))
            # task_labels_train.append((task_labels[i][:n]).reshape(-1))
            # task_labels_test.append((task_labels[i][n:n + 15]).reshape(-1))
        task_imgs_train = torch.from_numpy(np.array(task_imgs_train))
        task_imgs_test = torch.from_numpy(np.array(task_imgs_test))
        task_labels_train = torch.from_numpy(np.array(task_labels_train))
        task_labels_test = torch.from_numpy(np.array(task_labels_test))
        # task_imgs_train = task_imgs[:, :n].reshape(-1, h, w)  # [num_classes_per_task * num_shots, c, h, w]
        # task_imgs_train = task_imgs[:, :n].reshape(-1, h, w)
        # task_imgs_test = task_imgs[:, n:n + 15].reshape(-1, h,
        #                                                 w)  # [num_classes_per_task * (num_imgs_per_class - num_shots), c, h, w]
        # task_labels_train = task_labels[:, :n].reshape(-1)  # [num_classes_per_task * num_shots]
        # task_labels_test = task_labels[:, n:n + 15].reshape(
        #    -1)  # [num_classes_per_task * (num_imgs_per_class - num_shots)]

        assert len(task_imgs_train) == len(
            task_labels_train), f"Wrong sizes: {len(task_imgs_train)} != {len(task_labels_train)}"
        assert len(task_imgs_test) == len(
            task_labels_test), f"Wrong sizes: {len(task_imgs_test)} != {len(task_labels_test)}"

        task_dataset_train = list(zip(task_imgs_train, task_labels_train))
        task_dataset_test = list(zip(task_imgs_test, task_labels_test))

        # task_dataset_train = Dataset(task_dataset_train)
        # task_dataset_test = Dataset(task_dataset_test)

        return task_dataset_train, task_dataset_test
    '''

    def __iter__(self) -> Iterable[Tuple[DataLoader, DataLoader]]:
        k = self.config['training']['num_classes_per_task']
        # k ways
        classes_order = self.rnd.permutation(len(self.images))  # [num_classes_total]
        # classes_order = classes_order[:-(len(classes_order) % k)]  # Drop last classes
        # zhengchu
        # print(len(classes_order))
        # print(classes_order)
        # print("iter")
        # print(len(self.images))
        # print(classes_order)
        assert len(classes_order) % k == 0

        classes_order = classes_order.reshape(len(classes_order) // k, k)  # [num_tasks, num_classes_per_task]

        return iter([self.construct_datasets_for_classes(cs) for cs in classes_order])

    def __len__(self) -> int:
        return len(self.images) // self.config['training']['num_classes_per_task']


def get_datasets(config: Dict) -> Tuple[Dataset, Dataset]:
    transform = get_transform(config)
    # ds_train = Omniglot(config['data']['root_dir'], background=True, download=True, transform=transform)
    # ds_test = Omniglot(config['data']['root_dir'], background=False, download=True, transform=transform)

    # Image: plt image
    # Omniglot: torchvision.datasets.omniglot.Omniglot

    # for i, j in ds_train:
    #     print(type(ds_train))
    #     print(ds_train)
    #
    #     print(i.size(), j.size())

    DATA_PATH = './data/modelnet40_normal_resampled/'

    ds_train = ModelNetDataLoader(root=DATA_PATH, npoint=config['num_point'], split='train',
                                  normal_channel=config['normal'])
    ds_test = ModelNetDataLoader(root=DATA_PATH, npoint=config['num_point'], split='test',
                                 normal_channel=config['normal'])

    ds_train = torch.utils.data.DataLoader(ds_train,
                                           batch_size=config['data']['batch_size'],
                                           shuffle=True,
                                           num_workers=40)
    ds_test = torch.utils.data.DataLoader(ds_test,
                                          batch_size=config['data']['batch_size'],
                                          shuffle=False,
                                          num_workers=40)
    # for i, j in ds_train:
    #     print(i)
    #     print(j)
    #     print(type(i))
    #     print(type(j))
    #     print(i.shape)
    #     print(j.shape)
    return ds_train, ds_test


# def get_transform(config: Dict) -> Callable:
#     return transforms.Compose([
#         transforms.Resize(config['data']['target_img_size']),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ])


def get_transform(config: Dict) -> Callable:
    import torchvision.transforms as transforms

    # trans = transforms.Compose(
    #     [
    #         d_utils.PointcloudToTensor(),
    #         d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
    #         d_utils.PointcloudScale(),
    #         d_utils.PointcloudTranslate(),
    #         d_utils.PointcloudJitter(),
    #     ]
    # )
    # return trans


if __name__ == '__main__':
    config = construct_config("protonet")

    ds_train, ds_test = get_datasets(config)
    # print(ds_test.shape)
    # print(ds_train.shape)

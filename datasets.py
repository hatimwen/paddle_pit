#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset related classes and methods for ViT training and validation
Cifar10, Cifar100 and ImageNet2012 are supported
"""

import os
import math
import re
import paddle
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddle.vision import transforms
from paddle.vision import datasets
from paddle.vision import image_load
from augment import auto_augment_policy_original
from augment import AutoAugment
from transforms import RandomHorizontalFlip
from random_erasing import RandomErasing


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx
class ImageNet2012Dataset(Dataset):
    def __init__(self, file_folder, mode="train", transform=None):
        class_to_idx = None
        root = os.path.join(file_folder, mode)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        print(f'----- Imagenet2012 image {mode} list len = {len(self.samples)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = image_load(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = paddle.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)


def get_train_transforms(config):
    """ Get training transforms

    For training, a RandomResizedCrop is applied, then normalization is applied with
    [0.5, 0.5, 0.5] mean and std. The input pixel values must be rescaled to [0, 1.]
    Outputs is converted to tensor

    Args:
        config: configs contains IMAGE_SIZE, see config.py for details
    Returns:
        transforms_train: training transforms
    """

    aug_op_list = []
    # STEP1: random crop and resize
    aug_op_list.append(
        transforms.RandomResizedCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE),
                                     scale=(0.05, 1.0), interpolation='bicubic'))
    # STEP2: auto_augment or color jitter
    if config.TRAIN.AUTO_AUGMENT:
        policy = auto_augment_policy_original()
        auto_augment = AutoAugment(policy)
        aug_op_list.append(auto_augment)
    else:
        jitter = (float(config.TRAIN.COLOR_JITTER), ) * 3
        aug_op_list.append(transforms.ColorJitter(*jitter))
    # STEP3: other ops
    aug_op_list.append(transforms.ToTensor())
    aug_op_list.append(transforms.Normalize(mean=config.DATA.IMAGENET_MEAN,
                                            std=config.DATA.IMAGENET_STD))
    # STEP4: random erasing
    if config.TRAIN.RANDOM_ERASE_PROB > 0.:
        random_erasing = RandomErasing(prob=config.TRAIN.RANDOM_ERASE_PROB,
                                       mode=config.TRAIN.RANDOM_ERASE_MODE,
                                       max_count=config.TRAIN.RANDOM_ERASE_COUNT,
                                       num_splits=config.TRAIN.RANDOM_ERASE_SPLIT)
        aug_op_list.append(random_erasing)
    # Final: compose transforms and return
    transforms_train = transforms.Compose(aug_op_list)
    return transforms_train


def get_val_transforms(config):
    """ Get training transforms

    For validation, image is first Resize then CenterCrop to image_size.
    Then normalization is applied with [0.5, 0.5, 0.5] mean and std.
    The input pixel values must be rescaled to [0, 1.]
    Outputs is converted to tensor

    Args:
        config: configs contains IMAGE_SIZE, see config.py for details
    Returns:
        transforms_train: training transforms
    """

    scale_size = int(math.floor(config.DATA.IMAGE_SIZE / config.DATA.CROP_PCT))
    transforms_val = transforms.Compose([
        transforms.Resize(scale_size, interpolation='bicubic'),
        transforms.CenterCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD),
    ])
    return transforms_val


def get_dataset(config, mode='train', transform=None):
    """ Get dataset from config and mode (train/val)
    Returns the related dataset object according to configs and mode(train/val)
    Args:
        config: configs contains dataset related settings. see config.py for details
    Returns:
        dataset: dataset object
    """

    assert mode in ['train', 'val']
    if config.DATA.DATASET == "cifar10":
        if mode == 'train':
            if transform is None:
                transform = get_train_transforms(config)
            dataset = datasets.Cifar10(mode=mode, transform=transform)
        else:
            mode = 'test'
            if transform is None:
                transform = get_val_transforms(config)
            dataset = datasets.Cifar10(mode=mode, transform=transform)
    elif config.DATA.DATASET == "cifar100":
        if mode == 'train':
            if transform is None:
                transform = get_train_transforms(config)
            dataset = datasets.Cifar100(mode=mode, transform=transform)
        else:
            mode = 'test'
            if transform is None:
                transform = get_val_transforms(config)
            dataset = datasets.Cifar100(mode=mode, transform=transform)
    elif config.DATA.DATASET == "imagenet2012" or config.DATA.DATASET == "imagenetlight":
        if mode == 'train':
            if transform is None:
                transform = get_train_transforms(config)
            dataset = ImageNet2012Dataset(config.DATA.DATA_PATH,
                                          mode=mode,
                                          transform=transform)
        else:
            if transform is None:
                transform = get_val_transforms(config)
            dataset = ImageNet2012Dataset(config.DATA.DATA_PATH,
                                          mode=mode,
                                          transform=transform)
    else:
        raise NotImplementedError(
            "[{config.DATA.DATASET}] Only cifar10, cifar100, imagenet2012 are supported now")
    return dataset


def get_dataloader(config, dataset, mode='train', multi_process=False, drop_last=False):
    """Get dataloader with config, dataset, mode as input, allows multiGPU settings.

        Multi-GPU loader is implements as distributedBatchSampler.

    Args:
        config: see config.py for details
        dataset: paddle.io.dataset object
        mode: train/val
        multi_process: if True, use DistributedBatchSampler to support multi-processing
    Returns:
        dataloader: paddle.io.DataLoader object.
    """

    if mode == 'train':
        batch_size = config.DATA.BATCH_SIZE
    else:
        batch_size = config.DATA.BATCH_SIZE_EVAL

    if multi_process is True:
        sampler = DistributedBatchSampler(dataset,
                                          batch_size=batch_size,
                                          shuffle=(mode == 'train'),
                                          drop_last=drop_last)
        dataloader = DataLoader(dataset,
                                batch_sampler=sampler,
                                num_workers=config.DATA.NUM_WORKERS)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=config.DATA.NUM_WORKERS,
                                shuffle=(mode == 'train'),
                                drop_last=drop_last)
    return dataloader

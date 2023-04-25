import os
import re
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lib.nets import ToyModel, ToyTembModel
from lib.resnet_temb import ResNetTemb
from lib.utils import AddGaussianNoise, AddUniformNoise, DataSubset


def save_checkpoint(state, is_best, save_path_prefix="", filename="_checkpoint.pt"):
    if not os.path.exists(os.path.dirname(save_path_prefix)):
        os.makedirs(os.path.dirname(save_path_prefix))
    torch.save(state, save_path_prefix + filename)
    if is_best:
        shutil.copyfile(
            save_path_prefix + filename, save_path_prefix + "_model_best.pt"
        )


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def get_dataset_info(name):
    info_dict = {
        "mnist": {"image_shape": (1, 28, 28), "n_class": 10},
        "fashionmnist": {"image_shape": (1, 28, 28), "n_class": 10},
        "svhn": {"image_shape": (3, 32, 32), "n_class": 10,},
        "cifar10": {"image_shape": (3, 32, 32), "n_class": 10,},
    }
    return info_dict[name]


def configure_net(args):

    if args.model_structure.casefold().startswith("ResNetTemb".casefold()):
        net_params = [
            int(_)
            for _ in re.findall(
                r"[-+]?\d*\.\d+|\d+", args.model_structure.replace("ResNetTemb", "")
            )
        ]

        net = ResNetTemb(
            n_class=args.n_class,
            ch=net_params[0],
            num_res_blocks=net_params[1],
            ch_mult=tuple(net_params[2:]),
            in_channels=args.image_shape[0],
            resolution=args.image_shape[1],
            act_func=args.model_act_func,
            dp_prob=args.dp_prob,
            use_spectral_norm=args.use_spectral_norm,
            conv_shortcut=args.use_convshortcut,
            resamp_with_conv=args.resamp_with_conv,
        )
    elif args.model_structure.casefold().startswith("ToyTembModel".casefold()):
        net_params = [
            int(_)
            for _ in re.findall(
                r"[-+]?\d*\.\d+|\d+", args.model_structure.replace("ToyTembModel", "")
            )
        ]
        net = ToyTembModel(args.n_class, act_func=args.model_act_func, hidden_num=net_params[0])
    elif args.model_structure == "ToyModel":
        net = ToyModel(n_output=args.n_class, act_func=args.model_act_func,)
    else:
        raise ValueError("model structure unspecified or not in the given list")

    return net


def initialize_net(net, net_init_method):

    if net_init_method.startswith("kaiming"):
        if (
            "[" in net_init_method
            and "out" in re.findall(r"\[(.*?)\]", net_init_method)[0]
        ):
            net = kaiming_init_net(net, net.act_func, mode="fan_out")
        else:
            net = kaiming_init_net(net, net.act_func)
    elif net_init_method.startswith("xavier"):
        init_std = (
            float(re.findall(r"[-+]?\d*\.\d+|\d+", net_init_method)[0])
            if "[" in net_init_method
            else 1
        )
        net = xavier_init_net(net, net.act_func, init_std)
    elif net_init_method == "default":
        pass
    else:
        raise NotImplementedError

    return net


def dataset_fn(data_dir, train, transform):
    if "fashionmnist" in data_dir:
        dataset = datasets.FashionMNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif "mnist" in data_dir:
        dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        )

    return dataset


def get_data_loader(args, data_dir, batch_size):
    transform_train_list = []
    transform_test_list = []
    if args.uniform_dequantization:
        transform_train_list += [AddUniformNoise()]
        transform_test_list = [AddUniformNoise()]

    if args.data_augmentation:
        transform_train_list += [
            transforms.RandomCrop(
                args.image_shape[1],
                padding=4,
                padding_mode="constant" if "mnist" in data_dir else "reflect",
            )
        ]

    if args.random_horizontal_flip is True:
        transform_train_list += [transforms.RandomHorizontalFlip()]

    if args.data_transform == "gaussian_center_and_scale":
        transform_train_list += [
            transforms.Normalize(
                [0.5] * args.image_shape[0], [0.5] * args.image_shape[0]
            ),
            AddGaussianNoise(args.gaussian_noise_std),
        ]
        transform_test_list += [
            transforms.Normalize(
                [0.5] * args.image_shape[0], [0.5] * args.image_shape[0]
            )
        ]
    elif args.data_transform == "center_and_scale":
        transform_train_list += [
            transforms.Normalize(
                [0.5] * args.image_shape[0], [0.5] * args.image_shape[0]
            )
        ]
        transform_test_list += [
            transforms.Normalize(
                [0.5] * args.image_shape[0], [0.5] * args.image_shape[0]
            )
        ]
    elif args.data_transform.casefold() == "identity":
        transform_train_list += []
        transform_test_list += []
    else:
        raise NotImplementedError(
            f"Data transformation {args.data_transform} is not implemented"
        )

    transform_train = torch.nn.Sequential(*transform_train_list)
    transform_test = torch.nn.Sequential(*transform_test_list)

    full_train = dataset_fn(data_dir, True, transforms.ToTensor())
    full_test = dataset_fn(data_dir, False, transforms.ToTensor())

    train_idx = list(range(len(full_train)))

    dataset_info = get_dataset_info(args.main_dir.replace("./", ""))

    train_idx = np.array(train_idx)

    train_labels = (
        full_train.labels[train_idx]
        if "svhn" in args.main_dir
        else np.array(full_train.targets)[train_idx]
    )

    rep_imgs_idx = []
    for i in range(dataset_info["n_class"]):
        rep_imgs_idx.append(np.where(train_labels == i)[0][0])
    rep_imgs = torch.stack(
        [full_train[train_idx[img_idx]][0] for img_idx in rep_imgs_idx]
    )
    rep_imgs = transform_train(rep_imgs)

    train_set = DataSubset(full_train, inds=train_idx, transform=transform_train,)

    test_set = DataSubset(
        full_test, inds=list(range(len(full_test))), transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False,
    )
    return (
        train_loader,
        test_loader,
        rep_imgs,
    )


def configure_optimizer(
    params, optimizer_type, lr, weight_decay=0, betas=[0.9, 0.999], sgd_momentum=0.9
):

    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            params, lr=lr, betas=betas, weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=sgd_momentum
        )
    else:
        raise NotImplementedError("Wrong optimizer type")

    return optimizer


def configure_scheduler(
    optimizer, scheduler_type, milestones, lr_decay_factor, n_epochs, n_warm_iters,
):
    if scheduler_type == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_decay_factor
        )
    elif scheduler_type == "Linear":
        linear_decay_slope = (1 - lr_decay_factor) / (n_epochs - milestones[0] + 1)
        lambda_lr_func = (
            lambda epoch: 1 - linear_decay_slope * (epoch - milestones[0] + 1)
            if epoch >= milestones[0]
            else 1
        )  # noqa
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_lr_func
        )
    elif scheduler_type == "LinearPlateau":
        linear_decay_slope = (1 - lr_decay_factor) / (milestones[1] - milestones[0] + 1)
        lambda_lr_func = (
            lambda epoch: 1 - linear_decay_slope * (epoch - milestones[0] + 1)
            if epoch >= milestones[0] and epoch < milestones[1]
            else (lr_decay_factor if epoch >= milestones[1] else 1)
        )  # noqa
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_lr_func
        )

    warmup_scheduler = WarmUpLR(optimizer, n_warm_iters)

    return scheduler, warmup_scheduler


def kaiming_init_net(net, act_func, mode="fan_in"):
    "act_funct: input an object of activation function"
    negative_slope = 0
    if isinstance(act_func, torch.nn.modules.activation.ReLU):
        nonlinearity = "relu"
    elif isinstance(act_func, torch.nn.modules.activation.LeakyReLU):
        negative_slope = act_func.negative_slope
        nonlinearity = "leaky_relu"
    else:
        nonlinearity = "linear"

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                m.weight, a=negative_slope, mode=mode, nonlinearity=nonlinearity
            )
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(
                m.weight, a=negative_slope, mode=mode, nonlinearity=nonlinearity
            )
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    return net


def xavier_init_net(net, act_func, std=1):
    "act_funct: input an object of activation function"
    if isinstance(act_func, torch.nn.modules.activation.ReLU):
        gain = nn.init.calculate_gain("relu")
    elif isinstance(act_func, torch.nn.modules.activation.LeakyReLU):
        gain = nn.init.calculate_gain("leaky_relu", act_func.negative_slope)
    else:
        gain = nn.init.calculate_gain("linear")

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight, gain)
            m.weight.data *= std
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain)
            m.weight.data *= std
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    return net

import os
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from .tiny_imagenet_pairs import TinyImagenetPairs
from .tiny_imagenet import TinyImagenet
from .cifar_100 import CIFAR100
from .cifar_100_pairs import CIFAR100Pairs
from .cifar_100_c import CIFAR100_C
from .mix_cifar_100 import MixCIFAR100
from .celebA import CelebA
from .celebA_pairs import CelebAPairs
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

def get_mean_std(args):
    if "cifar100" in args.dataset:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std

def hard_transform(args):
    mean, std = get_mean_std(args)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=60),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomErasing(p=0.3, scale=(0.01, 0.05), ratio=(0.8, 1.8)),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def simple_transform(args):
    # simple augmentation
    mean, std = get_mean_std(args)
    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomRotation(degrees=60),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def minimum_transform(args):
    mean, std = get_mean_std(args)
    transform_train = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    return transform_train

def mnist_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform

def collate_fn(batched_samples):
    assert len(batched_samples) > 0
    batch = {}
    batch["image"] = torch.stack([sample["image"] for sample in batched_samples], dim=0)
    batch["label"] = torch.stack([sample["label"] for sample in batched_samples], dim=0)
    if "original_image" in batched_samples[0]:
        batch["original_image"] = torch.stack([sample["original_image"] for sample in batched_samples], dim=0)
    if "aux" in batched_samples[0]:
        batch["aux"] = torch.stack([sample["aux"] for sample in batched_samples], dim=0)
    for k in batched_samples[0]:
        if k not in batch:
            batch[k] = [sample[k] for sample in batched_samples]
    return batch

class GroupCollateFnClass:
    group = None

    @staticmethod
    def group_filter_collate_fn(batched_samples):
        assert len(batched_samples) > 0
        batch = {}
        batch["image"] = []
        batch["label"] = []
        batch["aux"] = []
        if "original_image" in batched_samples[0]:
            batch["original_image"] = []
        for sample in batched_samples:
            if sample["aux"] == GroupCollateFnClass.group[0] and \
                sample["label"] == GroupCollateFnClass.group[1]:
                batch["image"].append(sample["image"])
                batch["label"].append(sample["label"])
                batch["aux"].append(sample["aux"])
                if "original_image" in batched_samples[0]:
                    batch["original_image"].append(sample["original_image"])
        for key in batch:
            if len(batch[key]) == 0:
                return {}
            batch[key] = torch.stack(batch[key], dim=0)
        return batch

def build_dataset(args, split="train", include_path=False):
    if split == "train" and "pairs" in args.dataset:
        if args.dataset == "tiny_imagenet_pairs":
            dataset = TinyImagenetPairs(transform=simple_transform(args),
                                        subset=args.data_subset)
            dataset.collate_fn = collate_fn
        elif args.dataset == "cifar100_pairs":
            dataset = CIFAR100Pairs(transform=simple_transform(args),
                                    subset=args.data_subset,
                                    unbalanced=args.unbalanced)
            dataset.collate_fn = CIFAR100Pairs.collate_fn
        elif args.dataset == "celebA_pairs":
            dataset = CelebAPairs(transform=simple_transform(args),
                                  subset=args.data_subset,
                                  unbalanced=args.unbalanced)
            dataset.collate_fn = CelebAPairs.collate_fn
        else:
            raise NotImplementedError(f"{args.dataset} not supported")
        return dataset
    else:
        if args.dataset == "tiny_imagenet" or \
            args.dataset == "tiny_imagenet_pairs":
            dataset = TinyImagenet(train_transform=simple_transform(args),
                                val_transform=minimum_transform(args),
                                split=split,
                                subset=args.data_subset if split == 'train' else 1.0,
                                group=args.data_group,
                                include_path=include_path)
            dataset.collate_fn = collate_fn
        elif args.dataset == "cifar100" or \
            args.dataset == "cifar100_pairs":
            dataset = CIFAR100(train_transform=simple_transform(args),
                            val_transform=minimum_transform(args),
                            minimum_transform=minimum_transform(args),
                            split=split,
                            subset=args.data_subset if split == 'train' else 1.0,
                            group=args.data_group,
                            unbalanced=args.unbalanced,
                            include_path=include_path)
            dataset.collate_fn = collate_fn
        elif args.dataset == "cifar100_c":
            dataset = CIFAR100_C(transform=minimum_transform(args),
                                corruption=args.corruption,
                                severity=args.severity)
            dataset.collate_fn = collate_fn
        elif args.dataset == "celebA" or \
            args.dataset == "celebA_pairs":
            dataset = CelebA(train_transform=minimum_transform(args),
                             val_transform=minimum_transform(args),
                             minimum_transform=minimum_transform(args),
                             split=split,
                             subset=args.data_subset if split == 'train' else 1.0,
                             group=args.data_group,
                             unbalanced=args.unbalanced,
                             include_path=include_path)
            dataset.collate_fn = collate_fn
        elif args.dataset == "mnist":
            dataset = MNIST(root=os.path.join(config("DATASET_ROOT"), config("MNIST_ROOT")),
                            train=split=="train",
                            transform=mnist_transform(),
                            download=True)
        else:
            raise NotImplementedError(f"{args.dataset} not supported")
        return dataset
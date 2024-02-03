from torchvision.transforms import transforms
from .tiny_imagenet_pairs import TinyImagenetPairs
from .tiny_imagenet import TinyImagenet
from .cifar_100 import CIFAR100
from .cifar_100_pairs import CIFAR100Pairs
from .cifar_100_c import CIFAR100_C
from .mix_cifar_100 import MixCIFAR100

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

def build_dataset(args, split="train", include_path=False):
    if split == "train" and "pairs" in args.dataset:
        if args.dataset == "tiny_imagenet_pairs":
            return TinyImagenetPairs(transform=simple_transform(args),
                                    subset=args.data_subset)
        elif args.dataset == "cifar100_pairs":
            return CIFAR100Pairs(transform=simple_transform(args),
                                subset=args.data_subset)
    else:
        if args.dataset == "tiny_imagenet" or \
            args.dataset == "tiny_imagenet_pairs":
            return TinyImagenet(train_transform=simple_transform(args),
                                val_transform=minimum_transform(args),
                                split=split,
                                subset=args.data_subset if split == 'train' else 1.0,
                                group=args.data_group,
                                include_path=include_path)
        elif args.dataset == "cifar100" or \
            args.dataset == "cifar100_pairs":
            return CIFAR100(train_transform=simple_transform(args),
                            val_transform=minimum_transform(args),
                            minimum_transform=minimum_transform(args),
                            split=split,
                            subset=args.data_subset if split == 'train' else 1.0,
                            group=args.data_group,
                            include_path=include_path)
        elif args.dataset == "cifar100_c":
            return CIFAR100_C(corruption=args.corruption,
                              severity=args.severity)
    raise NotImplementedError(f"{args.dataset} not supported")
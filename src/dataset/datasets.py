from torchvision.transforms import transforms
from .tiny_imagenet_pairs import TinyImagenetPairs
from .tiny_imagenet import TinyImagenet

def hard_transform(args):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=60),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.8, 1.8)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def simple_transform(args):
    # simple augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform

def minimum_transform():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_train

def build_tiny_imagenet_dataset(split, args, include_path=False, include_origin=False):
    return TinyImagenet(train_transform=hard_transform(args),
                        val_transform=simple_transform(args),
                        split=split,
                        subset=args.data_subset if split == 'train' else 1.0,
                        group=args.data_group,
                        include_path=include_path,
                        include_origin=include_origin)

def build_tiny_imagenet_pairs_dataset(args):
    return TinyImagenetPairs(transform=simple_transform(args),
                             subset=args.data_subset)
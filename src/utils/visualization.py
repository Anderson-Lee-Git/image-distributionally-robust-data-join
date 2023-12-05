import torch
import matplotlib.pyplot as plt

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

CIFAR100_DEFAULT_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_DEFAULT_STD = [0.2675, 0.2565, 0.2761]

def visualize_image(img: torch.Tensor, path: str, rev_std: bool = True):
    if not isinstance(img, torch.Tensor):
        raise NotImplementedError(f"Please pass in tensor objects, dtype ({type(img)}) is not supported")
    # if passed in batched images
    if len(img.shape) > 3:
        img = img[0]
    mean = CIFAR100_DEFAULT_MEAN
    std = CIFAR100_DEFAULT_STD
    img = img * torch.tensor(std).cuda().view(3, 1, 1) + torch.tensor(mean).cuda().view(3, 1, 1)
    img.clamp_(min=0, max=1)
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    fig.savefig(path)
    plt.clf()
    plt.close()
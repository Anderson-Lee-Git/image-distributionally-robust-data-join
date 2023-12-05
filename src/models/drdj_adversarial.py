import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Softmax
import wandb

from .resnet import ResNet50, ResNet34
import models
from utils.init_utils import initialize_weights
from utils.visualization import visualize_image

supported_backbone = [
    "ResNet34",
    "ResNet50"
]

class DRDJAdversarial(nn.Module):
    def __init__(self, r_a: float, r_p: float,
                 adv_lr: float,
                 backbone: str, num_classes: int,
                 args) -> None:
        super(DRDJAdversarial, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.model = self._build_backbone()
        self.args = args
        # constants
        self.r_a, self.r_p = r_a, r_p
        self.adv_lr = adv_lr
        # cross entropy loss
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.softmax = Softmax(dim=1)
        # store the most recent x and x_other for forward loss
        self.x = None
        self.x_other = None
        self.embed = None
        self.embed_other = None
        # initialize weights
        self._initialize_weight()
        self._load_pretrained_backbone()
        # take over classifier layer
        self.fc = self.model.fc
        self.model.fc = nn.Identity()

    def freeze_params(self):
        for param in self.model.parameters():
            param.requires_grad = False
        assert self.fc.weight.requires_grad == True
    
    def random_start(self, center: torch.Tensor, epsilon: float):
        """
        Select a random point that is on the perimeter of a L2-ball. 
        This point is where the L2-norm-ball constraint is tight. 

        Arguments:
            center: origin of the L2-ball
            epsilon: radius of the L2-ball
        Returns:
            None

            The input 'center' is modified in place. 
        """
        B = center.size()[0]
        direction = torch.rand(center.size()) * 2 - 1
        direction = direction.cuda()
        length = torch.linalg.vector_norm(direction, dim=(1, 2, 3))  # (B,)
        center.data.add_(direction / length.view(B, 1, 1, 1) * epsilon)
        # center.data.clamp_(0, 1)
    
    def perturb(self, x, x_adv, anchor, desirable_distance, steps=10):
        B, C, H, W = x.shape
        assert anchor.requires_grad == False
        for i in range(steps):
            if x_adv.grad is not None:
                x_adv.grad.data.zero_()
            self.model.zero_grad()
            self.fc.zero_grad()
            output = self.fc(self.model(x_adv))
            if torch.isnan(output).any():
                print(torch.max(x_adv), torch.min(x_adv))
                is_nan = torch.stack([torch.isnan(p).any() for p in self.model.parameters()]).any().item()
                print(f"model weight exist nan: {is_nan}")
                print(f"detect embed to be nan")
                exit()
            # loss is defined as to maximize
            # the difference between attacked embed
            # and the anchor embed
            loss = torch.mean(torch.linalg.vector_norm(output - anchor, dim=1))
            if torch.isnan(loss).any():
                print(f"detect loss to be nan")
            retain_graph = i == steps-1
            loss.backward(retain_graph=retain_graph)
            # update using plus because we aim to maximize
            if torch.isnan(x_adv.grad).any():
                print(f"detect x_adv grad to be nan")
            x_adv.data.add_(self.adv_lr * x_adv.grad) 
            diff = x.detach() - x_adv.detach()
            dist = torch.linalg.vector_norm(diff, dim=(1, 2, 3)).cuda() # (B,)
            if torch.isnan(dist).any():
                print(f"detect distance to be nan")
            
            # replace zero value with one
            if desirable_distance != 0:
                dist[dist == 0] = desirable_distance
                x_adv.data = x - diff * torch.minimum(desirable_distance / dist,
                                                      torch.ones(B,).cuda()).view(B, 1, 1, 1)
            else:
                x_adv.data = x
            
            if torch.isnan(x_adv).any():
                print(f"x_adv becomes nan after update")
                print(f"max diff = {torch.max(diff)}")
                print(f"max dist = {torch.max(dist)}")
                print(f"any zero is dist? {torch.count_nonzero(dist) < len(dist)}")
        return x_adv

    def adversarial_samples(self, x, x_other):
        output = self.fc(self.model(x))
        output_other = self.fc(self.model(x_other))
        # attack x
        desirable_distance = self.r_p
        if torch.isnan(x).any():
            print(f"detect nan for x")
        x_adv = x.clone().detach().cuda()
        self.random_start(x_adv, self.r_p)
        if torch.isnan(x_adv).any():
            print(f"detect nan for x_adv")
        x_adv.requires_grad_(True)
        x_adv = self.perturb(x=x,
                            x_adv=x_adv, 
                            anchor=output_other.detach().cuda(),
                            desirable_distance=desirable_distance).detach().cuda()
        assert x_adv.requires_grad == False
        self.model.zero_grad()
        self.fc.zero_grad()
        # attack x_other
        desirable_distance = self.r_a
        x_other_adv = x_other.clone().detach().cuda()
        self.random_start(x_other_adv, self.r_a)
        x_other_adv.requires_grad_(True)
        x_other_adv = self.perturb(x=x_other,
                                    x_adv=x_other_adv,
                                    anchor=output.detach().cuda(),
                                    desirable_distance=desirable_distance).detach().cuda()
        assert x_other_adv.requires_grad == False
        if torch.isnan(x_other_adv).any():
            print(f"detech nan for x_other_adv")
        self.x_adv = x_adv
        self.x_other_adv = x_other_adv
        self.model.zero_grad()
        self.fc.zero_grad()
        # visualize a random image and adversarial example
        idx = torch.randint(low=0, high=len(x), size=(1,)).item()
        visualize_image(x[idx], os.path.join(self.args.output_dir, f"examples/image_p_{self.r_p}.png"))
        visualize_image(x_adv[idx], os.path.join(self.args.output_dir, f"examples/adversarial_image_p_{self.r_p}.png"))
        visualize_image(x_other[idx], os.path.join(self.args.output_dir, f"examples/image_a_{self.r_a}.png"))
        visualize_image(x_other_adv[idx], os.path.join(self.args.output_dir, f"examples/adversarial_image_a_{self.r_a}.png"))
    
    def forward_adv(self, x, x_other, labels, attack_prob=0.1):
        p = torch.rand(size=(1,)).item()
        if p < attack_prob:
            # generate adversarial samples
            self.adversarial_samples(x, x_other)
            x_adv = self.x_adv
            x_other_adv = self.x_other_adv
            # output
            worst_p = self.fc(self.model(x_adv))
            normal_p = self.fc(self.model(x))
            worst_a = self.fc(self.model(x_other_adv))
            normal_a = self.fc(self.model(x_other))
            output = torch.mean(torch.stack([worst_p, normal_p, worst_a, normal_a], dim=0), dim=0)
            # output = normal_p
            loss = torch.mean(self.loss_fn(output, labels))
        else:
            output = self.forward_eval(x)
            loss = torch.mean(self.loss_fn(output, labels))
        return output, loss
    
    def forward_eval(self, x):
        embed = self.model(x)
        return self.fc(embed)
    
    def _initialize_weight(self):
        for m in self.model.modules():
            if isinstance(m, models.resnet.Bottleneck) or \
                isinstance(m, models.resnet.ResNet):
                continue
            else:
                initialize_weights({}, m)

    def _build_backbone(self):
        if self.backbone not in supported_backbone:
            raise NotImplementedError(f"{self.backbone} not supported")
        if self.backbone == "ResNet50":
            return ResNet50(num_classes=self.num_classes, channels=3)
        elif self.backbone == "ResNet34":
            return ResNet34(num_classes=self.num_classes, channels=3)
    
    def _load_pretrained_backbone(self):
        # load pretrained model
        if self.args.pretrained_path:
            print(f"Load backbone from {self.args.pretrained_path}")
            state_dict = torch.load(self.args.pretrained_path, "cuda")
            self.model.load_state_dict(state_dict["model"])
        
    
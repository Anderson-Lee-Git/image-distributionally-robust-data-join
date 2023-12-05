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

class DRDJVanilla(nn.Module):
    def __init__(self, r_a: float, r_p: float,
                 kappa_a: float, kappa_p: float,
                 n_a: int, n_p: int,
                 lambda_1: float, lambda_2: float, lambda_3: float,
                 backbone: str, num_classes: int,
                 args) -> None:
        super(DRDJVanilla, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.model = self._build_backbone()
        self.args = args
        # parameters
        self.alpha_a = nn.Parameter(torch.tensor([1.0]))
        self.alpha_p = nn.Parameter(torch.tensor([1.0]))
        # constants
        self.r_a, self.r_p, self.kappa_a, self.kappa_p, self.n_a, self.n_p \
            = r_a, r_p, kappa_a, kappa_p, n_a, n_p
        self.lambda_1, self.lambda_2, self.lambda_3 = lambda_1, lambda_2, lambda_3
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
    
    def forward_eval(self, x):
        embed = self.model(x)
        return self.fc(embed)

    def forward_loss(self, x, x_other, labels, include_max_term=False, include_norm=False):
        embed = self.model(x)
        embed_other = self.model(x_other)
        output = self.fc(embed)
        B, L = output.shape
        cross_entropy_term = self.loss_fn(output, labels)
        # sum of misclassified logits
        if include_max_term:
            max_term = torch.maximum(output[torch.arange(len(labels)).cuda(), labels] - self.alpha_p * self.kappa_p,
                                    torch.zeros(B).cuda())
        else:
            max_term = torch.zeros(1).cuda()
        if include_norm:
            norm_term = self.alpha_a * torch.sigmoid(torch.linalg.vector_norm(embed - embed_other, dim=1))
        else:
            norm_term = torch.zeros(1).cuda()
        try:
            wandb.log({
                "max_term": torch.mean(max_term),
                "norm_term": torch.mean(norm_term)
            })
        except:
            pass
        summation_term = cross_entropy_term + max_term - norm_term
        # TODO: revisit the product of n_A and n_P
        loss = (self.alpha_a * self.r_a + self.alpha_p * self.r_p) + \
            (torch.mean(summation_term)) + \
            self._penalty().cuda()
        return output, loss
    
    def _penalty(self):
        penalty_1 = self.lambda_1 * (torch.linalg.matrix_norm(self.fc.weight) - \
                                     (self.alpha_a + self.alpha_p))
        penalty_2 = 0  # TODO: theta_2
        penalty_3 = self.lambda_3 * (self.alpha_a - self.alpha_p)
        return torch.sum(torch.Tensor([penalty_1, penalty_2, penalty_3]))
    
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
        
    
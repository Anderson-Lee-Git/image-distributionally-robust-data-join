import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Softmax
import wandb

import models
from models.build_baseline import build_resnet
from utils.init_utils import initialize_weights
from utils.visualization import visualize_image

supported_backbone = [
    "resnet34",
    "resnet50"
]

class DRDJVanilla(nn.Module):
    def __init__(self, r_a: float, r_p: float,
                 kappa_a: float, kappa_p: float,
                 n_a: int, n_p: int,
                 lambda_1: float, lambda_2: float, lambda_3: float,
                 backbone: str, num_classes: int,
                 embed_dim: int,
                 aux_embed_dim: int,
                 objective: str,
                 dist_weight: bool,
                 args) -> None:
        super(DRDJVanilla, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.aux_embed_dim = aux_embed_dim
        self.dist_weight = dist_weight
        self.args = args
        self.encoder = self._build_backbone()
        self.objective = objective
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
        # TODO: Not compatiable with imagenet pre-trained model
        self._load_pretrained_backbone()
        # take over classifier layer
        self.fc = nn.Linear(self.embed_dim + self.aux_embed_dim, self.num_classes)
        self.encoder.fc = nn.Identity()
        # initialize weights
        self._initialize_weight()

    def freeze_params(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        assert self.fc.weight.requires_grad == True
    
    def forward_eval(self, x, aux):
        B = x.shape[0]
        if aux is None:
            aux_embed = torch.zeros(B, self.aux_embed_dim).to(torch.float32).cuda()
        else:
            aux_embed = aux.view(B, self.aux_embed_dim).to(torch.float32).cuda()
        embed = self.encoder(x)
        embed = torch.concat([embed, aux_embed], dim=1)
        return self.fc(embed)
    
    def forward_loss(self, x, x_other, aux, labels, dist_weight=None, include_max_term=False, include_norm=False):
        B = x.shape[0]
        aux_embed = aux.view(B, self.aux_embed_dim).to(torch.float32)
        embed = self.encoder(x)
        embed_other = self.encoder(x_other)
        if dist_weight is None or not self.dist_weight:
            dist_weight = torch.ones(B).cuda()
        if self.objective == "P":
            output = self.fc(torch.concat([embed, aux_embed], dim=1)).cuda()
        else:
            output = self.fc(torch.concat([embed_other, aux_embed], dim=1)).cuda()
        B, L = output.shape
        cross_entropy_term = self.loss_fn(output, labels)
        # sum of misclassified logits
        if include_max_term:
            max_term = torch.maximum(output[torch.arange(len(labels)).cuda(), labels] - self.alpha_p * self.kappa_p,
                                    torch.zeros(B).cuda())
        else:
            max_term = torch.zeros(1).cuda()
        if include_norm:
            norm_term = self.alpha_a * torch.linalg.vector_norm(embed.detach() - embed_other.detach(), dim=1)
        else:
            norm_term = torch.zeros(1).cuda()
        # penalty term
        penalty_term = self._penalty().cuda()
        summation_term = cross_entropy_term * dist_weight + max_term - norm_term
        # TODO: revisit the product of n_A and n_P
        loss = (self.alpha_a * self.r_a + self.alpha_p * self.r_p) + \
            (torch.mean(summation_term)) + \
            penalty_term
        try:
            wandb.log({
                f"max_term ({self.objective})": torch.mean(max_term),
                f"norm_term ({self.objective})": torch.mean(norm_term),
                f"penalty_term ({self.objective})": penalty_term.item(),
                f"cross_entropy_term ({self.objective})": torch.mean(cross_entropy_term).item()
            }, commit=False)
        except:
            pass
        return output, loss
    
    def _penalty(self):
        penalty_1 = self.lambda_1 * torch.relu(torch.linalg.matrix_norm(self.fc.weight[:, :self.embed_dim]) - \
                                     (self.alpha_a + self.alpha_p))
        penalty_2 = self.lambda_2 * torch.relu(torch.linalg.matrix_norm(self.fc.weight[:, -self.aux_embed_dim:]) - \
                                     (self.kappa_a * self.alpha_a))
        if self.objective == "P":
            penalty_3 = self.lambda_3 * torch.relu(self.alpha_p - self.alpha_a)
        else:
            penalty_3 = self.lambda_3 * torch.relu(self.alpha_a - self.alpha_p)
        
        try:
            wandb.log({
                f"penalty_term_1 ({self.objective})": penalty_1.item(),
                f"penalty_term_2 ({self.objective})": penalty_2.item(),
                f"penalty_term_3 ({self.objective})": penalty_3.item()
            }, commit=False)
        except:
            pass
        return penalty_1 + penalty_2 + penalty_3
    
    def _initialize_weight(self):
        # for m in self.modules():
        #     if isinstance(m, models.resnet.Bottleneck) or \
        #         isinstance(m, models.resnet.ResNet):
        #         continue
        #     else:
        #         initialize_weights({}, m)
        initialize_weights({}, self.fc)

    def _build_backbone(self):
        if self.backbone not in supported_backbone:
            raise NotImplementedError(f"{self.backbone} not supported")
        else:
            model = build_resnet(num_classes=self.num_classes, pretrained=True,
                                 args=self.args)
            return model
            
    def _load_pretrained_backbone(self):
        # load pretrained model
        if self.args.pretrained_path:
            print(f"Load backbone from {self.args.pretrained_path}")
            state_dict = torch.load(self.args.pretrained_path, "cuda")
            self.encoder.load_state_dict(state_dict["model"])

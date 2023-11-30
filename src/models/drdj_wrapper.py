import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Softmax

from .resnet import ResNet50, ResNet34

import models
from utils.init_utils import initialize_weights

supported_backbone = [
    "ResNet34",
    "ResNet50"
]

class DRDJWrapper(nn.Module):
    def __init__(self, r_a: float, r_p: float,
                 kappa_a: float, kappa_p: float,
                 n_a: int, n_p: int,
                 lambda_1: float, lambda_2: float, lambda_3: float,
                 backbone: str, num_classes: int) -> None:
        super(DRDJWrapper, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.model = self._build_backbone()
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
        self.output = None
        self.output_other = None
        # initialize weights
        self._initialize_weight()
    
    def forward_loss(self, output, labels):
        """
        Must be called after forward to update self.x and self.x_other
        """
        B, L = output.shape
        cross_entropy_term = self.loss_fn(output, labels)
        max_term = torch.maximum(output[torch.arange(len(labels)).cuda(), labels] - self.alpha_p * self.kappa_p,
                                 torch.zeros(B).cuda())
        norm_term = 0.2 * torch.sigmoid(self.alpha_a * torch.linalg.vector_norm(self.output - self.output_other, dim=1))
        print(f"crossentropy term: {torch.mean(cross_entropy_term)}")
        print(f"max term: {torch.mean(max_term)}")
        print(f"norm term: {torch.mean(norm_term)}")
        summation_term = cross_entropy_term + max_term - norm_term
        # print(f"summation form shape: {summation_term.shape}")
        # TODO: revisit the product of n_A and n_P
        loss = (self.alpha_a * self.r_a + self.alpha_p * self.r_p) + \
            (torch.mean(summation_term)) + \
            self._penalty().cuda()
        return loss
    
    def forward(self, x, x_other):
        output = self.model(x)
        self.output = output
        self.output_other = self.model(x_other)
        return output
    
    def forward_eval(self, x):
        output = self.model(x)
        return output
    
    def _penalty(self):
        penalty_1 = self.lambda_1 * (torch.linalg.matrix_norm(self.model.fc.weight) - \
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
    
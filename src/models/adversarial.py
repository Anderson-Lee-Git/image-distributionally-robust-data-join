import torch
import torch.nn as nn
from .build_baseline import build_resnet

class Adversarial(nn.Module):
    def __init__(self, embed_dim, num_target_classes, num_attr_classes, beta, args):
        super(Adversarial, self).__init__()
        self.encoder = build_resnet(num_classes=num_target_classes, pretrained=True, args=args)
        self.encoder.fc = nn.Identity()
        self.target_fc = nn.Linear(embed_dim, num_target_classes)
        self.attr_fc = nn.Linear(embed_dim, num_attr_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.beta = beta
        
    def forward(self, x):
        x = self.encoder(x)
        y_hat = self.target_fc(x)
        return y_hat
    
    def forward_target_loss(self, x, y):
        x = self.encoder(x)
        y_hat = self.target_fc(x)
        loss = self.loss_fn(y_hat,  y)
        return y_hat, loss
    
    def forward_attr_loss(self, x, a):
        x = self.encoder(x)
        a_hat = self.attr_fc(x)
        # gradient ascent
        loss = -self.beta * self.loss_fn(a_hat, a)
        return a_hat, loss
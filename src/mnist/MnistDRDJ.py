import wandb
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Softmax

class MnistDRDJ(nn.Module):
    def __init__(self, r_a: float, r_p: float,
                 kappa_a: float, kappa_p: float,
                 n_a: int, n_p: int,
                 lambda_1: float, lambda_2: float, lambda_3: float,
                 num_classes: int,
                 embed_dim: int,
                 aux_embed_dim: int,
                 objective: str,
                 args) -> None:
        super(MnistDRDJ, self).__init__()
        # TODO
        self.x_encoder = nn.Sequential(
            nn.Linear(32, 64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 128),
            # nn.LeakyReLU(),
            # nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, embed_dim)
        )
        self.aux_encoder = nn.Sequential(
            nn.Linear(32, 64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 128),
            # nn.LeakyReLU(),
            # nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, aux_embed_dim)
        )
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.aux_embed_dim = aux_embed_dim
        self.args = args
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
        # take over classifier layer
        self.fc = nn.Linear(self.embed_dim + self.aux_embed_dim, self.num_classes)
        # initialize weights
        self._initialize_weight()

    def forward_eval(self, x, aux):
        B = x.shape[0]
        if aux is None:
            aux_embed = torch.zeros(B, self.aux_embed_dim).to(torch.float32).cuda()
        else:
            aux_embed = self.aux_encoder(aux)
        embed = self.x_encoder(x)
        embed = torch.concat([embed, aux_embed], dim=1)
        return self.fc(embed)
    
    def forward_loss(self, x, x_other, aux, labels, dist_weight=None, include_max_term=False, include_norm=False):
        B = x.shape[0]
        aux_embed = self.aux_encoder(aux)
        embed = self.x_encoder(x)
        embed_other = self.x_encoder(x_other)
        if dist_weight is None:
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
        for m in self.x_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        for m in self.aux_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

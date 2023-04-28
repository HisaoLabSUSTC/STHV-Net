import torch
import torch.nn as nn


class MLSEloss(nn.Module):
    def __init__(self):
        super(MLSEloss, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self, output, target):
        return self.MSE(torch.log(output), torch.log(target))


class PCTLoss(nn.Module):
    def __init__(self):
        super(PCTLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(
            torch.divide(
                torch.abs(
                    torch.sub(target, output)
                ),
                target
            )
        )


class NormPCTLoss(nn.Module):
    def __init__(self, norm_weight):
        super(NormPCTLoss, self).__init__()
        self.norm_weight = norm_weight

    def set_weight(self, norm_weight):
        self.norm_weight = norm_weight

    def forward(self, output, target):
        return torch.mean(
            torch.divide(
                torch.abs(
                    torch.sub(target + float('1e-20'), output)
                ),
                (target + float('1e-20'))
            )
        ) - torch.log(torch.var(output)) * self.norm_weight

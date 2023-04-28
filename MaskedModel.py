import torch

from setTransformer.modules import *


class MaskedModel(nn.Module):
    def __init__(self, input_dim, depth, device, hidden_dim=128):
        super(MaskedModel, self).__init__()
        self.device = device
        self.projection_layer = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Parameter(torch.Tensor(1, 1, input_dim))
        nn.init.xavier_uniform_(self.query)
        self.encoder_layers = nn.ModuleList([
            MSAB(hidden_dim, hidden_dim, 4, ln=True) for _ in range(depth)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.FloatTensor, mask_tensor: torch.FloatTensor = None):
        B, N, D = x.shape
        # cat_tensor = -torch.ones(B, 1, D)
        # cat_tensor = cat_tensor.to(self.device)
        # input_x = torch.cat((cat_tensor, x), dim=1)
        repeated_seed = (self.query.repeat(B, 1, 1))
        input_x = torch.cat((repeated_seed, x), dim=1)

        input_x = input_x.to(self.device)
        output = self.projection_layer(input_x)
        if mask_tensor is None:
            mask_tensor = torch.ones((B, N + 1, N + 1))
            mask_tensor[:, :, 0] = 0
            mask_tensor[:, 0, 0] = 1
            mask_tensor = mask_tensor.to(self.device)
        for layer in self.encoder_layers:
            output = layer(output, mask_tensor)
        output = output[:, 0, :]
        output = self.decoder(output)
        return torch.squeeze(output)


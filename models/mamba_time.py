import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear_proj_in = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, groups=d_model, padding=d_conv//2)
        self.silu = nn.SiLU()
        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.linear_proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, channels, length) -> transpose to (batch, length, channels) for norm
        x = x.transpose(1, 2)  # (batch, length, channels)
        x = self.norm(x)
        x = self.linear_proj_in(x)
        # transpose back for conv1d: (batch, channels, length)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.silu(x)
        # SSM expects (batch, length, d_model)
        x = x.transpose(1, 2)
        x = self.ssm(x)
        x = self.linear_proj_out(x)
        # output: (batch, length, channels) -> transpose to (batch, channels, length)
        x = x.transpose(1, 2)
        return x

class MambaEncoder(nn.Module):
    """Mamba Encoder for time series representation
    Attributes
    ----------
    in_channels: int
        Number of input channels
    mid_channels: int
        Hidden dimension for Mamba blocks
    num_layers: int
        Number of Mamba blocks
    """

    def __init__(self, in_channels: int, mid_channels: int = 64, num_layers: int = 3, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'mid_channels': mid_channels,
            'num_layers': num_layers
        }

        # Initial projection to mid_channels
        self.input_proj = nn.Conv1d(in_channels, mid_channels, kernel_size=1)

        self.layers = nn.Sequential(*[
            MambaBlock(d_model=mid_channels, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        # x: (batch, channels, length)
        x = self.input_proj(x)
        z = self.layers(x)
        z = z.mean(dim=-1)  # Global average pooling
        return z

def mamba_ts(**kwargs):
    return {'backbone': MambaEncoder(**kwargs), 'dim': kwargs['mid_channels']}
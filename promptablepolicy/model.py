import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(
            self,
            state_dim: int,
            prompt_dim: int,
            ac_dim: int,
            hid_dim: int = 128,
    ) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim + prompt_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, ac_dim)
    
    def forward(
            self,
            state: torch.Tensor,
            prompt: torch.Tensor,
    ) -> torch.Tensor:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(prompt.shape) == 1:
            prompt = prompt.unsqueeze(0)
        x = torch.cat([state, prompt], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch.nn as nn
import torch.nn.functional as F

class TextPrototypeProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x= self.proj(x)
        x = F.normalize(x, dim=2)
        # return self.proj(x)
        return x

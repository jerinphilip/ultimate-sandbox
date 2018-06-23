import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResnetMinus(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet_minus = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = resnet.fc.in_features
        self.model = resnet_minus
        self.freeze()
        
        
    def freeze(self):
        for child in self.model.children():    
            for param in child.parameters():
                 param.requires_grad = False
    
    def output_size(self):
        return self.feature_size
    
    def forward(self, x):
        return self.model(x)
    

class TimeDistributedDense(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
      
    def forward(self, x):
        # y-size is 8x1x512
        # we need to take linear mapping for 512 sized vectors.
        B, T, H = x.size()
        x = x.view(-1, H)
        x = self.linear(x)
        x = x.view(B, T, -1)
        return x
    

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, tgt, teacher_forcing):
        return self.decode(self.encode(src), tgt, teacher_forcing)
        
    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, context, seed, teacher_forcing):
        return self.decoder(context, seed, teacher_forcing)
    

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x    
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.project = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)

        
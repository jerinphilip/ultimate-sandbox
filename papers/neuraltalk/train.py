from torchvision import models
from torch import nn, optim
from models import ResnetMinus, TimeDistributedDense
import torch


class Trainer:
    def __init__(self, **kwargs):
        self.cnn = ResnetMinus()
        feature_size = self.cnn.output_size()
        self.params = kwargs
        self.vocab = kwargs['vocab']

        # Define LSTM
        self.emb = nn.Embedding(len(self.vocab), kwargs['embedding_size'])
        self.fc_in = nn.Linear(feature_size, kwargs['hidden_size'])
        self.rnn = nn.LSTM(input_size=kwargs['embedding_size'], 
                hidden_size=kwargs['hidden_size'],
                num_layers=kwargs['num_layers'],
                bidirectional=False,
                dropout=kwargs['dropout'], batch_first=True)
        
        # Linear, Softmax and decoding c
        self.fc_out = TimeDistributedDense(kwargs['hidden_size'], len(self.vocab))

        self.criterion = nn.CrossEntropyLoss()
        self.optparams = list( self.rnn.parameters()) +\
                    list(self.fc_out.parameters())
        self.optimizer = optim.SGD(self.optparams,                    
                    lr=1e-2)
        self.device = torch.device("cpu")
        self.clip = 5
        
        self.best_model, self.best_loss = None, float("inf")
    
    def _batch(self, img, z):
         # Encoded representation
        if self.train:
            self.optimizer.zero_grad()
        h0 = self.cnn(img)
        B, H, _, _ = h0.size()
        h0 = h0.reshape(B, -1)
        h0 = self.fc_in(h0)
        h0 = h0.repeat(self.params["num_layers"], 1, 1)
        h, c = h0, torch.zeros_like(h0)
        #h, c = [torch.zeros(3, 1, 512).to(self.device) for i in range(2)]
        B, T = z.size()
        #h = torch.zeros(3, 1, 512)
        loss = 0
        
        # Without teacher forcing.
        for t in range(1, T):
            x = z[:, t-1]
            x = self.emb(x)
            x = x.reshape(B, 1, -1)
            y, (h, c) = self.rnn(x, (h, c))
            y = self.fc_out(y)
            loss += self.criterion(y.view(B, -1), z[:,t])

        if self.train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optparams, self.clip)
            self.optimizer.step()
            
        return loss.item()

    def mode(self, train):
        self.train = train
        if(not train):
            self.rnn.eval()
            self.fc_in.eval()                       
            self.fc_out.eval()
            self.emb.eval()
        else:
            self.rnn.train()
            self.fc_in.train()
            self.fc_out.train()
            self.emb.train()
            

    def train_batch(self, img, z):
        self.mode(train=True)
        return self._batch(img, z)

        #print("Loss", loss.item())
        
    def export(self):
        params = {
            "rnn": self.rnn.state_dict(),
            "fc_in": self.fc_in.state_dict(),
            "fc_out": self.fc_out.state_dict(),
            "emb": self.emb.state_dict(),
            "opt": self.optimizer.state_dict()
        }
        return params
    
    def load(self, payload):
        self.rnn.load_state_dict(payload["rnn"])
        self.fc_in.load_state_dict(payload["fc_in"])
        self.fc_out.load_state_dict(payload["fc_out"])
        self.emb.load_state_dict(payload["emb"])
        self.optimizer.load_state_dict(payload["opt"])
        pass
    
    def valid_batch(self, img, z):
        self.mode(train=False)
        loss = self._batch(img, z)
        if loss < self.best_loss:
            self.best_model = self.export()
            self.best_loss = loss
        return loss
        
    def cuda(self, device):
        self.device = device
        self.cnn.cuda()
        self.rnn.cuda()
        self.emb.cuda()
        self.fc_in.cuda()
        self.fc_out.cuda()

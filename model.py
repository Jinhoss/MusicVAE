import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def inverse_sigmoid(epoch, k=20):
    return k / (k + np.exp(epoch/k))

class Encoder(nn.Module):
        
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1, bidirectional=True):          
        super(Encoder, self).__init__()
        
        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1
            
        self.hidden_size = hidden_size
        self.num_hidden = num_directions * num_layers
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional)
        
        self.mu = nn.Linear(self.num_hidden * self.hidden_size, latent_dim)
        self.std = nn.Linear(self.num_hidden * self.hidden_size, latent_dim)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        
    def forward(self, x): 
        x, (h, c) = self.lstm(x)
        h = h.transpose(0, 1).reshape(-1, self.num_hidden * self.hidden_size)
        
        mu = self.norm(self.mu(h))
        std = nn.Softplus()(self.std(h))
        
        # reparam
        z = self.reparameterize(mu, std)
        
        return z, mu, std
    
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)

        return mu + (eps * std)
        
class Conductor(nn.Module):
    
    def __init__(self, input_size, hidden_size, device, num_layers=2, bar=4):
        super(Conductor, self).__init__()

        num_directions = 1

        self.bar = bar
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = num_directions * num_layers
        
        self.norm = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.conductor = nn.LSTM(batch_first=True,
                                 input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=False)
        
    def init_hidden(self, batch_size, z):
        h0 = z.repeat(self.num_hidden, 1, 1)
        c0 = z.repeat(self.num_hidden, 1, 1)

        return h0, c0
    
    def forward(self, z):            
        batch_size = z.shape[0]
        h, c = self.init_hidden(batch_size, z)
        z = z.unsqueeze(1)
        
        # initialize
        feat = torch.zeros(batch_size, self.bar, self.hidden_size, device=self.device)
        
        # conductor
        z_input = z
        for i in range(self.bar):
            z_input, (h, c) = self.conductor(z_input, (h, c))
            feat[:, i, :] = z_input.squeeze()
            z_input = z
            
        feat = self.linear(feat)
            
        return feat
    

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=False):            
        super(Decoder, self).__init__()
        
        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden = num_directions * num_layers
        
        self.logits= nn.Linear(hidden_size, output_size)
        self.decoder = nn.LSTM(batch_first=True,
                               input_size=input_size+output_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional)
        
    def forward(self, x, h, c, z, temp=1):
        x = torch.cat((x, z.unsqueeze(1)), 2)
        
        x, (h, c) = self.decoder(x, (h, c))
        logits = self.logits(x) / temp
        prob = nn.Softmax(dim=2)(logits)
        out = torch.argmax(prob, 2)
                
        return out, prob, h, c

class MusicVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args.enc_input_size, args.enc_hidden_size, args.enc_latent_dim)
        self.conductor = Conductor(args.enc_latent_dim, args.con_hidden_size, args.device)
        self.decoder = Decoder(args.con_hidden_size, args.dec_hidden_size, args.dec_output_size)
        self.bar_units = args.bar_units
        self.num_hidden = self.decoder.num_hidden
        self.hidden_size = self.decoder.hidden_size
        self.output_size = self.decoder.output_size
        self.device = args.device

    def forward(self, x):
        seq_len = x.shape[0]
        latent_z, mu, std = self.encoder(x)
        feat = self.conductor(latent_z)
        x_train_inputs = torch.zeros((x.shape[0], 1, x.shape[2]), device=self.device)
        x_train_label = torch.zeros(x.shape[:-1], device=self.device)
        x_train_prob = torch.zeros(x.shape, device=self.device)
        for j in range(seq_len):
            bar_idx = j // self.bar_units
            bar_change_idx = j % self.bar_units
            z = feat[:, bar_idx, :]
            if bar_change_idx == 0:
                h = z.repeat(self.num_hidden, 1, int(self.hidden_size/z.shape[1]))
                c = z.repeat(self.num_hidden, 1, int(self.hidden_size/z.shape[1]))
            
            label, prob, h, c = self.decoder(x_train_inputs.to(self.device), h, c, z)
            
            x_train_label[:, j] = label.squeeze()
            x_train_prob[:, j, :] = prob.squeeze()
            x_train_inputs = x[:, j, :].unsqueeze(1)

        return x_train_prob, mu, std, x_train_label

    def generate(self, bar_units=16, seq_len=64, n=1):
        z = torch.empty((n, 512)).normal_(mean=0,std=1)
        feat = self.conductor(z.cuda())
        batch_size = n
        hidden_size = self.decoder.hidden_size
        output_size = self.decoder.output_size
        num_hidden = self.decoder.num_hidden
        
        inputs = torch.zeros((batch_size, 1, output_size), device=self.device)
        outputs = torch.zeros((batch_size, seq_len, output_size), device=self.device)
        for j in range(seq_len):
            bar_idx = j // bar_units
            bar_change_idx = j % bar_units
            
            z = feat[:, bar_idx, :]
            
            if bar_change_idx == 0:
                h = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                c = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                
            label, prob, h, c = self.decoder(inputs, h, c, z, temp=n)
            outputs[:, j, :] = prob.squeeze()

            inputs = F.one_hot(label, num_classes=output_size)
            
        return outputs.cpu().detach().numpy()
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(LSTM, self).__init__()
        
        self.hsize = hidden_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.i_x = nn.Linear(input_size, hidden_size)
        self.i_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.f_x = nn.Linear(input_size, hidden_size)
        self.f_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.o_x = nn.Linear(input_size, hidden_size)
        self.o_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.z_x = nn.Linear(input_size, hidden_size)
        self.z_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.proj = nn.Linear(hidden_size, out_size, bias=False)
        
    def forward(self, x, hc):
        hidden, cell = hc
        
        i = self.sigmoid(self.i_x(x) + self.i_h(hidden))
        f = self.sigmoid(self.f_x(x) + self.f_h(hidden))
        o = self.sigmoid(self.o_x(x) + self.o_h(hidden))
        z = self.tanh(self.z_x(x) + self.z_h(hidden))
        
        cell = f * cell + i * z
        hidden = self.tanh(cell) * o
        # project ouput
        output = self.proj(hidden)
        return output, (hidden, cell)
    
    def init_hidden(self):
        return torch.zeros(1, self.hsize, requires_grad=False), torch.zeros(1, self.hsize, requires_grad=False)
    
    
    def train(self, xs, ys, epochs, seqlen, enc_size, lr=0.001, print_interval=1000):
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        for e in range(epochs):
            for i, (x, y) in enumerate(zip(xs, ys)):
                # x = (seqlen, num_categories)
                # y = (num_categories,)
                hidden = self.init_hidden()
                out = torch.zeros((seqlen, enc_size))

                # forward 
                for ix, element in enumerate(x):
                    out[ix], hidden = self.forward(element, hidden)

                loss = F.cross_entropy(out, y)

                # backward
                loss.backward()
                optim.step()
                
                # print
                if i % print_interval == 0:
                    print(f"E{e}, I{i} -- Loss: {loss.item()}")
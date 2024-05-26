"""
Copying task is a benchmark for sequence models outlined in Arjovsky, Shah, Bengio (2016).
A network is trained to recall a segment of tokens from the past after some number of "blank" tokens.

For this example, an LSTM is used.
"""

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# generates datasets for static copying   
def get_data(mem_len, n_blanks, n_samples, batch_size, train_ratio=0.8):
    # 0 = blank category
    # [1, 8] = relevant categories
    # 9 = delimiter
    seq_len = mem_len*2 + 1 + n_blanks
    xdata = torch.zeros(n_samples, seq_len)
    rands = torch.randint(1, 9, size=(n_samples, mem_len)) # generate segments to memorize
    y_data = rands
    
    # populate x
    xdata[:, 0: mem_len] = rands 
    xdata[:, -mem_len-1] = 9     # set delimeter
    x_enc = F.one_hot(xdata.long(), 10)
    
    # batches
    n_batches = n_samples // batch_size
    x_enc = torch.reshape(x_enc, [n_batches, batch_size, seq_len, 10]).float()
    y_data = torch.reshape(y_data, [n_batches, batch_size, mem_len]).long()
    
    # divide into train and test
    train_batches = int(n_batches * train_ratio)
    x_train = x_enc[0 : train_batches]
    x_test = x_enc[train_batches:]
    y_train = y_data[0 : train_batches]
    y_test = y_data[train_batches:]

    return x_train, y_train, x_test, y_test


class LSTM(nn.Module):
    def __init__(self, idim, hdim, odim, depth):
        super(LSTM, self).__init__()
        
        self.recurrent = nn.LSTM(idim, hdim, num_layers=depth, batch_first=True)
        self.proj = nn.Linear(hdim, odim, bias=False)
        
    def forward(self, x):
        out, _ = self.recurrent(x)
        out = self.proj(out)
        return out  


def train(model, optim, xs, ys, epochs, mem_len, print_interval=1000, losses=[]):
    for e in range(epochs):
        for i, (x, y) in enumerate(zip(xs, ys)):
            out = model(x)
            selected = out.transpose(1,2)[:, :, -mem_len:] # transpose for crossentropy format and select the segment of interest
            loss = F.cross_entropy(selected, y)
            losses.append(loss.item())

            if i % print_interval == 0:
                print(f"Epoch: {e}|i: {i} \tLoss: {loss.item()}")

            loss.backward()
            optim.step()
            optim.zero_grad()   

def test(model, xs, ys, mem_len):
    mean_acc = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        out = model(x)
        preds = torch.argmax(out[:, -mem_len:], dim=-1)
        correct = preds == y
        acc = correct.float().mean()
        mean_acc += acc
        
    return mean_acc.item() / (i+1)

def infer(model, x):
    out = model(x)
    print(f"X: {x.argmax(dim=-1)} \nPred: {out.argmax(dim=-1).squeeze()}")


# test!
def main():
    torch.manual_seed(1337)
    # define dataset
    # 10 memorized tokens at the beginning, 10 blanks
    # 50k examples split into batches of 25
    mem = 10
    x_tr, y_tr, x_test, y_test = get_data(mem, 20, 50_000, 25)
    losses = []

    # define model
    # 10 input dims, 128 hidden dims, 10 output dims, layers=1
    model = LSTM(10, 128, 10, 1)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)

    train(model, optim, x_tr, y_tr, 20, mem, print_interval=2000, losses=losses)
    print(f"\nAccuracy: {test(model, x_test, y_test, mem)}")

    plt.plot(losses)
    plt.plot(0)
    plt.show()
    

if __name__== "__main__": 
    main() 
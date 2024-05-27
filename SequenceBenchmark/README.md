This is an implementation of the copying task as outlined in Arjovsky et al, 2016 (https://arxiv.org/pdf/1511.06464).

10 categories are used, of which 8 are relevant, 1 is the delimiter, and 1 is "blank". 
To evaluate a network's capability of recalling earlier segments of a sequence, it is given K tokens of relevant categories followed by T blanks followed by a delimiter. 
It is then asked to reproduce the K relevant tokens from the beginning in the exact order.

`copytask.py` uses an LSTM as an example of a model capable of completing this task.
Training and testing functions are included. With a single LSTM layer and a hidden state of size 128, it can achieve ~97% accuracy given K=10 and T=20.

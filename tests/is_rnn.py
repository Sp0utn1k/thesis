from utils.networks import LSTMNetwork, FCNetwork
import torch.nn as nn
from utils.functionnal import is_rnn


net1 = FCNetwork(1,1)
net2 = LSTMNetwork(1,1)
print(is_rnn(net1),is_rnn(net2))
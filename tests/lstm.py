from utils.networks import LSTMNetwork

lstm = LSTMNetwork(5,8)
print(lstm)
print(1+lstm.rnn.bidirectional,lstm.rnn.num_layers,lstm.rnn.hidden_size)
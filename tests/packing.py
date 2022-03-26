import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from pytorch_forecasting.utils import unpack_sequence
from utils.functionnal import squash_packed

input_size = 3
lstm_size = 1
seq_lengths = [2,1,30,20]


inputs = [seq_len*torch.ones(seq_len,input_size) for seq_len in seq_lengths]
print([obs.shape for obs in inputs])

pre_net = torch.nn.Linear(input_size,lstm_size)

seqs = pack_sequence(inputs,enforce_sorted=False)

seqs = squash_packed(seqs,pre_net)

lstm = torch.nn.LSTM(lstm_size,lstm_size)
seqs,hidden = lstm(seqs)
h,c = hidden

output, indices = unpack_sequence(seqs)
print(output.shape)
output, indices = pad_packed_sequence(seqs,batch_first=True)
print(output.shape)
indices -= 1
output = [output[i,0:indices[i],:].detach() for i in range(output.shape[0])]
# print(*output, sep='\n')
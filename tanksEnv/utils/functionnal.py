import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

def is_rnn(net):
	if isinstance(net,nn.RNNBase):
		return True
	else:
		for child in net.children():
			if is_rnn(child):
				return True
	return False

def build_network(input_size,output_size,layers):
	net = []
	layer_size = input_size

	for next_layer_size in layers:
		net += [nn.Linear(layer_size,next_layer_size),nn.ReLU()]
		layer_size = next_layer_size

	net += [nn.Linear(layer_size,output_size)]
	return net

def squash_packed(x, fn):
    return PackedSequence(fn(x.data), x.batch_sizes, 
                 x.sorted_indices, x.unsorted_indices)
import torch
import torch.nn as nn
from utils.functionnal import build_network
from utils.networks import RNNetwork, DecoderNN


if __name__ == '__main__':

	input_size = 2
	pre_layers = [5,5]
	post_layers = [5,10]
	output_size = 5

	decoder_hidden_layers = [10]
	sequence_lengths = [1,2,3]
	test_data = torch.ones(len(sequence_lengths),input_size)

	data = []
	for sequence_length in sequence_lengths:
		data.append(torch.rand(sequence_length,input_size))

	encoder = RNNetwork(input_size,output_size,rnn_size=10,rnn_type='lstm',pre_processing_layers=pre_layers,post_processing_layers=post_layers)
	output,_ = encoder(data,encoder.init_hidden('cpu'))
	# print(encoder)
	print(output.shape)

	decoder = DecoderNN(output_size+input_size,hidden_layers=decoder_hidden_layers)
	print('\n',decoder(output,test_data))
	print(decoder.predict(output,test_data))
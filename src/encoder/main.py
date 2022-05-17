from utils import *
import cProfile
import torch
from tanksEnv.utils.networks import RNNetwork, DecoderNN


def main():	
	visibility = 12
	max_id = 10
	max_length = 3
	batch_size = 5
	output_size = 5
	rnn_size = 32

	data = generate_tanks_data(visibility,max_id)

	input_size = len(data[0])
	seqs, positives = seq(data,max_length)

	encoder = RNNetwork(input_size,output_size,rnn_size=rnn_size)
	decoder = DecoderNN(output_size+input_size)
	test_data = torch.tensor(data,dtype=torch.float32)

	print(test_data.shape)
	labels = torch.zeros((test_data.size(0)),dtype=torch.float32)

	print(positives)

if __name__ == '__main__':

	cProfile.run('main()')
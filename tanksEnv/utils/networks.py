import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from tanksEnv.utils.functionnal import build_network,squash_packed
import time

class RNNetwork(nn.Module):
	def __init__(self,input_size,output_size,num_layers=1,rnn_size=0,
				pre_processing_layers=[],post_processing_layers=[],rnn_type='GRU',**kwargs):
		super().__init__()

		self.rnn_type = rnn_type.lower()
		pre_processing_network = build_network(input_size,rnn_size,pre_processing_layers)
		self.pre_net = nn.Sequential(*pre_processing_network,nn.ReLU())


		post_processing_network = build_network(rnn_size,output_size,post_processing_layers)
		self.post_net = nn.Sequential(nn.ReLU(),*post_processing_network)

		if rnn_type.lower() == 'lstm':
			self.rnn = nn.LSTM(rnn_size,rnn_size,num_layers=num_layers)
		elif rnn_type.lower() == 'gru':
			self.rnn = nn.GRU(rnn_size,rnn_size,num_layers=num_layers)

	def forward(self,inputs,hidden=None,benchmark=False):

		if not isinstance(inputs,list):
			if inputs.ndim == 2:
				inputs = inputs.unsqueeze(1)
			output = self.pre_net(inputs)
			self.rnn.flatten_parameters()
			if hidden == None:
				output, hidden = self.rnn(output)
			else:
				output, hidden = self.rnn(output,hidden)
			output = self.post_net(output[-1,:,:])
		else:
			if benchmark:
				timer = [0,0]
				timer[0] = time.time()
				timer[1] = time.time()

			seqs = pack_sequence(inputs,enforce_sorted=False)
			if benchmark:
				t = 1000*round(time.time()-timer[1],6)
				print(f'Pack: {t}')
				timer[1] = time.time()

			seqs = squash_packed(seqs,self.pre_net)
			if benchmark:
				t = 1000*round(time.time()-timer[1],6)
				print(f'Pre net: {t}')
				timer[1] = time.time()

			if hidden == None:
				output, hidden = self.rnn(seqs)
			else:
				output, hidden = self.rnn(seqs,hidden)
			if benchmark:
				t = 1000*round(time.time()-timer[1],6)
				print(f'Call rnn: {t}')
				timer[1] = time.time()

			output, indices = pad_packed_sequence(output,batch_first=True)
			indices -= 1
			if benchmark:
				t = 1000*round(time.time()-timer[1],6)
				print(f'Unpack: {t}')
				timer[1] = time.time()

			# output = torch.stack([output[i,indices[i],:] for i in range(output.shape[0])])
			indices = indices.to(output.device).reshape(-1,1,1).expand(-1,-1,output.shape[2])
			output = output.gather(1,indices).squeeze(1)

			if benchmark:
				t = 1000*round(time.time()-timer[1],6)
				print(f'Stack: {t}')
				timer[1] = time.time()

			output = self.post_net(output)
			if benchmark:
				t = 1000*round(time.time()-timer[1],6)
				print(f'Post net: {t}')
				timer[1] = time.time()

			if benchmark:
				t_total = 1000*round(time.time()-timer[0],6)
				print(f'Total: {t_total}\n')
		return output, hidden

	def init_hidden(self,device):
		h = torch.zeros((1+self.rnn.bidirectional)*self.rnn.num_layers,1,self.rnn.hidden_size,device=device)
		if self.rnn_type == 'lstm':
			c = torch.zeros((1+self.rnn.bidirectional)*self.rnn.num_layers,1,self.rnn.hidden_size,device=device)
			return (h,c)
		elif self.rnn_type == 'gru':
			return h

	def freeze_layer(self,layer):
		relevant_parameters = [i for i,(param_name,_) in enumerate(self.rnn.named_parameters()) if 'l'+str(layer) in param_name]
		for i,cur_parameter in enumerate(self.rnn.parameters()):
			if i in relevant_parameters:
				cur_parameter.requires_grad=False

	def unfreeze_layer(self,layer):
		relevant_parameters = [i for i,(param_name,_) in enumerate(self.rnn.named_parameters()) if 'l'+str(layer) in param_name]
		for i,cur_parameter in enumerate(self.rnn.parameters()):
			if i in relevant_parameters:
				cur_parameter.requires_grad=True

	def unfreeze_all(self):
		for cur_parameter in self.rnn.parameters():
			cur_parameter.requires_grad=True

	def freeze_all_except(self,layer):
		for i in range(self.rnn.num_layers):
			if i != layer:
				self.freeze_layer(i)

class DecoderNN(nn.Module):

	def __init__(self,input_size,hidden_layers=[],**kwargs):
		super(DecoderNN,self).__init__()
		net = []
		layer = input_size

		for next_layer in hidden_layers:
			net += [nn.Linear(layer,next_layer),nn.ReLU()]
			layer = next_layer
		net += [nn.Linear(layer,1)]

		self.pipe = nn.Sequential(*net)
		self.sigmoid = nn.Sigmoid()

	def forward(self,data,x):
		data = torch.cat([data,x],dim=1)
		return self.pipe(data)

	def predict(self,data,x):
		data = torch.cat([data,x],dim=1)
		return torch.round(self.sigmoid(self.pipe(data)))


class FCNetwork(nn.Module):
	def __init__(self,n_inputs,n_outputs,hidden_layers=[],**kwargs):
		super().__init__()
		net = []
		layers = [n_inputs] + hidden_layers + [n_outputs]
		layer = n_inputs

		for next_layer in hidden_layers:
			net += [nn.Linear(layer,next_layer),nn.ReLU()]
			layer = next_layer
		net += [nn.Linear(layer,n_outputs)]

		self.pipe = nn.Sequential(*net)

	def forward(self,x,*args,**kwargs):
		return self.pipe(x), None

	def init_hidden(self,*args,**kwargs):
		return None
import torch
from torch import nn
import copy

default = {
	'unique_net':False,
	'use_gpu':False,
	'use_mixer':True,
	'mixer_hidden_size':32,
	'optimizer':'adam',
	'lr':1e-3
}

class Agent:
	def __init__(self,env,network,**kwargs):
		self.device = torch.device("cuda" if torch.cuda.is_available() and kwargs.get('use_gpu',default['use_gpu']) else "cpu")
		print(f'Device: {self.device}')

		agent_names = copy.copy(env.agents)
		obs_space = env.obs_size
		self.n_actions = env.n_actions
		self.state_size = env.state_size

		if kwargs.get('use_mixer',default['use_mixer']):
			mixer_hidden = kwargs.get('mixer_hidden_size',default['mixer_hidden_size'])
			self.mixer = Mixer(env.state_size,len(agent_names),hidden_size=mixer_hidden)
		else:
			self.mixer = VDNMixer()

		unique_net = kwargs.get('unique_net',default['unique_net'])
		if isinstance(network,torch.nn.Module):
			network = network.to(self.device)
			self.agents = {agent_name:network for agent_name in agent_names}
		else:
			assert isinstance(network,dict) and list(network.keys()) == agent_names, 'Not valid networks for DQN agents'
			self.agents = network
		if not unique_net:
			for agent_name in self.agents.keys():
				self.agents[agent_name] = copy.deepcopy(self.agents[agent_name])
		
		self.target_agents = copy.deepcopy(self.agents)
		for net in self.target_agents.values():
			net.eval()

		parameters = []
		for net in self.agents.values():
			parameters += list(net.parameters())
		parameters += list(self.mixer.parameters())
		parameters = list(set(parameters))

		optim = kwargs.get('optimizer',default['optimizer']).lower()
		lr = kwargs.get('lr',default['lr'])
		assert optim in ['adam','sgd'], 'Unsupported optimizer'
		if optim == 'adam':
			self.optimizer = torch.optim.Adam(parameters,lr=lr)
		elif optim == 'sgd':
			self.optimizer = torch.optim.SGD(parameters,lr=lr)

		

class Mixer(nn.Module):
	def __init__(self,state_space,n_agents,hidden_size=64):
		super().__init__()
		print('Initializing QMixer...')
		self.n_agents = n_agents
		self.hidden_size = hidden_size

		self.W1 = nn.Linear(state_space,n_agents*hidden_size)
		self.B1 = nn.Linear(state_space,hidden_size)
		self.W2 = nn.Linear(state_space,hidden_size)
		self.B2 = nn.Sequential(
					nn.Linear(state_space,state_space),
					nn.ReLU(),
					nn.Linear(state_space,1))
		self.ELU = nn.ELU()

	def forward(self,Qagents,state):

		w1 = torch.abs(self.W1(state)).reshape((-1,self.n_agents,self.hidden_size))
		b1 = self.B1(state).reshape((-1,1,self.hidden_size))
		Qtot = self.ELU(torch.add(torch.bmm(Qagents.unsqueeze(1),w1),b1))

		w2 = torch.abs(self.W2(state)).reshape((-1,self.hidden_size,1))
		b2 = self.B2(state).reshape((-1,1,1))
		Qtot = torch.add(torch.bmm(Qtot,w2),b2).reshape(-1)
		return Qtot

class VDNMixer(nn.Module):
	def __init__(self,*args,**kwargs):
		super().__init__()

	def forward(self,Qagents,state):
		return torch.sum(Qagents,dim=1)

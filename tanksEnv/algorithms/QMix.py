import torch
from torch import nn
import copy
import math
import random
from collections import deque

default = {
	'unique_net':False,
	'use_gpu':False,
	'use_mixer':True,
	'mixer_hidden_size':32,
	'optimizer':'adam',
	'lr':1e-3,
	'gamma':0.9,
	'loss_function':nn.MSELoss,
	'epsilon':.5,
	'buffer_size':None,
	'batch_size':5,
	'net_sync_period':2
}

class QMixRunner:
	def __init__(self,env,qmix,**kwargs):
		self.env = env
		self.qmix = qmix
		self.batch_size = kwargs.get('batch_size',default['batch_size'])
		self.buffer_size = kwargs.get('buffer_size',default['buffer_size'])
		self.device = self.qmix.device
		self.net_sync_period = kwargs.get('net_sync_period',default['net_sync_period'])

	def run(self,N_episodes,render=False):
		qmix_buffer = deque(maxlen=self.buffer_size)
		agents_buffers = {agent:deque(maxlen=self.buffer_size) for agent in self.env.agents}

		for episode_id in range(N_episodes):
			self.env.reset()
			hidden = self.qmix.init_hidden()
			total_reward = 0.0
			episode_length = 0.0
			loss = 0.0
			for agent in self.env.agent_iter():
				if agent == self.env.agents[0]:
					episode_length += 1.0
					self.qmix.set_epsilon(episode_id)
					state = torch.tensor(self.env.get_state(),dtype=torch.float32).unsqueeze(0)
					if qmix_buffer:
						qmix_buffer[-1].next_state = copy.deepcopy(state)
					qmix_buffer.append(MixerTransition(state,0,None,True))

				obs,reward,done,_ = self.env.last()
				total_reward += reward
				if len(qmix_buffer) > 1:
					qmix_buffer[-2].reward += reward
					qmix_buffer[-2].done &= done
				obs = torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
				if done:
					action = None
				else:
					action,hidden[agent] = self.qmix.get_action(agent,obs,hidden[agent])
				buffer = agents_buffers[agent]
				if buffer:
					buffer[-1].next_obs = copy.deepcopy(obs)
				buffer.append(AgentTransition(obs,action,None))

				self.env.step(action)

				if render:
					self.env.render()

			if episode_id % self.net_sync_period == 0:
				self.qmix.sync_nets()

			# if loss != None:
			# 	loss /= episode_length

			qmix_buffer.pop()
			for buffer in agents_buffers.values():
				buffer.pop()
			# yield episode_id,episode_length,total_reward,loss
		self.qmix.learn((qmix_buffer,agents_buffers))

class QMix:
	def __init__(self,env,network,**kwargs):
		self.device = torch.device("cuda" if torch.cuda.is_available() and kwargs.get('use_gpu',default['use_gpu']) else "cpu")
		print(f'Device: {self.device}')

		agent_names = copy.copy(env.agents)
		self.n_actions = None
		self.state_size = env.state_size

		if kwargs.get('use_mixer',default['use_mixer']):
			mixer_hidden = kwargs.get('mixer_hidden_size',default['mixer_hidden_size'])
			self.mixer = Mixer(env.state_size,len(agent_names),hidden_size=mixer_hidden)
		else:
			self.mixer = VDNMixer()

		self.target_mixer = copy.deepcopy(self.mixer)

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

		self.gamma = kwargs.get('gamma',default['gamma'])
		self.loss_fn = kwargs.get('loss_function',default['loss_function'])()

		if 'epsilon_decay' in kwargs.keys():
			self.eps_min = kwargs['epsilon_decay']['stop']
			self.eps_max = kwargs['epsilon_decay']['start']
			self.eps_period = kwargs['epsilon_decay']['period']
			self.eps_decay_shape = kwargs['epsilon_decay']['shape']
			self.epsilon = self.eps_max
			self.eps_decay = True
		else:
			self.epsilon = kwargs.get('epsilon',default['epsilon'])
			self.eps_decay = False

	def learn(self,batches):
		qmix_batch = batches[0]
		agents_batch = batches[1]

		states = torch.cat([trans.state for trans in qmix_batch])
		next_states = torch.cat([trans.next_state for trans in qmix_batch])
		dones = torch.BoolTensor([trans.done for trans in qmix_batch],device=self.device)
		rewards = torch.tensor([trans.reward for trans in qmix_batch],device=self.device)
		actions = torch.tensor([[trans.action for trans in batch] for batch in agents_batch.values()],device=self.device).transpose(0,1).unsqueeze(-1)

		Qagents,hiddens = self.get_Q_agents(agents_batch)
		Qagents = Qagents.gather(2,actions).squeeze(-1)

		Qvalues = self.mixer(Qagents,states)

		with torch.no_grad():
			Qagents_target = self.get_Q_target_agents(agents_batch).max(2).values
			Qtarget = self.target_mixer(Qagents_target,next_states)
			Qtarget[dones] = 0.0
			Qtarget = Qtarget.detach()
			Qtarget = rewards+self.gamma*Qtarget

		loss = self.loss_fn(Qvalues,Qtarget)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def get_action(self,agent_name,obs,hidden):
		net = self.agents[agent_name]
		if self.n_actions == None:
			q_values,_ = net(obs.to(self.device))
			self.n_actions = q_values.size(-1)
		assert(obs.shape[0] == 1)
		with torch.no_grad():
				Qvalues,next_hidden = net(obs.to(self.device),hidden=hidden)
		if random.random() < self.epsilon:
			action = random.randrange(self.n_actions)
		else:	
			action = Qvalues.cpu().squeeze().argmax().numpy()
		return int(action), next_hidden

	def init_hidden(self):
		return {agent:net.init_hidden(self.device) for agent,net in self.agents.items()}

	def get_Q_agents(self,batch,hidden=None):
		Qagents = []
		hiddens = []
		for agent,net in self.agents.items():
			obs =  torch.cat([trans.obs for trans in batch[agent]])
			Q,hidden = net(obs)
			hiddens.append(hidden)
			Qagents += [Q.unsqueeze(1)]
		Qagents = torch.cat(Qagents,dim=1)
		return Qagents,hiddens

	def get_Q_target_agents(self,batch,hidden=None):
		Qagents = []
		for agent,net in self.target_agents.items():
			next_obs =  torch.cat([trans.next_obs for trans in batch[agent]])
			Q, _ = net(next_obs,hidden=hidden)
			Qagents += [Q.unsqueeze(1)]
		Qagents = torch.cat(Qagents,dim=1)
		return Qagents

	def sync_nets(self):
		self.target_mixer.load_state_dict(self.mixer.state_dict())
		for name,net in self.agents.items():
			self.target_agents[name].load_state_dict(net.state_dict())

	def set_epsilon(self,t):
		if not self.eps_decay:
			return self.epsilon
		shape = self.eps_decay_shape.lower()
		if shape == 'exponential':
			rate = math.log(self.eps_min/self.eps_max)/self.eps_period
			epsilon = self.eps_max*math.exp(t*rate)
		elif shape == 'linear':
			rate = (self.eps_max-self.eps_min)/self.eps_period
			epsilon = self.eps_max - t*rate
		else:
			print('Unknown epsilon decay shape')
		self.epsilon = max(self.eps_min,epsilon)
		return self.epsilon


class Mixer(nn.Module):
	def __init__(self,state_space,n_agents,hidden_size=64):
		super().__init__()
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

class MixerTransition:
    def __init__(self,state,reward,next_state,done):
        self.state = state
        self.reward = reward
        self.next_state = next_state
        self.done = done
    
    def __str__(self):
        return str((self.state,self.reward,self.next_state,self.done))

    def __repr__(self):
        return self.__str__()

class AgentTransition:
    def __init__(self,obs,action,next_obs):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs

    def __str__(self):
        return str((self.obs,self.action,self.next_obs))
    def __repr__(self):
        return self.__str__()
import torch
from torch import nn
import copy
import math
import random
from collections import deque
from tanksEnv.utils.functionnal import is_rnn, weights_init_uniform
import time

default = {
	'unique_net':False,
	'use_gpu':False,
	'use_mixer':True,
	'mixer_hidden_layer':32,
	'optimizer':'rmsprop',
	'lr':1e-3,
	'gamma':0.9,
	'loss_function':nn.MSELoss,
	'epsilon':.1,
	'buffer_size':None,
	'batch_size':10,
	'net_sync_period':2,
	'max_depth':0
}

class QMixRunner:
	def __init__(self,env,qmix,**kwargs):
		self.env = env
		self.qmix = qmix
		self.batch_size = kwargs.get('batch_size',default['batch_size'])
		self.buffer_size = kwargs.get('buffer_size',default['buffer_size'])
		self.device = self.qmix.device
		self.net_sync_period = kwargs.get('net_sync_period',default['net_sync_period'])

	def run(self,N_episodes,render=False,train=True):
		qmix_buffer = deque(maxlen=self.buffer_size)
		agents_buffers = {agent.name:deque(maxlen=self.buffer_size) for agent in self.env.get_agents()}

		# qmix_buffer_test = []
		# agents_buffers_test = {agent.name:[] for agent in self.env.get_agents()}
		if not train:
			# agent.net.eval()
			for agent in self.env.agents:
				agent.epsilon = 0
		# else:
		# 	agent.net.train()
			

		for episode_id in range(N_episodes):
			self.env.reset()
			hidden = self.qmix.init_hidden()
			total_reward = 0.0
			episode_length = 0
			loss = None
			all_obs = {agent.name:[] for agent in self.env.get_agents()}
			episode_lengths = {agent.name:0 for agent in self.env.get_agents()}
			for agent_obj, first_agent in self.env.agent_iter():
				agent = agent_obj.name
				episode_lengths[agent] += 1
				if first_agent:
					episode_length += 1
					if train:
						self.qmix.set_epsilon(episode_id)
					state = torch.tensor(self.env.get_state(),dtype=torch.float32,device=self.device).unsqueeze(0)
					if episode_length > 1:
						qmix_buffer[-1].next_state = state
						# qmix_buffer_test[-1][2] = len(qmix_buffer_test)
					qmix_buffer.append(MixerTransition(state,0,None,True))
					# qmix_buffer_test.append([len(qmix_buffer_test),0,None,True])

				obs,reward,done,_ = self.env.last()
				if first_agent:
					total_reward += reward
				if episode_length > 1:
					qmix_buffer[-2].reward = reward
					qmix_buffer[-2].done &= done
					# qmix_buffer_test[-2][1] = reward
					# qmix_buffer_test[-2][3] &= done
				
				obs = torch.tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
				if self.qmix.rnn:
					all_obs[agent].append(obs)
				if done:
					action = None
				else:
					action,hidden[agent] = self.qmix.get_action(agent,obs,hidden[agent])
				buffer = agents_buffers[agent]
				# buffer_test = agents_buffers_test[agent]
				if buffer:
					buffer[-1].next_obs = obs
					# buffer[-1].done = done
					# buffer_test[-1][2] = len(buffer_test)
					# buffer_test[-1][3] = done
				buffer.append(AgentTransition(obs,action or 0,None,all_obs[agent],len(all_obs[agent]),done,False))
				# buffer_test.append([len(buffer_test),action or 0, None,None,False])
				self.env.step(action,prompt_action=render)

				if render:
					self.env.render()

			if train and episode_id % self.net_sync_period == 0:
				self.qmix.sync_nets()
			
			qmix_buffer.pop()
			# qmix_buffer_test.pop()
			# empty_trans_test = [0,0,0,True,True]
			empty_obs = torch.zeros_like(obs)
			empty_trans = AgentTransition(empty_obs,0,empty_obs,all_obs[agent],1,True,True)
			for agent,buffer in agents_buffers.items():
				buffer.pop()
				# agents_buffers_test[agent].pop()
				last_trans = buffer[-1]
				last_trans.done = 1
				for _ in range(episode_length-episode_lengths[agent]):
					buffer.append(last_trans)
					# agents_buffers_test[agent].append(empty_trans_test)
				buffer[-1] = last_trans

			# print('\n'*3,qmix_buffer_test)
			# for agent,buffer_test in agents_buffers_test.items():
			# 	print('\n',agent,'\n',buffer_test)

			if train and len(qmix_buffer) >= self.batch_size:
				
				agents,agents_buffers_list = zip(*list(agents_buffers.items()))
				buffers = (qmix_buffer,*agents_buffers_list)
				batches = list(zip(*random.sample(list(zip(*buffers)),self.batch_size)))
				qmix_batch = batches[0]
				agents_batch = batches[1:]
				agents_batch = dict(zip(agents, agents_batch))
				batches = (qmix_batch,agents_batch)
				loss = self.qmix.learn(batches)

			yield episode_id,episode_length,total_reward,loss

class QMix:
	def __init__(self,env,network,**kwargs):
		self.device = torch.device("cuda" if torch.cuda.is_available() and kwargs.get('use_gpu',default['use_gpu']) else "cpu")
		print(f'Device: {self.device}')

		agent_names = [agent.name for agent in env.get_agents()]
		self.n_actions = None
		self.state_size = env.state_size

		if kwargs.get('use_mixer',default['use_mixer']):
			mixer_hidden = kwargs.get('mixer_hidden_layer',default['mixer_hidden_layer'])
			self.mixer = Mixer(env.state_size,len(agent_names),hidden_layer_size=mixer_hidden).to(self.device)
		else:
			self.mixer = VDNMixer()


		self.target_mixer = copy.deepcopy(self.mixer)
		self.target_mixer.eval()

		unique_net = kwargs.get('unique_net',default['unique_net'])
		if isinstance(network,torch.nn.Module):
			network = network.to(self.device)
			self.agents = {agent_name:network for agent_name in agent_names}
		else:
			assert isinstance(network,dict) and list(network.keys()) == agent_names, 'Not valid Q-Networks dict.'
			self.agents = network
		for agent_name,net in self.agents.items():
			if not unique_net:
				self.agents[agent_name] = copy.deepcopy(net.to(self.device))
			else:
				self.agents[agent_name] = net.to(self.device)
		
		self.target_agents = copy.deepcopy(self.agents)
		for net in self.target_agents.values():
			net.eval()

		parameters = []
		for net in self.agents.values():
			parameters += list(net.parameters())
			if unique_net:
				break
		parameters += list(self.mixer.parameters())

		optim = kwargs.get('optimizer',default['optimizer']).lower()
		lr = kwargs.get('lr',default['lr'])
		assert optim in ['adam','sgd','rmsprop'], 'Unsupported optimizer'
		if optim == 'adam':
			self.optimizer = torch.optim.Adam(parameters,lr=lr)
		elif optim == 'sgd':
			self.optimizer = torch.optim.SGD(parameters,lr=lr)
		elif optim == 'rmsprop':
			self.optimizer = torch.optim.RMSprop(parameters, lr=lr)
		else:
			print('Unknown optimizer')

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
		
		self.rnn = any([is_rnn(net) for net in self.agents.values()])
		self.max_depth = kwargs.get('max_depth',default['max_depth'])

	def learn(self,batches):
		qmix_batch = batches[0]
		agents_batch = batches[1]

		states = torch.cat([trans.state for trans in qmix_batch])
		next_states = torch.cat([trans.next_state for trans in qmix_batch])
		dones = torch.BoolTensor([trans.done for trans in qmix_batch])
		rewards = torch.tensor([trans.reward for trans in qmix_batch],device=self.device,dtype=torch.float32)
		actions = torch.tensor([[trans.action for trans in batch] for batch in agents_batch.values()],device=self.device).transpose(0,1).unsqueeze(-1)

		# print(states[0,:])
		# print(agents_batch['agent0'][0].obs)
		# print(agents_batch['agent1'][0].obs)
		# print('\n')
		# time.sleep(1)

		Qagents,hiddens = self.get_Q_agents(agents_batch)
		Qagents = Qagents.gather(2,actions).squeeze(-1)

		Qvalues = self.mixer(Qagents,states)
		with torch.no_grad():
			Qagents_target = self.get_Q_target_agents(agents_batch,hiddens=hiddens).max(2).values
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
			q_values,_ = net(obs)
			self.n_actions = q_values.size(-1)
		assert(obs.shape[0] == 1)
		with torch.no_grad():
				Qvalues,next_hidden = net(obs,hidden=hidden)

		if random.random() < self.epsilon:
			action = random.randrange(self.n_actions)
		else:	
			action = Qvalues.cpu().squeeze().argmax().numpy()
		return int(action), next_hidden

	def init_hidden(self):
		return {agent:net.init_hidden(self.device) for agent,net in self.agents.items()}

	def get_Q_agents(self,batch):
		Qagents = []
		hiddens_out = self.init_hidden()
		for agent,net in self.agents.items():
			if self.rnn:
				if self.max_depth:
					obs = [
					torch.stack(trans.all_obs[max(0,trans.index-self.max_depth):trans.index]).squeeze(1) for trans in batch[agent]]
				else:
					obs = [torch.stack(trans.all_obs[:trans.index]).squeeze(1) for trans in batch[agent]]
			else:
				obs =  torch.cat([trans.obs for trans in batch[agent]])
			Q,hidden = net(obs)
			is_dead = torch.BoolTensor([trans.dead for trans in batch[agent]])
			Q[is_dead] = 0.0

			hiddens_out[agent] = hidden
			Qagents += [Q.unsqueeze(1)]
		Qagents = torch.cat(Qagents,dim=1)
		return Qagents,hiddens_out

	def get_Q_target_agents(self,batch,hiddens=None):
		hiddens_in=hiddens
		Qagents = []
		for agent,net in self.target_agents.items():
			next_obs =  torch.cat([trans.next_obs for trans in batch[agent]])
			if self.rnn:
				next_obs = next_obs.unsqueeze(0)
			Q, _ = net(next_obs,hidden=hiddens_in[agent])
			is_done = torch.BoolTensor([trans.done for trans in batch[agent]])
			Q[is_done] = 0.0
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
	def __init__(self,state_space,n_agents,hidden_layer_size=32):
		super().__init__()
		self.n_agents = n_agents

		# self.layers = []
		# self.layers_size = [n_agents] + hidden_layers + [1]
		# prev_hidden = self.layers_size[0]
		# for next_hidden in self.layers_size[1:-1]:
		# 	W = nn.Linear(state_space,prev_hidden*next_hidden)
		# 	B = nn.Linear(state_space,next_hidden)
		# 	self.layers.append((W,B))
		# 	prev_hidden = next_hidden

		# B_last = nn.Sequential(
		# 			nn.Linear(state_space,state_space),
		# 			nn.ReLU(),
		# 			nn.Linear(state_space,1))
		# W_last = nn.Linear(state_space,hidden_layers[-1])

		# self.layers.append((W_last,B_last))

		self.hidden_size = hidden_layer_size
		self.W1 = nn.Linear(state_space,n_agents*hidden_layer_size)
		self.B1 = nn.Linear(state_space,hidden_layer_size)
		self.W2 = nn.Linear(state_space,hidden_layer_size)
		self.B2 = nn.Sequential(
					nn.Linear(state_space,state_space),
					nn.ReLU(),
					nn.Linear(state_space,1))
		
		self.ELU = nn.ELU()


		# weights_init_uniform(self,1e-10)
		# weights_init_uniform(self.B2,1e-10)

	def forward(self,Qagents,state):

		# Qtot = Qagents.unsqueeze(1)
		# prev_hidden = self.layers_size[0]
		# for idx in range(len(self.layers)):
		# 	next_hidden = self.layers_size[idx+1]
		# 	W,B = self.layers[idx]
		# 	w = torch.abs(W(state)).reshape((-1,prev_hidden,next_hidden))
		# 	b = B(state).reshape((-1,1,next_hidden))
		# 	if idx:
		# 		Qtot = self.ELU(Qtot)
		# 	Qtot = torch.add(torch.bmm(Qtot,w),b)
		# 	prev_hidden = next_hidden

		w1 = torch.abs(self.W1(state)).reshape((-1,self.n_agents,self.hidden_size))
		b1 = self.B1(state).reshape((-1,1,self.hidden_size))
		Qtot = self.ELU(torch.add(torch.bmm(Qagents.unsqueeze(1),w1),b1))

		w2 = torch.abs(self.W2(state)).reshape((-1,self.hidden_size,1))
		b2 = self.B2(state).reshape((-1,1,1))
		Qtot = torch.add(torch.bmm(Qtot,w2),b2).reshape(-1)

		return Qtot.flatten()

	# def to(self,device):
	# 	for W,B in self.layers:
	# 		W = W.to(device)
	# 		B = B.to(device)
	# 	return self


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
        return 'MixerTransition('+str((self.state,self.reward,self.next_state,self.done))+')'

    def __repr__(self):
        return self.__str__()

class AgentTransition:
	def __init__(self,obs,action,next_obs,all_obs,index,done,dead):
		self.obs = obs
		self.action = action
		self.next_obs = next_obs
		self.all_obs=all_obs
		self.index=index
		self.done=done
		self.dead=dead

	def __str__(self):
		return 'AgentTransition('+str((self.obs,self.action,self.next_obs,self.index))+')'
	def __repr__(self):
		return self.__str__()
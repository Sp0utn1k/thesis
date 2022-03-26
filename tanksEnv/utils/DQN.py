import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple, deque
import math
import copy
import os
from utils.functionnal import is_rnn
import time

Episode = namedtuple('Episode',['state','action','reward','next_state','done','all_states','index'])

class Agent:

	def __init__(self,*args,**kwargs):
		
		self.device = torch.device("cuda" if torch.cuda.is_available() and kwargs.get('use_gpu',False) else "cpu")
		print(f'Device: {self.device}')

		self.net = kwargs['network'].to(self.device)
		self.target_net = copy.deepcopy(self.net)
		self.target_net.eval()
		
		self.epsilon = kwargs.get('epsilon',.5)
		self.gamma = kwargs.get('gamma',.9)

		self.loss_fn = kwargs.get('loss_fn',nn.MSELoss())
		self.optimizer = kwargs.get('optimizer',torch.optim.Adam)(self.net.parameters(),lr=kwargs.get('lr',1e-3))
		self.n_actions = None

		if 'epsilon_decay' in kwargs.keys():
			self.eps_min = kwargs['epsilon_decay']['stop']
			self.eps_max = kwargs['epsilon_decay']['start']
			self.eps_period = kwargs['epsilon_decay']['period']
			self.eps_decay_shape = kwargs['epsilon_decay']['shape']
			self.epsilon = self.eps_max

		self.rnn = is_rnn(self.net)
		self.max_depth = kwargs.get('max_depth',0)

	def sync_nets(self):
		self.target_net.load_state_dict(self.net.state_dict())

	def init_hidden(self):
		return self.net.init_hidden(self.device)

	def set_epsilon(self,t):
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

	def get_action(self,state,hidden):
		if self.n_actions == None:
			q_values,_ = self.net(state.to(self.device))
			self.n_actions = q_values.size(-1)
		assert(state.shape[0] == 1)
		if random.random() < self.epsilon:
			action = random.randrange(self.n_actions)
			next_hidden = hidden
		else:
			with torch.no_grad():
				Qvalues,next_hidden = self.net(state.to(self.device),hidden=hidden)
				action = Qvalues.cpu().squeeze().argmax().numpy()
		return int(action), next_hidden

	def train_net(self,batch,benchmark=False):


		if benchmark:
			timer = [0,0]
			timer[0] = time.time()
			timer[1] = time.time()
		device = self.device
		action = torch.tensor([episode.action for episode in batch],device=device)
		reward = torch.tensor([episode.reward for episode in batch],device=device)
		next_state = torch.cat([episode.next_state for episode in batch])
		done = torch.cuda.BoolTensor([episode.done for episode in batch])
		if self.rnn:
			if self.max_depth:
				all_states = [
				torch.stack(episode.all_states[max(0,episode.index-self.max_depth):episode.index]).squeeze(1) for episode in batch]
			else:
				all_states = [torch.stack(episode.all_states[:episode.index]).squeeze(1) for episode in batch]
			next_state = next_state.unsqueeze(0)
			states = all_states
		else:
			state = torch.cat([episode.state for episode in batch])
			states = state

		if benchmark:
			t_load = 1000*round(time.time()-timer[1],6)
			print(f'Load tensors: {t_load}')
			timer[1] = time.time()

		# print([s.shape for s in all_states])
		Qvalues,hidden = self.net(states)
		Qvalues = Qvalues.gather(1, action.unsqueeze(-1)).squeeze(-1)
		with torch.no_grad():
			Q_next,_ = self.target_net(next_state,hidden=hidden)
			Q_next = Q_next.max(1)[0]
			Q_next[done] = 0.0
			Q_next = Q_next.detach()

		target_Q = reward + self.gamma*Q_next
		loss = self.loss_fn(Qvalues,target_Q)

		if benchmark:
			t_loss = 1000*round(time.time()-timer[1],6)
			print(f'Compute loss: {t_loss}')
			timer[1] = time.time()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if benchmark:
			t_backward = 1000*round(time.time()-timer[1],6)
			print(f'Learning time: {t_backward}')
			t_total = 1000*round(time.time()-timer[0],6)
			print(f'Total: {t_total}\n')

		return loss.item()

class Buffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE):
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return batch

class DQNRunner:
	def __init__(self,env,agent,**kwargs):
		self.env = env
		self.agent = agent
		self.buffer = Buffer(kwargs.get('buffer_size',None))
		self.batch_size = kwargs.get('batch_size',10)
		self.device = self.agent.device
		self.net_sync_period = kwargs.get('net_sync_period',1)
		# self.rnn = agent.rnn

	def run(self,N_episodes,render=False):

		timer = {'total':0,'train':0}
		agent = self.agent
		env = self.env
		hidden = agent.init_hidden()
		for episode_id in range(N_episodes):
			
			total_reward = 0.0
			state = torch.tensor(env.reset(),device=self.device,dtype=torch.float32).unsqueeze(0)
			done = False
			episode_length = 0.0
			loss = 0.0
			all_states = []
			while not done:
				# timer['total'] = time.time()
				episode_length += 1.0
				agent.set_epsilon(episode_id)
				action, hidden = agent.get_action(state,hidden)
				next_state, reward, done, _ = env.step(action)
				next_state = torch.tensor(next_state,device=self.device,dtype=torch.float32).unsqueeze(0)
				all_states.append(state)
				episode = Episode(state,action,reward,next_state,done,all_states,len(all_states))
				self.buffer.append(episode)
				state = copy.deepcopy(next_state)
				total_reward += reward
				if loss != None and len(self.buffer) >= self.batch_size:
					# timer['train'] = time.time()
					loss += agent.train_net(self.buffer.sample(self.batch_size))
					# training_time = 1000*round(time.time() - timer['train'],6)
					# print(f'Training: {training_time}')
				else:
					loss = None
				if render:
					env.render()
				# total_time = 1000*round(time.time() - timer['total'],6)
				# print(f'Total: {total_time}')

			if episode_id % self.net_sync_period == 0:
				agent.sync_nets()

			if loss != None:
				loss /= episode_length

			yield episode_id,episode_length,total_reward,loss
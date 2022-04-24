import torch
import torch.nn as nn
import random
import time
import numpy as np
import math
import os
import copy
from collections.abc import Iterable
from collections import deque
import yaml
from tensorboardX import SummaryWriter

class EncoderNN(nn.Module):
	def __init__(self,input_size,output_size,hidden_size):
		super().__init__()
		self.rnn = nn.LSTM(input_size,hidden_size)
		self.fc = nn.Sequential(nn.ReLU(),
								nn.Linear(hidden_size,output_size))

	def forward(self,x):

		# hn = torch.zeros(self.rnn.num_layers,x.shape[1],self.rnn.hidden_size)
		# cn = torch.zeros(self.rnn.num_layers,x.shape[1],self.rnn.hidden_size)
		# for i in range(x.shape[0]):
		# 	out, (hn, cn) = self.rnn(x[i,:,:].unsqueeze(0),(hn,cn))
		# 	out = self.fc(out)


		self.rnn.flatten_parameters()
		out, _ = self.rnn(x)
		out = self.fc(out)[-1,:,:]


		return out

class AgentDQN(nn.Module):
	def __init__(self,encoder_input_size,encoder_output_size,encoder_hidden_size,input_size,N_actions,hidden_layers):
		super().__init__()
		self.encoder = EncoderNN(encoder_input_size,encoder_output_size,encoder_hidden_size)

		net = []
		layer_size = input_size+encoder_output_size

		for (next_layer,next_layer_size) in hidden_layers:
			if next_layer.lower() == 'linear':
				net += [nn.Linear(layer_size,next_layer_size),nn.ReLU()]
				layer_size = next_layer_size
			elif next_layer.upper() == 'GRU':
				pass

		net += [nn.Linear(layer_size,N_actions)]

		self.pipe = nn.Sequential(*net)

	def forward(self,batch):

		observation = torch.cat([episode[0].unsqueeze(0) for episode in batch])
		terrains = [episode[1] for episode in batch]
		coded_terrains = []
		for terrain in terrains:
			coded_terrains.append(self.encoder(terrain))
		coded_terrains = torch.stack(coded_terrains).squeeze(1)
		state = torch.cat([observation,coded_terrains],1)

		return self.pipe(state)


class Agent:
	def __init__(self,env,**kwargs):

		use_gpu = kwargs.get('use_gpu',False)
		self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
		
		encoder_input_size = env.decoded_state_size
		encoder_output_size = kwargs.get('encoder_output_size',32)
		encoder_hidden_size = kwargs.get('encoder_hidden_size',32)
		input_size = env.observation_space
		self.n_actions = env.N_actions
		hidden_layers = kwargs.get('hidden_layers',[])
		self.net = AgentDQN(encoder_input_size,encoder_output_size,encoder_hidden_size,
					input_size,self.n_actions,hidden_layers).to(self.device)
		self.tgt_net = copy.deepcopy(self.net)
		self.tgt_net.eval()

		self.loss_fn = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.net.parameters())
		
		self.gamma = kwargs.get('gamma',.9)

		if 'epsilon_decay' in kwargs.keys():
			self.eps_min = kwargs['epsilon_decay']['stop']
			self.eps_max = kwargs['epsilon_decay']['start']
			self.eps_period = kwargs['epsilon_decay']['period']
			self.eps_decay_shape = kwargs['epsilon_decay']['shape']
			self.epsilon = self.eps_max
		else:
			self.epsilon = kwargs.get('epsilon',.5)

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

	def sync_nets(self):
		self.tgt_net.load_state_dict(self.net.state_dict())

	def get_action(self,S):
		if random.random() < self.epsilon:
			A = random.randrange(self.n_actions)
		else:
			with torch.no_grad():
				A = self.net([S]).cpu().squeeze().argmax().numpy()
		return int(A)

	def train(self,batch):

		device = self.device
		S = [t.state for t in batch]
		A = torch.tensor([t.action for t in batch],device=device)
		R = torch.tensor([t.reward for t in batch],device=device)
		S_ = [t.next_state for t in batch]
		done = torch.cuda.BoolTensor([t.done for t in batch])

		with torch.no_grad():
			Q_ = self.tgt_net(S_).max(1)[0]
			Q_[done] = 0.0
			Q_ = Q_.detach()
		
		Q = self.net(S).gather(1, A.unsqueeze(-1)).squeeze(-1)
		target_Q = R + self.gamma*Q_

		loss = self.loss_fn(Q,target_Q)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()


class Environment:
	def __init__(self,**kwargs):
		self.maxsize = kwargs.get('maxsize',10)
		use_gpu = kwargs.get('use_gpu',False)
		self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
		self.size = kwargs.get('size','random')
		self.random_size = self.size == 'random'
		self.N_goals = kwargs.get('N_goals',1)
		self.reset()
		self.decoded_state_size = self.decode_goals().shape[-1]
		self.N_actions = 4
		self.observation_space = len(self.observe_state())
		self.max_steps = kwargs.get('max_steps',25)

	def reset(self):
		self.pos = [0,0]
		if self.random_size:
			self.size = (random.randrange(2,self.maxsize),random.randrange(1,self.maxsize))
		grid = [[i,j] for i in range(self.size[0]) for j in range(self.size[1]) if i or j]
		self.goals = random.sample(grid,self.N_goals)

		self.steps_taken = 0
		return (torch.tensor(self.observe_state(),device=self.device),self.decode_goals())

	def take_action(self,action):
		if action == 0 and self.pos[0] > 0:
			self.pos[0] -= 1
		elif action == 1 and self.pos[1] < self.size[1] - 1:
			self.pos[1] += 1
		elif action == 2 and self.pos[0] < self.size[0] - 1:
			self.pos[0] += 1
		elif action == 3 and self.pos[1] > 0:
			self.pos[1] -= 1
		
	def observe_state(self):
		pos = copy.copy(self.pos)
		obs = pos
		return obs

	def step(self,action):
		self.steps_taken += 1
		old_pos = copy.copy(self.pos)
		self.take_action(action)
		if old_pos == self.pos:
			reward = -1
		else:
			reward = -.1
		for i,goal in enumerate(self.goals):
			if goal == self.pos:
				reward = 1
				self.goals.pop(i)


		done = len(self.goals) == 0 or self.steps_taken >= self.max_steps
		state = (torch.tensor(self.observe_state(),device=self.device),self.decode_goals())
		return state, reward,done

	def render(self):

		os.system('clear')
		img = np.array([['. ' for _ in range(self.size[1])]for _ in range(self.size[0])])
		for goal in self.goals:
			img[tuple(goal)] = 'G '
		img[tuple(self.pos)] = 'o '
		for line in img:
			print(''.join(line))

	def decode_goals(self):
		
		state = torch.tensor(self.goals,device=self.device,dtype=torch.float32).flatten()
		return state.reshape(-1,1,2)

		# state = torch.zeros(self.size,device=self.device)
		# state[self.goal] = 1
		# pos = torch.tensor([[[i,j] for i in range(self.size[1])] for j in range(self.size[0])],device=self.device)
		# state = torch.cat([pos,state.unsqueeze(-1)],2).reshape(-1,1,3)

		return state

class Transition:
	def __init__(self,state,action,reward,next_state,done):
		self.state = copy.deepcopy(state)
		self.action = action
		self.reward = reward
		self.next_state = copy.deepcopy(next_state)
		self.done = done

if __name__ == '__main__':

	writer = SummaryWriter()
	config_file = 'config.yml'
	with open('config.yml','r') as file:
		settings = yaml.safe_load(file)

	N_episodes = settings.get('N_episodes',10)
	batch_size = settings.get('batch_size',128)
	net_sync_period = settings.get('net_sync_period',1)
	buffer_size = settings['buffer_size']
	
	env = Environment(**settings)
	agent = Agent(env,**settings)
	buffer = deque(maxlen=buffer_size)

	for episode in range(N_episodes):
		state = env.reset()
		done = False
		total_reward = 0
		agent.set_epsilon(episode)
		if episode % 10 == 0:
			print(f'Episode {episode}, epsilon {round(agent.epsilon,2)}')
		while not done:
			action = agent.get_action(state)
			next_state,reward,done = env.step(action)
			total_reward += reward

			if done:
				transition = Transition(state,action,reward,state,done)
			else:
				transition = Transition(state,action,reward,next_state,done)
			buffer.append(transition)
			state = copy.deepcopy(next_state)

		if len(buffer) < batch_size:
			continue

		batch = random.sample(buffer,batch_size)
		loss = agent.train(batch)
		if episode % net_sync_period == 0:
			agent.sync_nets()
		writer.add_scalar('reward', total_reward, episode)
		writer.add_scalar('loss', loss, episode)

	input('Press enter to show demo')
	agent.net.eval()
	agent.epsilon = 0
	for _ in range(settings.get('N_demos',10)):
		state = env.reset()
		done = False
		while not done:
			action = agent.get_action(state)
			next_state,reward,done = env.step(action)
			state = copy.deepcopy(next_state)
			env.render()
			time.sleep(.1)
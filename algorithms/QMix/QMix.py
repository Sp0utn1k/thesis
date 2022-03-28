import torch
import torch.nn as nn
from pettingzoo.mpe import simple_spread_v2
import random
from collections import namedtuple, deque
import copy
import math
from tensorboardX import SummaryWriter
import time
import numpy as np

BUFFER_SIZE = 50000
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
N_EPISODES = 200000
N_demos = 4
DEMOS_PERIOD = 10000
GAMMA = .9
PRINT_PROGRESS_PERIOD = 1000
NET_SYNC_PERIOD = 10
MAX_CYCLES = 25
USE_QMIXER = True
P_DROPOUT = 0
N_AGENTS = 2
USE_GPU = True

EPSILON_DECAY = {
				'period': N_EPISODES,
				'start':1,
				'stop':.01,
				'shape':'exponential'
				}

class QMixer(nn.Module):
	def __init__(self,state_space,n_agents,hidden_size=64,p_dropout=0):
		super(QMixer,self).__init__()
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
		self.dropout = nn.Dropout(p=p_dropout)

	def forward(self,Qagents,state):

		state = self.dropout(state)
		w1 = torch.abs(self.W1(state)).reshape((-1,self.n_agents,self.hidden_size))
		b1 = self.B1(state).reshape((-1,1,self.hidden_size))
		Qtot = self.ELU(torch.add(torch.matmul(Qagents.unsqueeze(1),w1),b1))

		w2 = torch.abs(self.W2(state)).reshape((-1,self.hidden_size,1))
		b2 = self.B2(state).reshape((-1,1,1))
		Qtot = torch.add(torch.matmul(Qtot,w2),b2).reshape(-1)
		return Qtot

class VDNMixer:
	def __init__(self):
		pass

	def __call__(self,Qagents,state):
		return torch.sum(Qagents,dim=1)

	def load_state_dict(self,args):
		pass

	def state_dict(self):
		return None

	def eval(self):
		pass

	def parameters(self):
		return []

class Mixer:
	def __init__(self,env,device='cpu',use_QMixer=True,p_dropout=0,**kwargs):

		self.agent_names = copy.copy(env.agents)
		obs_space = env.observation_space(self.agent_names[0]).shape[0]
		self.action_space = env.action_space(self.agent_names[0]).n
		self.state_space = env.state_space.shape[0]

		self.agents = {}
		agentNet = AgentDQN(obs_space,self.action_space,hidden_layers=[('linear',32),('linear',32)],
				device=device,p_dropout=p_dropout)
		for name in self.agent_names:
			self.agents[name] = Agent(name,agentNet,epsilon=1,epsilon_decay=EPSILON_DECAY)

		if use_QMixer:
			self.net = QMixer(self.state_space,len(self.agents),p_dropout=p_dropout).to(device)
		else:
			self.net = VDNMixer()
		self.target_net = copy.deepcopy(self.net)
		self.target_net.eval()

		self.gamma = kwargs.get('gamma',0.9)
		lr = kwargs.get('lr',1e-3)	

		self.loss_fn = nn.MSELoss()
		parameters = list(self.net.parameters())
		# for agent in self.agents.values():
		# 	parameters += list(agent.net.parameters())
		parameters += list(agentNet.parameters())
		
		self.optimizer = torch.optim.Adam(parameters,lr=lr)

	def sync_nets(self):
		self.target_net.load_state_dict(self.net.state_dict())
		for agent in self.agents.values():
			agent.sync_nets()

	def set_epsilon(self,t):
		for agent in self.agents.values():
			agent.set_epsilon(t)

		return self.agents['agent_0'].epsilon

	def get_Q_agents(self,batch):
		Qagents = []
		for agent in self.agent_names:
			obs =  torch.cat([step[agent].obs for step in batch])
			Q = self.agents[agent].net(obs).unsqueeze(1)
			Qagents += [Q]
		Qagents = torch.cat(Qagents,dim=1)
		return Qagents

	def get_Q_target_agents(self,batch):
		Qagents = []
		for agent in self.agent_names:
			obs =  torch.cat([step[agent].next_obs for step in batch])
			Q = self.agents[agent].target_net(obs).unsqueeze(1)
			Qagents += [Q]
		Qagents = torch.cat(Qagents,dim=1)
		return Qagents

	def learn(self,batch):

		S = torch.cat([step['tot'].state for step in batch])
		S_ = torch.cat([step['tot'].next_state for step in batch])
		A = torch.tensor([[step[a].action for step in batch] for a in self.agent_names],device=device)
		A = A.transpose(0,1).unsqueeze(2)
		dones = torch.tensor([step['tot'].done for step in batch],device=device)

		Qagents = self.get_Q_agents(batch).gather(2,A).squeeze(2)
		Qtot = self.net(Qagents,S)

		R = torch.tensor([step['tot'].reward for step in batch],device=device)

		with torch.no_grad():
			Qagents_target = self.get_Q_target_agents(batch).max(2).values
			Q_target = self.target_net(Qagents_target,S_)
			Q_target[dones] = 0.0
			Q_target = Q_target.detach()
			y_tot = R+self.gamma*Q_target

		loss = self.loss_fn(y_tot,Qtot)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()

class AgentDQN(nn.Module):
	def __init__(self,n_inputs,n_outputs,hidden_layers=[],device='cpu',p_dropout=0):
		super(AgentDQN,self).__init__()
		net = []
		layers = [n_inputs] + hidden_layers + [n_outputs]
		layer_size = n_inputs

		for (next_layer,next_layer_size) in hidden_layers:
			if next_layer.lower() == 'linear':
				net += [nn.Linear(layer_size,next_layer_size),nn.Dropout(p=p_dropout),nn.ReLU()]
				layer_size = next_layer_size
			elif next_layer.upper() == 'GRU':
				pass

		net += [nn.Linear(layer_size,n_outputs)]

		self.pipe = nn.Sequential(*net).to(device)

	def forward(self,x):
		return self.pipe(x)

class Agent:
	def __init__(self,name,net,**kwargs):
		self.name = name
		self.net = net
		self.target_net = copy.deepcopy(net)
		self.target_net.eval()
		self.action_space = len([param for param in net.parameters()][-1])

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
		self.target_net.load_state_dict(self.net.state_dict())

	def get_action(self,S,done):
		if done:
			return None
		assert(S.shape[0] == 1)
		if random.random() < self.epsilon:
			A = random.randrange(self.action_space)
		else:
			with torch.no_grad():
				A = self.net(S).cpu().squeeze().argmax().numpy()
		return int(A)

class AgentStep:
	def __init__(self,obs,action,next_obs):
		self.obs = obs
		self.action = action
		self.next_obs = next_obs

class MixerStep:
	def __init__(self,state,reward,next_state,done):
		self.state = state
		self.reward = reward
		self.next_state = next_state
		self.done = done

def generate_episode(buffer,env,agents,device='cpu',render=False):

	env.reset()
	last_agent = env.agents[-1]
	episode = deque()
	n_agents = len(env.agents)

	R = 0.0
	total_reward = 0.0
	i = 0
	transition = {}
	for agent_name in env.agent_iter():
		
		agent = agents[agent_name]
		observation, reward, done, _ = env.last()
		observation = torch.tensor(observation,device=device).unsqueeze(0)
		action = agent.get_action(observation,done)
		step = AgentStep(observation,action,None)
		R += reward/n_agents
		if i:
			episode[-1][agent_name].next_obs = observation

		transition[agent_name] = step

		if agent_name == last_agent:
			state = torch.tensor(env.state(),device=device).unsqueeze(0)
			step = MixerStep(state,None,None,None)
			if i:
				episode[-1]['tot'].reward = R
				episode[-1]['tot'].next_state = state
				episode[-1]['tot'].done = all(env.dones.values())
				total_reward += R
				R = 0.0
			transition['tot'] = step
			i += 1
			episode.append(transition)
			transition = {}
		if render:
			env.render()
		
		env.step(action)

	episode.pop()
	buffer.extend(episode)

	return total_reward

def batch_from_buffer(buffer,N_samples):
	batch = random.sample(buffer,N_samples)
	return batch

def show_demo(env,agents,device='cpu',repeat=1):

	for _ in range(repeat):
		env.reset()
		last_agent = env.agents[-1]

		for agent_name in env.agent_iter():
			agent = agents[agent_name]
			observation, reward, done, _ = env.last()
			observation = torch.tensor(observation,device=device).unsqueeze(0)
			action = agent.get_action(observation,done)
			step = AgentStep(observation,action,None)
			env.render()
			env.step(action)

def make_grid_demos(env,agents,device='cpu',repeat=1):
	images = []
	for _ in range(repeat):
		env.reset()
		last_agent = env.agents[-1]

		for agent_name in env.agent_iter():
			agent = agents[agent_name]
			observation, reward, done, _ = env.last()
			observation = torch.tensor(observation,device=device).unsqueeze(0)
			action = agent.get_action(observation,done)
			step = AgentStep(observation,action,None)
			env.step(action)
		img = env.render(mode='rgb_array')
		env.close()
		images.append(img)

	return np.array(images)/255.0

if __name__ == '__main__':

	writer = SummaryWriter(f'runs/{N_AGENTS} agents/')
	device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
	print(f'Device:{device}')

	env = simple_spread_v2.env(max_cycles=MAX_CYCLES,N=N_AGENTS)
	env.reset()
	mixer = Mixer(env,device=device,lr=LEARNING_RATE,gamma=GAMMA,use_QMixer=USE_QMIXER,p_dropout=P_DROPOUT)

	time.sleep(10)
	buffer = deque(maxlen=BUFFER_SIZE)

	rewards = []
	losses = []

	for episode_n in range(N_EPISODES):
		epsilon = mixer.set_epsilon(episode_n)

		R = generate_episode(buffer,env,mixer.agents,device=device)
		rewards.append(R)

		if len(buffer) < BATCH_SIZE:
			continue

		if episode_n % NET_SYNC_PERIOD == 0:
			mixer.sync_nets()

		batch = batch_from_buffer(buffer,BATCH_SIZE)
		loss  = mixer.learn(batch)
		losses.append(loss)

		writer.add_scalar('reward', R/MAX_CYCLES, episode_n)
		writer.add_scalar('loss', loss, episode_n)
		# writer.add_scalar('epsilon', epsilon, episode_n)


		if episode_n % DEMOS_PERIOD == 0:
			img = make_grid_demos(env,mixer.agents,device=device,repeat=N_demos)
			writer.add_image('image',img,episode_n,dataformats='NHWC')


		if (episode_n) % PRINT_PROGRESS_PERIOD == 0:
			R = rewards[-PRINT_PROGRESS_PERIOD:]
			R = sum(R)/len(R)
			loss = losses[-PRINT_PROGRESS_PERIOD:]
			loss = sum(loss)/len(loss)
			print(f'Episode {episode_n}, Reward {round(R/MAX_CYCLES,2)}, loss {round(loss,2)}')
			# for agent in mixer.agents.values():
			# 	print(list(agent.net.parameters())[0])

		# print(list(mixer.agents['agent_0'].target_net.parameters())[-1])
	txt = input('Press enter to show demo ("skip" to skip)')
	if not txt.lower() == 'skip':
		for agent in mixer.agents.values():
			agent.epsilon = 0
			agent.net.eval()
		show_demo(env,mixer.agents,device=device,repeat=N_demos)
	writer.close()
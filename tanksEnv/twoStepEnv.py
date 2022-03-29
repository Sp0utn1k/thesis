import numpy as np
import random

class Environment:

	def __init__(self):
		self.agents = ['agent0','agent1']
		self.n_actions = 2
		self.obs_size = 2
		self.state_size = 4

	def reset(self):
		self.current_agent = self.agents[0]
		self.state2 = -1
		self.cycles = {agent:0 for agent in self.agents}
		self.coord = [-1,-1]

	def get_obs(self,agent):
		if self.cycle == 0:
			return np.array([self.cycle,0])
		elif self.cycle == 1:
			return np.array([self.cycle,self.state2])
		else:
			return None

	def get_state(self):
		state = []
		for agent in self.agents:
			state.append(self.get_obs(agent))
		return np.array(state).flatten()


	def last(self,agent=None):
		if agent == None:
			agent = self.current_agent
		obs = self.get_obs(agent)
		reward = self.reward(agent)
		done = self.is_done(agent)
		info = None
		return obs,reward,done,info

	def is_done(self,agent):
		if self.cycles[agent] > 1:
			return True
		return False

	def step(self,action):
		if self.is_done(self.current_agent):
			assert action==None, 'Action must be None when episode is done'
			return 
		if self.cycle==0:
			if self.current_agent == self.agents[0]:
				assert action in [0,1], 'Action must be 0 or 1'
				self.state2 = action

		elif self.cycle==1:
			idx = self.agents.index(self.current_agent)
			self.coord[idx] = action

		self.cycles[self.current_agent] += 1

	def reward(self,agent):
		if -1 in self.coord:
			return 0
		elif self.state2:
			matrix = np.array([[0,1],[1,8]])
			return matrix[tuple(self.coord)]
		else:
			return 7

	def agent_iter(self):
		done_agents = []
		while len(done_agents) != len(self.agents):
			for agent in self.agents:
				self.current_agent = agent
				if not self.is_done(self.current_agent):
					yield agent
				elif agent not in done_agents:
					done_agents.append(agent)
					yield agent

	@property
	def cycle(self):
		return self.cycles[self.current_agent]

	@property
	def dones(self):
		return {agent:self.is_done(agent=agent) for agent in self.agents}

if __name__ == '__main__':

	class RandomAgent:
		def __init__(self,n_actions):
			self.n_actions = n_actions

		def get_action(self,obs,done):
			if done:
				return None
			return random.randrange(self.n_actions)

	env = Environment()
	agent = RandomAgent(env.n_actions)

	for _ in range(3):
		env.reset()
		for agent_name in env.agent_iter():
			if agent_name == env.agents[0]:
				print(env.get_state())
			print(agent_name+': ',end='')
			obs,reward,done,_ = env.last()
			action = agent.get_action(obs,done)
			env.step(action)
			print(obs,reward,done,action)

		print('\n')
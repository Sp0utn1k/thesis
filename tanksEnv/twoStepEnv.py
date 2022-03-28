import numpy as np
import random

class Environment:

	def __init__(self):
		self.agents = ['agent0','agent1']
		self.n_actions = 2

	def reset(self):
		self.current_agent = self.agents[0]
		self.state2 = -1
		self.cycles = {agent:0 for agent in self.agents}
		self.coord = [-1,-1]

	def get_state(self):
		if self.cycle == 0:
			return [self.cycle,0]
		elif self.cycle == 1:
			return [self.cycle,self.state2]
		else:
			return None

	def last(self):
		obs = self.get_state()
		reward = self.reward()
		done = self.is_done()
		info = None
		return obs,reward,done,info

	def is_done(self,agent=None):
		if agent == None:
			agent = self.current_agent
		if self.cycles[agent] > 1:
			return True
		return False

	def step(self,action):
		if self.is_done():
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

	def reward(self):
		if -1 in self.coord:
			return 0
		elif self.state2:
			matrix = np.array([[0,1],[1,8]])
			return matrix[tuple(self.coord)]
		else:
			return 7

	def agent_iter(self):
		while False in self.dones.values():
			for agent in self.agents:
				self.current_agent = agent
				if not self.is_done():
					yield agent

	@property
	def cycle(self):
		return self.cycles[self.current_agent]

	@property
	def dones(self):
		return {agent:self.is_done(agent=agent) for agent in self.agents}

class RandomAgent:
	def __init__(self,n_actions):
		self.n_actions = n_actions

	def get_action(self,obs):
		return random.randrange(self.n_actions)


if __name__ == '__main__':

	env = Environment()
	env.reset()
	agent = RandomAgent(env.n_actions)


	for agent_name in env.agent_iter():
		print(agent_name)
		obs,reward,done,_ = env.last()
		print(obs,reward,done)
		action = agent.get_action(obs)
		env.step(action)
		print(action)

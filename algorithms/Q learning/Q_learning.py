import numpy as np
import os
import time
import pandas as pd

SIZE = [3,10]
ACTIONS = ['up','down','left','right']
N_epsiodes = 100

class Environment:

	def __init__(self,size = [4,12]):
		self.size = size
		self.pos = (0,0)
		self.cliff = [(0,i) for i in range(1,size[1]-1)]
		self.goal = (0,size[1]-1)

	def reset(self):
		self.pos = (0,0)

	def step(self,A):
		is_done = False
		i,j = self.pos

		if A == 'up':
			i = min(i+1,self.size[0]-1)
		elif A == 'down':
			i = max(i-1,0)
		elif A == 'left':
			j = max(j-1,0)
		elif A == 'right':
			j = min(j+1,self.size[1]-1)
		else:
			print('Error: unvalid action')
			time.sleep(1)
			return

		self.pos = (i,j)
		if self.pos == self.goal or self.pos in self.cliff:
			is_done = True

		return (self.reward(),self.pos,is_done)

	def reward(self):
		if self.pos == self.goal:
			return 0.0
		elif self.pos in self.cliff:
			return -1.0
		else:
			return -.1

	def display(self,twait=0.1):
		os.system('clear')
		for i in range(self.size[0]):
			line = ''
			for j in range(self.size[1]):
				line += f' {self.tile(self.size[0]-i-1,j)} '
			print(line)
		time.sleep(twait)

	def tile(self,i,j):
		if (i,j) == self.pos:
			return 'o'
		elif (i,j) in self.cliff:
			return 'X'
		elif (i,j) == self.goal:
			return 'G'
		else:
			return '.'

class Qlearning:

  def __init__(self,actions,eps=0.1,alpha=0.1,gamma=0.9):
    self.gamma = gamma
    self.alpha = alpha
    self.eps = eps
    self.actions = actions
    self.action_space = len(actions)
    self.actions = actions
    self.table = pd.DataFrame(data=np.zeros((self.action_space,0),dtype=np.float_), index=actions, columns=None,dtype=np.float_)

  def eps_greedy(self,S):
    agent.table = agent.table.sample(frac=1)
    if S not in self.table.columns:
      self.get_Q(S,self.actions[0])
    if np.random.rand() < self.eps:
      A = np.random.choice(self.actions)
    else:
      A = agent.table[S].idxmax()
    return A

  def get_Q(self,S,A):
    if S not in self.table.columns:
      self.table.insert(0,S,np.zeros((self.action_space,1),dtype=np.float_))
    print(self.table)
    return self.table.at[A,S]

  def learn(self,S,A,R,S_,is_done=False):
    if S_ not in self.table.columns:
      self.get_Q(S_,self.actions[0])
    
    if not is_done:
      new_Q = R+self.gamma*self.table[S_].max()
    else:
      new_Q = R
    self.table.at[A,S] = (1-self.alpha)*self.get_Q(S,A) + self.alpha*(new_Q)

  def setup_epsilon_decay(self,start=0,stop=100,limits=[1,0.02]):
    self.eps = limits[0]
    self.eps_params = {
      'start' : start,
      'stop' : stop,
      'limits' : limits
    }


if __name__ == "__main__":

	twait = 0.05
	env = Environment(size=SIZE)
	agent = Qlearning(actions=ACTIONS,eps=0.1,alpha=0.25)

	env.display(twait=10)
	
	agent.get_Q((0,1),'up')
	agent.table.at['down',(0,1)] = -.1



	for i in range(N_epsiodes):
		env.reset()
		is_done = False
		while not is_done:
			S = env.pos
			A = agent.eps_greedy(S)
			(R,S_,is_done) = env.step(A)
			agent.learn(S,A,R,S_)

	for i in range(10):
		env.reset()
		env.display(twait=twait)
		is_done = False
		while not is_done:
			S = env.pos
			A = agent.eps_greedy(S)
			(R,S_,is_done) = env.step(A)
			env.display(twait=twait)
			agent.learn(S,A,R,S_,is_done=is_done)
		time.sleep(.5)

	print(agent.table)
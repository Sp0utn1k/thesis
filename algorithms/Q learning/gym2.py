import numpy as np
import gym
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

class Qlearning:

  def __init__(self,actions,eps=0.1,alpha=0.1,gamma=0.9):
    self.gamma = gamma
    self.alpha = alpha
    self.eps = eps
    self.actions = actions
    self.action_space = len(actions)
    self.actions = actions
    self.table = pd.DataFrame(data=np.zeros((self.action_space,0),dtype=np.float), index=actions, columns=None,dtype=np.float)

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
      self.table.insert(0,S,np.zeros((self.action_space,1),dtype=np.float))
    return self.table.at[A,S]

  def learn(self,S,A,R,S_,is_done=False):
    if S_ not in self.table.columns:
      self.get_Q(S_,self.actions[0])
    
    if not is_done:
      new_Q = R+self.gamma*self.table[S_].max()
    else:
      new_Q = R
    self.table.at[A,S] = (1.0-self.alpha)*self.get_Q(S,A) + self.alpha*new_Q

  def setup_epsilon_decay(self,start=0,stop=100,limits=[1,0.02]):
    self.eps = limits[0]
    self.eps_params = {
      'start' : start,
      'stop' : stop,
      'limits' : limits
    }

  def epsilon_decay(self,step):
    start = self.eps_params['start']
    stop = self.eps_params['stop']
    limits = self.eps_params['limits']
    r = (step-start)/(stop-start)
    self.eps = round(limits[0] + min(1,max(0,r))*(limits[1]-limits[0]),2)

def process_obs(obs):
  pos = int(4.0*obs[0])
  speed = int(4.0*obs[1])
  angle = int(100.0*obs[2])
  omega = int(100.0*obs[3])
  return (pos,speed,angle,omega)

if __name__ == "__main__":

  N_episodes = 30000
  plot_points = 300

  episodes_per_plot = int(round(N_episodes/plot_points))

  env = gym.make("CartPole-v1")
  actions = range(env.action_space.n)
  agent = Qlearning(actions,alpha=.5,gamma=1.0)
  image_name = 'eps0.05_a0.5(adaptative).png'

  # agent.setup_epsilon_decay(start=1,stop=N_episodes,limits=[.05,0.01])
  agent.eps = .05

  # sys.exit()

  rewards = np.zeros((episodes_per_plot,1))
  plot_reward = np.array([])
  plot_std = np.array([])

  for episode in range(N_episodes):
    # agent.epsilon_decay(episode)
    S = process_obs(env.reset())
    is_done = False
    R_tot = 0
    while not is_done:
      A = agent.eps_greedy(S)
      S_,R,is_done,_ = env.step(A)
      R_tot += R
      if R_tot == 200:
        is_done = True
      S_ = process_obs(S_)
      agent.learn(S,A,R,S_,is_done=is_done)
      S = S_

      # if np.std(rewards) > 50:
      #   agent.alpha *= 0.9999
      #   agent.alpha = min(1,agent.alpha)
      
    if episode % episodes_per_plot == 0:
      plot_reward = np.append(plot_reward,round(np.mean(rewards),1))
      plot_std = np.append(plot_std,np.std(rewards))
      print(f'Episode {episode}, alpha = {agent.alpha:.3e}, reward = {plot_reward[-1]}, std = {plot_std[-1]:.1f}')
      t = np.linspace(1,episode,num=len(plot_reward)-1)
      plt.clf()
      plt.plot(t,plot_reward[1:])
      plt.plot(t,plot_std[1:])
      plt.savefig(image_name)

    rewards[episode%episodes_per_plot] = R_tot

  input('Press enter to show result.')
  for _ in range(5):
    S = process_obs(env.reset())
    env.render()
    is_done = False
    while not is_done:
      A = agent.eps_greedy(S)
      S_,R,is_done,_ = env.step(A)
      S_ = process_obs(S_)
      agent.learn(S,A,R,S_,is_done=is_done)
      S = S_
      env.render()

  env.close()
  print(agent.table.to_numpy().shape)
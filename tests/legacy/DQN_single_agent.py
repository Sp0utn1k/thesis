import sys
sys.path.insert(1, '../Q learning/DQN')
from env import *
import DQN

from setup import args
env = Environment(args)
n_actions = len(env.actions)
agent = DQN.Agent(n_states=env.obs_size,n_actions=n_actions,hidden_layers=[512,256])
# print(agent.net.pipe)


for idx in env.agent_iter():
	if idx == 0:
		A = env.get_random_action()
	env.step(A,prompt_action=True)
	# env.show_fpv(0,twait=500)
	env.render(twait=500)
import yaml
from tanksEnv import tanksEnv
from tanksEnv.utils.networks import RNNetwork, FCNetwork
from tanksEnv.algorithms.DQN import Agent, DQNRunner 
from tensorboardX import SummaryWriter
import time
import sys

DQN_mode = sys.argv[1]
assert DQN_mode in ['classical','RNN','RNN_encoder','encoder'], 'Unknown DQN mode.'


if __name__ == "__main__":
	with open(f'./configs/{DQN_mode}.yml','r') as file:
		settings = yaml.safe_load(file)

	env = tanksEnv.Environment(**settings)
	S = env.reset()
	obs_size = len(S)
	n_actions = env.n_actions

	if 'RNN' in DQN_mode:
		agent_net = RNNetwork(obs_size,n_actions,**settings)
	else:
		agent_net = FCNetwork(obs_size,n_actions,**settings)
	agent = Agent(network=agent_net,**settings)
	runner = DQNRunner(env,agent,**settings)

	timestr = time.strftime('%Y_%m_%d-%Hh%M')
	writer = SummaryWriter(f'runs/{DQN_mode}/'+timestr)
	
	for (episode,episode_length,reward,loss) in runner.run(settings['n_episodes'],render=settings.get('render',False)):
		writer.add_scalar('reward',reward,episode)
		if loss != None:
			writer.add_scalar('loss',loss,episode)
	writer.close()

	input('Press enter to show demo')
	for (episode,episode_length,reward,loss) in runner.run(10,render=True,train=False):
		print(f'Reward: {reward}')
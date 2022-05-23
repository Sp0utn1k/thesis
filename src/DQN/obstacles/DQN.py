import yaml
from tanksEnv import tanksEnv
from tanksEnv.utils.networks import RNNetwork, FCNetwork
from tanksEnv.algorithms.DQN import Agent, DQNRunner 
from tensorboardX import SummaryWriter
import time
import os, sys
import cProfile

def main():

	DQN_mode = sys.argv[1]
	assert DQN_mode in ['classical','RNN','RNN_encoder','encoder'], 'Unknown DQN mode.'
	if len(sys.argv) > 2:
		n_red = int(sys.argv[2])
	else:
		n_red = 1

	with open(f'./configs/{DQN_mode}.yml','r') as file:
		settings = yaml.safe_load(file)

	settings['agents_description'] = [{'pos0': 'random','team': 'blue'},
								{'pos0': 'random','team': 'red','policy': 'random', 'replicas':n_red}
				] 

	if n_red > 2:
		settings['n_episodes'] *= 2
		settings['epsilon_decay']['period'] *= 2
		
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
	if settings['random_obstacles']:
		writer = SummaryWriter(f'runs_randobs/{DQN_mode}{n_red}/'+timestr)
	elif settings['random_ids']:
		writer = SummaryWriter(f'runs_randid/{DQN_mode}{n_red}/'+timestr)
	else:
		writer = SummaryWriter(f'runs/{DQN_mode}{n_red}/'+timestr)
	
	for (episode,episode_length,reward,loss) in runner.run(settings['n_episodes'],render=settings.get('render',False)):
		writer.add_scalar('reward',reward,episode)
		if loss != None:
			writer.add_scalar('loss',loss,episode)
	writer.close()
	
	rewards = []
	for (episode,episode_length,reward,loss) in runner.run(100,render=False,train=False):
		rewards.append(reward)

	filename = os.path.join('test_data',DQN_mode)
	if settings['random_obstacles']:
		filename += '_randobs'
	elif settings['random_ids']:
		filename += '_randid'
	with open(filename, 'a') as f:
		f.write(str(rewards))
		f.write(';...\n')

	# input('Press enter to show demo')
	# for (episode,episode_length,reward,loss) in runner.run(5,render=True,train=False):
	# 	print(f'Reward: {reward}')

if __name__ == "__main__":
	# cProfile.run('main()')
	for _ in range(3):
		main()
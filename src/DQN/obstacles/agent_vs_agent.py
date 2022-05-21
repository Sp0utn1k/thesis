import yaml
from tanksEnv import tanksEnv
from tanksEnv.utils.networks import RNNetwork, FCNetwork
from tanksEnv.algorithms.DQN import Agent, DQNRunner 
from tensorboardX import SummaryWriter
import time
import sys
import cProfile

def main():

	DQN_mode1 = sys.argv[1]
	assert DQN_mode1 in ['classical','RNN','RNN_encoder','encoder'], 'Unknown DQN mode.'
	DQN_mode2 = sys.argv[2]
	assert DQN_mode2 in ['classical','RNN','RNN_encoder','encoder'], 'Unknown DQN mode.'

	with open(f'./configs/{DQN_mode1}.yml','r') as file:
		settings1 = yaml.safe_load(file)
	with open(f'./configs/{DQN_mode1}.yml','r') as file:
		settings2 = yaml.safe_load(file)
	with open(f'./configs/agent_vs_agent.yml','r') as file:
		main_settings = yaml.safe_load(file)

	settings1['agents_description'] = [{'pos0': 'random','team': 'blue'},
				{'pos0': 'random','team': 'red','policy': 'smart_agent'}] 

	env = tanksEnv.Environment(**settings1)
	S = env.reset()
	obs_size = len(S)
	n_actions = env.n_actions

	if 'RNN' in DQN_mode1:
		agent_net1 = RNNetwork(obs_size,n_actions,**settings1)
	else:
		agent_net1 = FCNetwork(obs_size,n_actions,**settings1)
	agent1 = Agent(network=agent_net1,**settings1)
	runner1 = DQNRunner(env,agent1,**settings1)


	if 'RNN' in DQN_mode1:
		agent_net2 = RNNetwork(obs_size,n_actions,**settings2)
	else:
		agent_net2 = FCNetwork(obs_size,n_actions,**settings2)
	agent2 = Agent(network=agent_net2,**settings2)
	runner2 = DQNRunner(env,agent2,**settings2)

	timestr = time.strftime('%Y_%m_%d-%Hh%M')
	writer = SummaryWriter(f'agents_vs_agents_runs/{DQN_mode1}_vs_{DQN_mode2}/'+timestr)
	

	swap_freq = main_settings['swap_freq']
	N_runs = settings1['n_episodes']//swap_freq

	for run_x in range(N_runs):
		env.red_agent = agent2
		env.observation_mode = settings1['observation_mode']
		env.red_observation_mode = settings2['observation_mode']
		# agent2.epsilon = 0
		agent2.net.eval()
		agent1.net.train()

		for (episode,episode_length,reward,loss,red_reward) in runner1.run(swap_freq,render=settings1.get('render',False),
			start_episode=run_x*swap_freq,get_red_reward=True):
			corrected_episode = run_x*swap_freq+episode
			writer.add_scalar('reward1',reward,corrected_episode)
			writer.add_scalar('reward2',red_reward,corrected_episode)
			if loss != None:
				writer.add_scalar('loss1',loss,corrected_episode)

		env.red_agent = agent1
		env.observation_mode = settings2['observation_mode']
		env.red_observation_mode = settings1['observation_mode']
		# agent1.epsilon = 0
		agent1.net.eval()
		agent2.net.train()

		for (episode,episode_length,reward,loss,red_reward) in runner2.run(swap_freq,render=settings2.get('render',False),
			start_episode=run_x*swap_freq,get_red_reward=True):
			corrected_episode = (run_x+1)*swap_freq+episode
			writer.add_scalar('reward2',reward,corrected_episode)
			writer.add_scalar('reward1',red_reward,corrected_episode)
			if loss != None:
				writer.add_scalar('loss2',loss,corrected_episode)


	
	writer.close()

	env.red_agent = agent2
	env.observation_mode = settings1['observation_mode']
	env.red_observation_mode = settings2['observation_mode']
	agent1.epsilon = 0
	agent2.epsilon = 0
	agent2.net.eval()
	agent1.net.eval()

	input('Press enter to show demo1')
	for (episode,episode_length,reward,loss) in runner1.run(5,render=True,train=False):
		print(f'Reward: {reward}')

	env.red_agent = agent1
	env.observation_mode = settings2['observation_mode']
	env.red_observation_mode = settings1['observation_mode']

	input('Press enter to show demo2')
	for (episode,episode_length,reward,loss) in runner2.run(5,render=True,train=False):
		print(f'Reward: {reward}')


if __name__ == "__main__":
	cProfile.run('main()')
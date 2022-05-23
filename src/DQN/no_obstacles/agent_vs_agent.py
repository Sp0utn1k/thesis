import yaml
from tanksEnv import tanksEnv
from tanksEnv.utils.networks import RNNetwork, FCNetwork
from tanksEnv.algorithms.DQN import Agent, DQNRunner 
from tensorboardX import SummaryWriter
import time
import os,sys
import cProfile

def main():

	DQN_mode1 = sys.argv[1]
	assert DQN_mode1 in ['classical','RNN','RNN_encoder','encoder'], 'Unknown DQN mode.'
	DQN_mode2 = sys.argv[2]
	assert DQN_mode2 in ['classical','RNN','RNN_encoder','encoder'], 'Unknown DQN mode.'

	with open(f'./configs/{DQN_mode1}.yml','r') as file:
		settings1 = yaml.safe_load(file)
	with open(f'./configs/{DQN_mode2}.yml','r') as file:
		settings2 = yaml.safe_load(file)
	with open(f'./configs/agent_vs_agent.yml','r') as file:
		main_settings = yaml.safe_load(file)


	for key in settings1.keys():
		settings1[key] = main_settings.get(key,settings1[key])

	for key in settings2.keys():
		settings2[key] = main_settings.get(key,settings2[key])

	for key,val in main_settings.items():
		settings1[key] = val
		settings2[key] = val

	env1 = tanksEnv.Environment(**settings1)
	S = env1.reset()
	obs_size = len(S)
	n_actions = env1.n_actions

	if 'RNN' in DQN_mode1:
		agent_net1 = RNNetwork(obs_size,n_actions,**settings1)
	else:
		agent_net1 = FCNetwork(obs_size,n_actions,**settings1)
	agent1 = Agent(network=agent_net1,**settings1)
	runner1 = DQNRunner(env1,agent1,**settings1)


	env2 = tanksEnv.Environment(**settings2)
	S = env2.reset()
	obs_size = len(S)
	n_actions = env2.n_actions
	if 'RNN' in DQN_mode2:
		agent_net2 = RNNetwork(obs_size,n_actions,**settings2)
	else:
		agent_net2 = FCNetwork(obs_size,n_actions,**settings2)
	agent2 = Agent(network=agent_net2,**settings2)
	runner2 = DQNRunner(env2,agent2,**settings2)

	env1.red_agent = agent2
	env1.red_observation_mode = settings2['observation_mode']
	env2.red_agent = agent1
	env2.red_observation_mode = settings1['observation_mode']

	timestr = time.strftime('%Y_%m_%d-%Hh%M')
	writer = SummaryWriter(f'agents_vs_agents_runs/{DQN_mode1}_vs_{DQN_mode2}/'+timestr)
	
	swap_freq = main_settings['swap_freq']
	N_runs = settings1['n_episodes']//swap_freq

	for run_x in range(N_runs):

		agent2.net.eval()
		agent1.net.train()

		for (episode,episode_length,reward,loss,red_reward) in runner1.run(swap_freq,render=settings1.get('render',False),
			start_episode=run_x*swap_freq,get_red_reward=True):
			corrected_episode = run_x*swap_freq+episode
			writer.add_scalar('reward1',reward,corrected_episode)
			writer.add_scalar('reward2',red_reward,corrected_episode)
			if loss != None:
				writer.add_scalar('loss1',loss,corrected_episode)

		
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

	
	agent2.net.eval()
	agent1.net.eval()

	rewards1 = []
	for (episode,episode_length,reward,loss) in runner1.run(100,render=False,train=False):
		rewards1.append(reward)

	rewards2 = []
	for (episode,episode_length,reward,loss) in runner2.run(100,render=False,train=False):
		rewards2.append(reward)

	filename = os.path.join('test_data',f'{DQN_mode1}_vs_{DQN_mode2}')
	with open(filename, 'a') as f:
		f.write(str(rewards1))
		f.write('\n')
		f.write(str(rewards2))
		f.write('\n\n')

	input('Press enter to show demo1')
	for (episode,episode_length,reward,loss) in runner1.run(5,render=True,train=False):
		print(f'Reward: {reward}')

	input('Press enter to show demo2')
	for (episode,episode_length,reward,loss) in runner2.run(5,render=True,train=False):
		print(f'Reward: {reward}')


if __name__ == "__main__":
	cProfile.run('main()')
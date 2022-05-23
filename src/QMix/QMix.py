from tanksEnv import tanksEnv
from tanksEnv.algorithms import QMix
from tanksEnv.utils.networks import FCNetwork,RNNetwork
from tensorboardX import SummaryWriter
import yaml
import os,sys
import cProfile
import time

def main():
	
	##########  ENVIRONMENT SETUP  ########## 
	with open(f'./configs/environment.yml','r') as file:
		env_params= yaml.safe_load(file)
	env = tanksEnv.Environment(**env_params)
	n_actions = env.n_actions

	##########  LEARNING SETUP  ##########
	config_id = sys.argv[1]
	with open(f'./configs/config{config_id}.yml','r') as file:
		config= yaml.safe_load(file)

	if config.get('RNN',False):
		agent_net = RNNetwork(env.obs_size,n_actions,**config)
	else:
		agent_net = FCNetwork(env.obs_size,n_actions,**config)

	qmix = QMix.QMix(env,agent_net,**config)
	runner = QMix.QMixRunner(env,qmix,**config)

	timestr = time.strftime('%Y_%m_%d-%Hh%M')
	if config['use_mixer']:
		writer_name = 'QMix'
	else:
		writer_name = 'VDN'

	if qmix.rnn:
		writer_name += '_RNN'
	else:
		writer_name += '_FC'

	if config['observation_mode'] == 'encoded':
		writer_name += '_encoder'

	use_writer = config.get('use_writer',False)
	if use_writer:
		writer = SummaryWriter(os.path.join('runs',writer_name,timestr))
	for episode,episode_length,reward,loss in runner.run(config['n_episodes']):
		if use_writer:
			writer.add_scalar('Reward',reward,episode)
			writer.add_scalar('N_cycles',episode_length,episode)
			if loss != None:
				writer.add_scalar('Loss',loss,episode)

	env.reset()

	rewards = []
	for (episode,episode_length,reward,loss) in runner.run(100,render=False,train=False):
		rewards.append(reward)

	filename = os.path.join('test_data',writer_name)
	with open(filename, 'a') as f:
		f.write(str(rewards))
		f.write('\n;\n')

	env.reset()
	input('Press enter to show demo')
	for episode,episode_length,reward,loss in runner.run(5,render=True,train=False):
		print('Reward ',reward)
	writer.close()

if __name__ == '__main__':
	cProfile.run('main()')
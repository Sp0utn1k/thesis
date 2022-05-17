from tanksEnv import tanksEnv
from tanksEnv.algorithms import QMix
from tanksEnv.utils.networks import FCNetwork,RNNetwork
from tensorboardX import SummaryWriter
import yaml
import sys
import cProfile

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
	agent_net = FCNetwork(env.obs_size,n_actions,**config)
	qmix = QMix.QMix(env,agent_net,**config)
	runner = QMix.QMixRunner(env,qmix,**config)

	use_writer = config.get('use_writer',False)
	if use_writer:
		writer = SummaryWriter()
	for episode,episode_length,reward,loss in runner.run(config['n_episodes']):
		if use_writer:
			writer.add_scalar('Reward',reward,episode)
			writer.add_scalar('N_cycles',episode_length,episode)
			if loss != None:
				writer.add_scalar('Loss',loss,episode)

	input('Press enter to show demo')
	for episode,episode_length,reward,loss in runner.run(10,render=True,train=False):
		pass
	writer.close()

if __name__ == '__main__':
	cProfile.run('main()')
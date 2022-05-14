from tanksEnv import tanksEnv
from tanksEnv.algorithms import QMix
from tanksEnv.utils.networks import FCNetwork,RNNetwork
from tensorboardX import SummaryWriter
import yaml
import sys


if __name__ == '__main__':
	
	##########  ENVIRONMENT SETUP  ########## 
	with open(f'./configs/environment.yml','r') as file:
		env_params= yaml.safe_load(file)
	env = tanksEnv.Environment(**env_params)
	S = env.reset()
	obs_size = len(S)
	n_actions = env.n_actions


	##########  LEARNING SETUP  ##########
	config_id = sys.argv[1]
	with open(f'./configs/config{config_id}.yml','r') as file:
		config= yaml.safe_load(file)
	agent_net = FCNetwork(obs_size,n_actions,**config)
	qmix = QMix.QMix(env,agent_net,**config)
	runner = QMix.QMixRunner(env,qmix,**config)
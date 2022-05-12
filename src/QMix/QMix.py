from tanksEnv import twoStepEnv
from tanksEnv.algorithms import QMix
from tanksEnv.utils.networks import FCNetwork,RNNetwork
from tensorboardX import SummaryWriter
import torch


if __name__ == '__main__':
	
	with open(f'./configs/env.yml','r') as file:
		env_params= yaml.safe_load(file)
from tanksEnv.utils.networks import RNNetwork, DecoderNN
from tanksEnv import tanksEnv
import yaml
import torch
from torch import nn,optim

with open(f'configs/config1.yml','r') as file:
    config = yaml.safe_load(file)

encoder = RNNetwork(**config)
env = tanksEnv.Environment(**config)
print(env.get_observation())
env.render_fpv(0,twait=0)
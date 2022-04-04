from tanksEnv.utils.networks import RNNetwork, DecoderNN
from tanksEnv import tanksEnv
import yaml
from math import sqrt
import torch
from torch import nn,optim

with open(f'configs/config1.yml','r') as file:
    config = yaml.safe_load(file)

encoder = RNNetwork(**config)

# env = tanksEnv.Environment(**config)
# print(env.get_observation())
# env.render_fpv(0,twait=0)

visibility = 20
vis_grid = []
for x in range(1,visibility+1):
    for y in range(1,visibility+1):
        if sqrt(x**2+y**2) <= visibility:
            vis_grid += [[x,y],[-x,y],[x,-y],[-x,-y]]
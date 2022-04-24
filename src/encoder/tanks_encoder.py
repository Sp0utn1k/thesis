from tanksEnv.utils.networks import RNNetwork, DecoderNN
from tanksEnv import tanksEnv
import yaml
import math
import torch
import random
from torch import nn,optim

def generate_dummy_foe(vis_grid,idx=0):

    if isinstance(idx,int):
        assert idx>-1, "id must be positive or 0."
    else:
        assert isinstance(idx,list) and all(list(map(lambda x: isinstance(x,int), idx))), "id must be an integer or a list of intigers."
        idx = random.choice(idx)

    [x,y] = random.choice(vis_grid)
    return [x,y,idx]

def generate_vis_grid(size):
    visibility=size
    vis_grid = []
    for x in range(0,visibility+1):
        for y in range(0,visibility+1):
            if [x,y] != [0,0] and math.sqrt(x**2+y**2) <= visibility:
                vis_grid.append([x,y])
                if x:
                    vis_grid.append([-x,y])
                    if y:
                        vis_grid.append([-x,-y])
                if y:
                    vis_grid.append([x,-y])
    return vis_grid

def generate_sample(vis_grid,max_length,max_id,device='cpu'):
    observation = []
    positions = []
    length = random.randrange(max_length) + 1
    for _ in range(length):
        foe = generate_dummy_foe(vis_grid,idx=list(range(max_id+1)))
        observation.append(foe)
        positions.append(foe[:2])
    observation = torch.tensor(observation,dtype=torch.float32,device=device)


    positive = random.choice(observation)
    empty_tiles = [pos for pos in vis_grid if pos not in positions]
    negative = random.choice(empty_tiles)
    negative.append(random.randrange(max_id+1))
    negative = torch.tensor(negative,dtype=torch.float32,device=device)

    return observation,positive,negative

def generate_batch(vis_grid,max_length,max_id,batch_size,device='cpu'):
    observations = []
    positives = []
    negatives = []
    for _ in range(batch_size):
        observation,positive,negative = generate_sample(vis_grid,max_length,max_id,device=device)
        observations.append(observation)
        positives.append(positive)
        negatives.append(negative)
    positives = torch.cat(positives)
    negatives = torch.cat(negatives)
    return observations, positives, negatives

if __name__ == '__main__':
    with open(f'configs/config1.yml','r') as file:
        config = yaml.safe_load(file)

    visibility = config['visibility']
    max_id = config['max_id']
    max_foes = config['max_foes']
    batch_size = config['batch_size']

    vis_grid = generate_vis_grid(visibility)

    encoder = RNNetwork(**config)
    observations,positives,negatives = generate_batch(vis_grid,max_foes,max_id,batch_size)
    print(observations)
    output,_ = encoder(observations)
    print(output)
    
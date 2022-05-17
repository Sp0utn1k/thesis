import random
import math
import torch

def generate_obs_data(visibility):
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


def generate_tanks_data(visibility,max_id):
    vis_grid = []
    for x in range(0,visibility+1):
        for y in range(0,visibility+1):
        	for idx in range(max_id+1):
	            if [x,y] != [0,0] and math.sqrt(x**2+y**2) <= visibility:
	                vis_grid.append([x,y,idx])
	                if x:
	                    vis_grid.append([-x,y,idx])
	                    if y:
	                        vis_grid.append([-x,-y,idx])
	                if y:
	                    vis_grid.append([x,-y,idx])
    return vis_grid


def generate_seq(data,max_length,device='cpu',length='random'):

    if length == 'random':
        length = random.randrange(max_length+1)

    if length:
        positives = random.choices(list(range(len(data))),k=length)
        seq = [data[pos] for pos in positives]
    else:
        seq = [0,0,-100,-100][:len(data[0])]
        seq = [seq]
        positives = []
    
    positions = [tank[:2] for tank in seq]
    if len(seq) != len(unique(seq)):
        return generate_seq(data,max_length,device=device,length=length)

    return seq, positives


def generate_batch(batch_size,data,max_length,device='cpu'):

    batch = []
    positives = []
    for _ in range(batch_size):
        seq,positive = generate_seq(data,max_length)
        batch.append(torch.tensor(seq,device=device,dtype=torch.float32))
        positives += positive

    positives = torch.tensor(positives,device=device,dtype=torch.uint8)
    return batch, positives

def unique(data):
    output = []
    for x in data:
        if x not in output:
            output.append(x)
    return output
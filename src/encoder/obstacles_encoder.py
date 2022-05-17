from tanksEnv.utils.networks import RNNetwork, DecoderNN
import yaml
import math
import random
import torch
from torch import nn,optim
from tensorboardX import SummaryWriter
import time
import sys
import pickle
import os
import cProfile
import copy 
import numpy as np
import cv2 as cv

def generate_dummy_obstacle(vis_grid):

    if isinstance(idx,int):
        assert idx>-1, "id must be positive or 0."
    else:
        assert isinstance(idx,list) and all(list(map(lambda x: isinstance(x,int), idx))), "id must be an integer or a list of integers."
        idx = random.choice(idx)

    [x,y] = random.choice(vis_grid)
    return [x,y,idx]

def generate_vis_grid(size,obs_len=2):
    visibility=size
    vis_grid = []
    for x in range(0,visibility+1):
        for y in range(0,visibility+1):
            x2 = x**2
            y2 = y**2
            x3 = 2*x*math.pi/20
            y3 = 2*y*math.pi/20
            if [x,y] != [0,0] and math.sqrt(x**2+y**2) <= visibility:
                vis_grid.append([x,y,x2,y2,x*y,math.sin(x3),math.sin(y3),
                    math.sin(2*x3),math.sin(2*y3),math.sin(3*x3),math.sin(3*y3)][:obs_len])
                if x:
                    vis_grid.append([-x,y,x2,y2,-x*y,-math.sin(x3),math.sin(y3),
                        -math.sin(2*x3),math.sin(2*y3),math.sin(3*x3),math.sin(3*y3)][:obs_len])
                    if y:
                        vis_grid.append([-x,-y,x2,y2,x*y,-math.sin(x3),-math.sin(y3),
                            -math.sin(2*x3),-math.sin(2*y3),math.sin(3*x3),math.sin(3*y3)][:obs_len])
                if y:
                    vis_grid.append([x,-y,x2,y2,-x*y,math.sin(x3),-math.sin(y3),
                        math.sin(2*x3),-math.sin(2*y3),math.sin(3*x3),math.sin(3*y3)][:obs_len])
    return vis_grid

def generate_sample(vis_grid,hidden_dict,max_length,device='cpu'):

    observation = []
    length = random.randrange(max_length+1)

    observation = random.sample(vis_grid,length)
    # grid = vis_grid
    # for _ in range(length):
    #     # prev_grid = copy.deepcopy(grid)
    #     prev_grid = grid

    #     obs = random.choice(grid)
    #     x,y = obs
    #     observation.append(obs)
        
    #     hiddens = [obs]
    #     hiddens += hidden_dict[x,y]

    #     grid = [pos for pos in grid if pos not in hiddens]

    #     if not grid:
    #         grid = prev_grid
    #         observation.pop()
    #         break


    length = len(observation)
    if length == 0:
        observation.append([0,0,0,0,0,0,0,0,0])
        positive = random.choice(vis_grid)
        no_obstacle = True
    else:
        positive = random.choice(observation)
        no_obstacle = False

    # negative = random.choice(grid)
    negative = observation[0]
    while negative  in observation:
        negative = random.choice(vis_grid)

    positive = torch.tensor(positive,dtype=torch.float32,device=device)
    negative = torch.tensor(negative,dtype=torch.float32,device=device)
    observation = torch.tensor(observation,dtype=torch.float32,device=device)

    return observation,positive,negative,no_obstacle

def generate_batch(vis_grid,hidden_dict,max_length,batch_size,device='cpu'):
    observations = []
    positives = []
    negatives = []
    no_obstacles_mask = []
    for _ in range(batch_size):
        observation,positive,negative,no_obstacle = generate_sample(vis_grid,hidden_dict,max_length,device=device)
        observations.append(observation)
        positives.append(positive)
        negatives.append(negative)
        no_obstacles_mask.append(no_obstacle)

    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    no_obstacles_mask = torch.BoolTensor(no_obstacles_mask)
    return observations, positives, negatives, no_obstacles_mask

def main():

    filename = os.path.join(os.getcwd(),'hidden_cells20.pkl')
    with open(filename, 'rb') as file:
        hidden_dict = pickle.load(file)


    config_id = sys.argv[1]
    config_id = f'obs{config_id}'

    print(f'Config: {config_id}.yml')
    with open(f'configs/{config_id}.yml','r') as file:
        config = yaml.safe_load(file)


    netfile = f'nets/{config_id}/'
    device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu',False) else "cpu")
    print(f'Device: {device}')

    visibility = config['visibility']
    max_obs = config['max_obs']
    batch_size = config['batch_size']
    epochs = config['epochs']
    use_writer = config.get('use_writer',False)
    learning_rate = config['learning_rate']
    save_model = config['save_model']
    load_model = config['load_model']
    save_period = config['save_period']

    obs_length = config['input_size']

    if load_model:
        encoder = torch.load(netfile+'encoder.pk')
        decoder = torch.load(netfile+'decoder.pk')
    else:
        encoder = RNNetwork(**config)
        decoder = DecoderNN(config['output_size']+obs_length,hidden_layers=config['decoder_hidden_layers'])
    encoder = encoder.to(device)
    decoder = decoder.to(device)


    if not load_model and save_model:
        input(f'A new model will be created and saved in {netfile}.\nPress enter to continue.')
    
    vis_grid = generate_vis_grid(visibility,obs_len=obs_length)
    loss_f = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=learning_rate)
    if use_writer:
        timestr = time.strftime('/%Y_%m_%d-%Hh%M')
        writer = SummaryWriter('runs/'+config_id+timestr)

    for epoch in range(epochs):

        observations,positives,negatives,no_obstacles_mask = generate_batch(vis_grid,hidden_dict,max_obs,batch_size,device=device)
        output,_ = encoder(observations)
        pos_pred = decoder(output,positives)
        neg_pred = decoder(output,negatives)
        pos_labels = torch.ones_like(pos_pred)
        pos_labels[no_obstacles_mask] = 0
        neg_labels = torch.zeros_like(neg_pred)
        pred = torch.cat([pos_pred,neg_pred])
        labels = torch.cat([pos_labels,neg_labels])

        optimizer.zero_grad()
        loss = loss_f(pred,labels)
        loss.backward()
        optimizer.step()

        if use_writer:
            pos_pred = list(decoder.predict(output,positives).detach().cpu().flatten().numpy())
            neg_pred = list(decoder.predict(output,negatives).detach().cpu().flatten().numpy())
            pred = pos_pred+neg_pred
            labels  = list(labels.detach().cpu().flatten().numpy())

            true_pos = sum(map(lambda x: x[0] and x[1],zip(pred,labels)))
            false_pos = sum(map(lambda x: x[0] and not x[1],zip(pred,labels)))
            true_neg = sum(map(lambda x: not x[0] and not x[1],zip(pred,labels)))
            false_neg = sum(map(lambda x: not x[0] and x[1],zip(pred,labels)))

            accuracy = (true_pos+true_neg)/(2*batch_size)
            recall = true_pos/(true_pos+false_neg)

            writer.add_scalar('Loss',loss.item(),epoch)
            writer.add_scalar('Accuracy',accuracy,epoch)
            writer.add_scalar('Recall',recall,epoch)

        if save_model and (epoch+1) % save_period == 0:
            torch.save(encoder,netfile+'encoder.pk')
            torch.save(decoder,netfile+'decoder.pk')


    if use_writer:
        writer.close()


def test():

    TN = [0,50,0]
    FP = [0,0,50]
    TP = [0,255,0]
    FN = [0,0,255]

    config_id = sys.argv[1]
    config_id = f'obs{config_id}'
    print(f'Config: {config_id}.yml')
    with open(f'configs/{config_id}.yml','r') as file:
        config = yaml.safe_load(file)

    netfile = f'nets/{config_id}/'
    device = 'cpu'
    print(f'Device: {device}')

    visibility = config['visibility']
    max_obs = config['max_obs']
    input_size = config['input_size']

    vis_grid = generate_vis_grid(visibility,obs_len=input_size)
    encoder = torch.load(netfile+'encoder.pk')
    decoder = torch.load(netfile+'decoder.pk')
    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')
    encoder.train()
    decoder.train()

    seq,_,_,_ = generate_sample(vis_grid,None,max_obs,device='cpu')
    seqs = [seq for _ in range(len(vis_grid))]
    encoded_seqs,_ = encoder(seqs)
    seq = seq.numpy()

    print(len(seq))
    length = 0 if np.array_equal(seq,np.array([[0for _ in range(config['input_size'])]]))  else len(seq)

    test_data = vis_grid
    test_data = torch.tensor(test_data,dtype=torch.float32)
    
    predictions = decoder.predict(encoded_seqs,test_data,threshold=0).detach().flatten().numpy()
    test_data = test_data.numpy()
    
    image = np.full((2*visibility+1,2*visibility+1,3),[0,0,0],dtype=np.uint8)
    image[visibility,visibility,:] = [255,255,255]

    for i in range(len(vis_grid)):
        test = test_data[i,:]

        pred = predictions[i]
        test_in_seq = any([all(test==foe) for foe in seq])

        x = int(test[0]) + visibility
        y = int(test[1]) + visibility

        if test_in_seq:
            if pred:
                image[x,y,:] = TP
            else:
                image[x,y,:] = FN
        else:
            if pred:
                image[x,y,:] = FP
            else:
                image[x,y,:] = TN

                    
    image = cv.resize(image,(480,480),interpolation=cv.INTER_NEAREST)
    # cv.imshow('image',image)
    # cv.waitKey(0)             
    cv.imwrite(f'tests/{config_id}/{length}obs.png', image) 

if __name__ == '__main__':
    
    # cProfile.run('main()')
    if len(sys.argv) > 2 and sys.argv[2] == '--test':
        for _ in range(100):
            test()
    else:
        main()
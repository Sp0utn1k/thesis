from tanksEnv.utils.networks import RNNetwork, DecoderNN
import yaml
import math
import random
import copy
import torch
from torch import nn,optim
from tensorboardX import SummaryWriter
import time
import sys
import numpy as np
import cv2 as cv

def generate_dummy_foe(vis_grid,idx=0):

    if isinstance(idx,int):
        assert idx>-1, "id must be positive or 0."
    else:
        assert isinstance(idx,list) and all(list(map(lambda x: isinstance(x,int), idx))), "id must be an integer or a list of integers."
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
    length = random.randrange(max_length+1)
    ids = list(range(max_id+1))
    for _ in range(length):
        foe = generate_dummy_foe(vis_grid,idx=ids)
        pos = foe[:2]
        while pos in positions:
            foe = generate_dummy_foe(vis_grid,idx=ids)
            pos = foe[:2]
        ids.remove(foe[2])
        observation.append(foe)
        positions.append(pos)
    if length == 0:
        observation.append([0,0,-100])
        positive = observation[0]
        r = random.random()
        while positive in observation:
            if r < .5:
                positive = random.choice(observation)[:2]
            else:
                positive = copy.deepcopy(random.choice(vis_grid))
            positive.append(random.randrange(max_id+1))
        no_agent = True
    else:
        positive = copy.deepcopy(random.choice(observation))
        no_agent = False

    positive = torch.tensor(positive,dtype=torch.float32,device=device)

    negative = observation[0]
    r = random.random()
    while negative in observation:
        if r < .5:
            negative = random.choice(observation)[:2]
        else:
            negative = copy.deepcopy(random.choice(vis_grid))
        negative.append(random.randrange(max_id+1))
    negative = torch.tensor(negative,dtype=torch.float32,device=device)
    observation = torch.tensor(observation,dtype=torch.float32,device=device)

    return observation,positive,negative,no_agent

def generate_batch(vis_grid,max_length,max_id,batch_size,device='cpu'):
    observations = []
    positives = []
    negatives = []
    no_agents_mask = []
    for _ in range(batch_size):
        observation,positive,negative,no_agent = generate_sample(vis_grid,max_length,max_id,device=device)
        observations.append(observation)
        positives.append(positive)
        negatives.append(negative)
        no_agents_mask.append(no_agent)

    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    no_agents_mask = torch.BoolTensor(no_agents_mask)
    return observations, positives, negatives, no_agents_mask

def main():

    config_id = sys.argv[1]

    config_id = f'foes{config_id}'
    print(f'Config: {config_id}.yml')
    with open(f'configs/{config_id}.yml','r') as file:
        config = yaml.safe_load(file)


    netfile = f'nets/{config_id}/'
    device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu',False) else "cpu")
    print(f'Device: {device}')

    visibility = config['visibility']
    max_id = config['max_id']
    max_foes = config['max_foes']
    batch_size = config['batch_size']
    epochs = config['epochs']
    use_writer = config.get('use_writer',False)
    freeze_period = config['freeze_period']
    num_layers = config['num_layers']
    start_freezing = config['start_freezing']
    learning_rate = config['learning_rate']
    save_model = config['save_model']
    load_model = config['load_model']
    save_period = config['save_period']

    obs_length = 3
    frozen_layer = 0

    if load_model:
        encoder = torch.load(netfile+'encoder.pk')
        decoder = torch.load(netfile+'decoder.pk')
    else:
        encoder = RNNetwork(**config)
        decoder = DecoderNN(config['output_size']+obs_length,hidden_layers=config['decoder_hidden_layers'])
    encoder = encoder.to(device)
    decoder = decoder.to(device)


    if save_model and not load_model:
        input(f'A new model will be created and saved in {netfile}.\nPress enter to continue.')
    
    vis_grid = generate_vis_grid(visibility)
    loss_f = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=learning_rate)
    if use_writer:
        timestr = time.strftime('/%Y_%m_%d-%Hh%M')
        writer = SummaryWriter('runs/'+config_id+timestr)

    for epoch in range(epochs):

        if epoch % freeze_period == 0 and epoch > start_freezing:
            encoder.unfreeze_all()
            frozen_layer += 1
            frozen_layer = frozen_layer % num_layers
            encoder.freeze_all_except(frozen_layer)

        observations,positives,negatives,no_agents_mask = generate_batch(vis_grid,max_foes,max_id,batch_size,device=device)
        output,_ = encoder(observations)
        pos_pred = decoder(output,positives)
        neg_pred = decoder(output,negatives)
        pos_labels = torch.ones_like(pos_pred)
        pos_labels[no_agents_mask] = 0
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

    TN = [0,32,0]
    FP = [0,0,32]
    TP = [0,255,0]
    FN = [0,0,255]

    config_id = sys.argv[1]
    config_id = f'foes{config_id}'
    print(f'Config: {config_id}.yml')
    with open(f'configs/{config_id}.yml','r') as file:
        config = yaml.safe_load(file)

    netfile = f'nets/{config_id}/'
    device = 'cpu'
    print(f'Device: {device}')

    visibility = config['visibility']
    max_id = config['max_id']
    max_foes = config['max_foes']

    vis_grid = generate_vis_grid(visibility)
    encoder = torch.load(netfile+'encoder.pk')
    decoder = torch.load(netfile+'decoder.pk')
    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')
    encoder.train()
    decoder.train()

    seq,_,_,_ = generate_sample(vis_grid,max_foes,max_id,device='cpu')
    seqs = [seq for _ in range(len(vis_grid))]
    encoded_seqs,_ = encoder(seqs)
    seq = seq.numpy()

    print(seq)
    length = 0 if np.array_equal(seq,np.array([[0,0,-100]]))  else len(seq)

    for idx in range(max_id+1):

        test_data = [coords + [idx] for coords in vis_grid]
        test_data = torch.tensor(test_data,dtype=torch.float32)
        
        predictions = decoder.predict(encoded_seqs,test_data).detach().flatten().numpy()
        test_data = test_data.numpy()
        
        image = np.full((2*visibility+1,2*visibility+1,3),[0,0,0],dtype=np.uint8)
        image[visibility,visibility,:] = [255,255,255]

        for i in range(len(vis_grid)):
            test = test_data[i,:]

            pred = int(round(predictions[i]))
            test_in_seq = any([all(test==foe) for foe in seq])
            test = test_data[i,:]

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
        cv.imwrite(f'tests/{config_id}/{length}tanks//id={idx}.png', image) 






if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '--test':
        test()
    else:
        main()
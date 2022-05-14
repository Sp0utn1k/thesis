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

def generate_dummy_friend(vis_grid,idx=0,ammo=0):

    if isinstance(idx,int):
        assert idx>-1, "id must be positive or 0."
    else:
        assert isinstance(idx,list) and all(list(map(lambda x: isinstance(x,int), idx))), "id must be an integer or a list of intigers."
        idx = random.choice(idx)

    if isinstance(ammo,int):
        assert ammo>-1, "id must be positive or 0."
    else:
        assert isinstance(ammo,list) and all(list(map(lambda x: isinstance(x,int), ammo))), "ammo must be an integer or a list of intigers."
        ammo = random.choice(ammo)

    [x,y] = random.choice(vis_grid)
    return [x,y,idx,ammo]

def generate_vis_grid(size):
    
    vis_grid = [[x,y] for x in range(-1,size+1) for y in range(-1,size+1)]
    return vis_grid

def generate_sample(vis_grid,max_length,max_id,max_ammo,device='cpu'):
    observation = []
    positions = []
    length = random.randrange(max_length+1)
    ids = list(range(max_id+1))
    ammos = list(range(max_ammo+1))
    for _ in range(length):
        friend = generate_dummy_friend(vis_grid,idx=ids,ammo=ammos)
        pos = friend[:2]
        while pos in positions:
            friend = generate_dummy_friend(vis_grid,idx=ids,ammo=ammos)
            pos = friend[:2]
        ids.remove(friend[2])
        ammos.remove(friend[3])
        observation.append(friend)
        positions.append(pos)


    if length == 0:
        observation.append([0,0,-1,-1])
        # while positive in observation:
        #     print(r)
        #     if  r < .5:
        #         positive = random.choice(observation)
        #         if r < .25:
        #             positive[2] = random.randrange(max_id+1)
        #         else:
        #             positive[3] = random.randrange(max_ammo+1)
        #     else:
        positive = copy.deepcopy(random.choice(vis_grid))
        positive.append(random.randrange(max_id+1))
        positive.append(random.randrange(max_ammo+1))
        no_agent = True
    else:
        positive = copy.deepcopy(random.choice(observation))
        no_agent = False

    positive = torch.tensor(positive,dtype=torch.float32,device=device)

    
    negative = observation[0]
    r = random.random()
    while negative in observation:
        if  r < .5:
            negative = copy.copy(random.choice(observation))
            if r < .25:
                negative[2] = random.randrange(max_id+1)
            else:
                negative[3] = random.randrange(max_ammo+1)
        else:
            negative = copy.deepcopy(random.choice(vis_grid))
            negative.append(random.randrange(max_id+1))
            negative.append(random.randrange(max_ammo+1))
    negative = torch.tensor(negative,dtype=torch.float32,device=device)
    observation = torch.tensor(observation,dtype=torch.float32,device=device)

    return observation,positive,negative,no_agent

def generate_batch(vis_grid,max_length,max_id,max_ammo,batch_size,device='cpu'):
    observations = []
    positives = []
    negatives = []
    no_agents_mask = []
    for _ in range(batch_size):
        observation,positive,negative,no_agent = generate_sample(vis_grid,max_length,max_id,max_ammo,device=device)
        observations.append(observation)
        positives.append(positive)
        negatives.append(negative)
        no_agents_mask.append(no_agent)

    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    no_agents_mask = torch.BoolTensor(no_agents_mask)
    return observations, positives, negatives, no_agents_mask

if __name__ == '__main__':

    config_id = sys.argv[1]

    config_id = f'friends_state{config_id}'
    print(f'Config: {config_id}.yml')
    with open(f'configs/{config_id}.yml','r') as file:
        config = yaml.safe_load(file)


    netfile = f'nets/{config_id}/'
    device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu',False) else "cpu")
    print(f'Device: {device}')

    size = config['size']
    max_id = config['max_id']
    max_ammo = config['max_ammo']
    max_friends = config['max_friends']
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

    obs_length = 4
    frozen_layer = 0

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
    
    vis_grid = generate_vis_grid(size)
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

        observations,positives,negatives,no_agents_mask = generate_batch(vis_grid,max_friends,max_id,max_ammo,batch_size,device=device)
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
    
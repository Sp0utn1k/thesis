from tanksEnv.utils.networks import RNNetwork, DecoderNN
from tanksEnv import tanksEnv
import yaml
import math
import random
import copy
import torch
from torch import nn,optim
from tensorboardX import SummaryWriter


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
        pos = foe[:2]
        if pos in positions:
            return generate_sample(vis_grid,max_length,max_id,device=device)
        positions.append(pos)
    observation = torch.tensor(observation,dtype=torch.float32,device=device)


    positive = random.choice(observation)
    empty_tiles = [pos for pos in vis_grid if pos not in positions]
    negative = copy.deepcopy(random.choice(empty_tiles))
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
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return observations, positives, negatives

if __name__ == '__main__':
    with open(f'configs/config1.yml','r') as file:
        config = yaml.safe_load(file)

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

    obs_length = 3
    frozen_layer = 0

    vis_grid = generate_vis_grid(visibility)
    encoder = RNNetwork(**config).to(device)
    decoder = DecoderNN(config['output_size']+obs_length,hidden_layers=config['decoder_hidden_layers']).to(device)
    loss_f = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=0.003)
    if use_writer:
        writer = SummaryWriter()

    for epoch in range(epochs):

        if epoch % freeze_period == 0 and epoch > start_freezing:
            encoder.unfreeze_all()
            frozen_layer += 1
            frozen_layer = frozen_layer % num_layers
            encoder.freeze_all_except(frozen_layer)

        observations,positives,negatives = generate_batch(vis_grid,max_foes,max_id,batch_size,device=device)
        output,_ = encoder(observations)
        pos_pred = decoder(output,positives)
        neg_pred = decoder(output,negatives)
        pos_labels = torch.ones_like(pos_pred)
        neg_labels = torch.zeros_like(neg_pred)
        pred = torch.cat([pos_pred,neg_pred])
        labels = torch.cat([pos_labels,neg_labels])

        optimizer.zero_grad()
        loss = loss_f(pred,labels)
        loss.backward()
        optimizer.step()

        if use_writer:
            pos_pred = decoder.predict(output,positives)
            neg_pred = decoder.predict(output,negatives)
            true_pos = sum(pos_pred).detach().cpu().numpy()[0]
            false_neg = batch_size - true_pos
            false_pos = sum(neg_pred).detach().cpu().numpy()[0]
            true_neg = batch_size - false_pos

            accuracy = (true_pos+true_neg)/(2*batch_size)
            recall = true_pos/(true_pos+false_neg)

            writer.add_scalar('Loss',loss.item(),epoch)
            writer.add_scalar('Accuracy',accuracy,epoch)
            writer.add_scalar('Recall',recall,epoch)


    if use_writer:
        writer.close()

    rnn1 = nn.LSTM(input_size=10, hidden_size=5, num_layers=3)
    
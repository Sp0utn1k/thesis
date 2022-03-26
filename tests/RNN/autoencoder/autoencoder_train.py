from utils.networks import RNNetwork, DecoderNN
import torch
import torch.nn as nn
import sys
import random
import os
import yaml
import numpy as np
import math
from tensorboardX import SummaryWriter
import time

sys.path.append(os.path.abspath('../../'))
from tanksEnv import tanksEnv

class trainerNN(nn.Module):
	def __init__(self,**kwargs):
		super().__init__()
		input_size = kwargs['input_size']
		output_size = kwargs['output_size']
		self.encoder = RNNetwork(**kwargs)
		self.decoder = DecoderNN(input_size+output_size,hidden_layers=kwargs.get('decoder_hidden_layers',[]))

	def forward(self,data,x):
		data,_ = self.encoder(data)
		return self.decoder(data,x)

	def predict(self,data,x):
		data,_ = self.encoder(data)
		return self.decoder.predict(data,x)

def visibility_grid(vis):
	return [[i,j] for i in range(-vis-1,vis) 
		for j in range(-vis-1,vis) if norm([i,j])<vis]

def norm(vect):
	x,y = vect
	res = x**2 + y**2
	return math.sqrt(res)

def sample_tensor(x,sz):
	return x[torch.randint(0, x.size(0), (sz,))]

def generate_batch(env,device='cpu',testing_reduction=1,**settings):

	N_obstacles = settings.get('N_obstacles',0)
	elements_per_sample = settings.get('elements_per_sample',2)
	batch_size = settings.get('batch_size',10)

	batch_size /= testing_reduction

	env_grid = [[i,j] for i in range(env.size[0]) for j in range(env.size[1])]
	vis_grid = visibility_grid(env.visibility)
	
	all_obs = []
	all_coords = []
	all_labels = []
	while len(all_labels) < batch_size:
		env.obstacles = random.sample(env_grid,N_obstacles)
		env.add_borders_to_obstacles()
		env.reset()
		for player in env.players:
			pos = env.positions[player]
			obs = env.get_list_obstacles(player)
			if len(obs) < np.ceil(elements_per_sample/2):
				coords = obs
			else:
				coords = random.sample(obs,int(np.ceil(elements_per_sample/2)))
			true_vis = [coord for coord in vis_grid if coord not in obs and env.is_visible(pos,add(pos,coord))]
			sample_size = min(len(true_vis),elements_per_sample-len(coords))
			coords += random.sample(true_vis,sample_size)
			labels = [c in obs for c in coords]

			obs = [obs for _ in range(elements_per_sample)]
			obs = torch.tensor(obs,dtype=torch.float32,device=device)
			all_obs += obs
			all_coords += coords
			all_labels += labels
	all_obs = all_obs[:int(batch_size)]
	all_coords = torch.tensor(all_coords[:int(batch_size)],dtype=torch.float32,device=device)
	all_labels = torch.tensor(all_labels[:int(batch_size)],dtype=torch.float32,device=device).unsqueeze(-1)

	return all_obs, all_coords, all_labels

def add(v1,v2):
	return [v1[i]+v2[i] for i in range(len(v1))]


def train(config_id=1,write_result=False):

	with open(f'configs/config{config_id}.yml','r') as file:
		settings = yaml.safe_load(file)

	epochs = settings.get('epochs',1)
	device = 'cuda' if settings.get('use_gpu',False) and torch.cuda.is_available() else 'cpu'
	settings['device'] = device

	print(f'Device : {device}')
	env = tanksEnv.Environment(**settings)
	netfile = f'nets/config{config_id}.pk'

	if settings.get('load_model',False):
		print(f'Loading existing model in {netfile}')
		trainer = torch.load(netfile)
	else:
		trainer = trainerNN(**settings).to(device)


	optimizer = torch.optim.Adam(trainer.parameters())
	loss_f = nn.BCEWithLogitsLoss()

	if write_result:
		timestr = time.strftime('%Y_%m_%d-%Hh%M')
		writer = SummaryWriter(f'runs/config{config_id}/'+timestr)
	losses = []
	for epoch in range(epochs):

		print(f'{epoch}/{epochs}')
		obs, coords, labels = generate_batch(env,**settings)
		output = trainer(obs,coords)
		if epoch == 0 and write_result:
			writer.add_graph(trainer.encoder,obs[0].unsqueeze(-2))

		with torch.no_grad():
			accuracy = torch.round(nn.Sigmoid()(output)) == labels
			accuracy = accuracy.detach().cpu().flatten().numpy()
			accuracy = np.sum(accuracy)/accuracy.shape[0]

		loss = loss_f(output,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if settings.get('save_model',False):
			torch.save(trainer,netfile)

		if write_result:
			writer.add_scalar('loss',loss.item(),epoch)
			writer.add_scalar('accuracy',accuracy,epoch)
	if write_result:
		writer.close()	

def train_multiple_configs(config_ids=[1],write_result=False):
	for config_id in config_ids:
		train(config_id=config_id,write_result=write_result)

if __name__ == '__main__':
	# config_folder = 'configs'
	# for config_file in os.listdir(config_folder):
	# 	if 'yml' in config_file:
	# 		config_id = int(config_file.split('.')[0][-1])
	# 		train(config_id=config_id,write_result=True)

	train(write_result=True)
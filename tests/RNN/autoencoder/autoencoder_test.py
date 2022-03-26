from networks import EncoderNN, DecoderNN
import torch
import torch.nn as nn
from autoencoder_train import *
import yaml
import sys, os
import numpy as np

sys.path.append(os.path.abspath('../../'))
from tanksEnv import env as tanksEnv

def test(testing_reduction=100):

	with open('encoder_params.yml','r') as file:
		settings = yaml.safe_load(file)

	device = 'cuda' if settings.get('use_gpu',False) and torch.cuda.is_available() else 'cpu'
	# device = 'cuda'
	netfile = settings.get('netfile','model.pk')
	autoencoder = torch.load(netfile).to(device)
	env = tanksEnv.Environment(**settings)

	with torch.no_grad():
		obs, coords, labels = generate_batch(env,device = device,testing_reduction=testing_reduction,**settings)
		print(obs)
		output = autoencoder.predict(obs,coords)
		res = labels == output
		res = res.detach().cpu().flatten().numpy()

	return np.sum(res)/res.shape[0]

if __name__ == '__main__':
	print(test(testing_reduction=10))
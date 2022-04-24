import os
import yaml

for configfile in os.listdir(os.getcwd()):
	if 'yml' in configfile:
		with open(configfile,'r') as file:
			config = yaml.safe_load(file)

		config['elements_per_sample'] = 2
		config['epochs'] = 2000
		config['batch_size'] = 400
		config['use_gpu'] = True
		config['load_model'] = False

		with open(configfile,'w') as file:
			yaml.dump(config,file)

		print(f'{configfile} successfuly modified')
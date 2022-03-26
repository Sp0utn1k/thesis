import yaml
import tanksEnv.tanksEnv
from tanksEnv.utils.networks import FCNetwork
from tanksEnv.utils.DQN import Agent, DQNRunner 
from tensorboardX import SummaryWriter
import time

if __name__ == "__main__":
	with open('./configs/dqn_config.yml','r') as file:
		settings = yaml.safe_load(file)

	env = tanksEnv.Environment(**settings)
	S = env.reset()
	obs_size = len(S)
	n_actions = env.n_actions

	agent_net = FCNetwork(obs_size,n_actions,**settings)
	agent = Agent(network=agent_net,**settings)
	runner = DQNRunner(env,agent,**settings)

	timestr = time.strftime('%Y_%m_%d-%Hh%M')
	writer = SummaryWriter('runs/simple DQN/'+timestr)
	
	for (episode,episode_length,reward,loss) in runner.run(settings['n_episodes'],render=settings.get('render',False)):
		writer.add_scalar('reward',reward,episode)
		if loss != None:
			writer.add_scalar('loss',loss,episode)
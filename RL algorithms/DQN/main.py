from utils.networks.FCNetwork import FCNetwork
from utils.algorithms.DQN import Agent, DQNRunner 
from tensorboardX import SummaryWriter
import gym

if __name__ == "__main__":	
	########################### PARAMETERS #################################
	BATCH_SIZE = 128
	BUFFER_SIZE = 2000
	N_EPISODES = 1500
	LEARNING_RATE = 1e-4
	EPSILON_DECAY = {
					'period': N_EPISODES,
					'start':.8,
					'stop':.01,
					'shape':'exponential'
					}

	GAMMA = .9
	NET_SYNC_PERIOD = 2
	########################################################################

	env = gym.make("CartPole-v0")
	obs_space = env.observation_space.shape[0]
	n_actions = env.action_space.n

	agent_net = FCNetwork(obs_space,n_actions,hidden_layers=[16,64,64,16,4])
	agent = Agent(network=agent_net,gamma=GAMMA,epsilon_decay=EPSILON_DECAY,use_gpu=True,lr=LEARNING_RATE)
	runner = DQNRunner(env,agent,buffer_size=BUFFER_SIZE,batch_size=BATCH_SIZE,net_sync_period=NET_SYNC_PERIOD)
	
	writer = SummaryWriter()

	for (episode,episode_length,reward,loss) in runner.run(N_EPISODES):
		writer.add_scalar('reward',reward,episode)
		if loss != None:
			writer.add_scalar('loss',loss,episode)

	writer.close()

	# # Demo:
	# agent.epsilon = 0
	# for _ in range(N_demos):
	# 	done = False
	# 	S = torch.tensor(env.reset(),device=agent.device).unsqueeze(0)
	# 	while not done:
	# 		env.render()
	# 		A = agent.get_action(S)
	# 		S, _, done, _ = env.step(A)
	# 		S = torch.tensor(S,device=agent.device).unsqueeze(0)
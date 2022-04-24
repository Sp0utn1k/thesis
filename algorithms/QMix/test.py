from tanksEnv import twoStepEnv
from tanksEnv.algorithms import QMix
from tanksEnv.utils.networks import FCNetwork,RNNetwork
from tensorboardX import SummaryWriter
import torch

if __name__ == '__main__':

    N_episodes = 5000
    # epsilon_decay = {
    #     'start':1.0,
    #     'stop':0.2,
    #     'period':N_episodes,
    #     'shape':'linear'
    # }
    env = twoStepEnv.Environment()
    network = FCNetwork(env.obs_size,env.n_actions,hidden_layers=[64])
    # network = RNNetwork(env.obs_size,env.n_actions,rnn_size=64)
    qmix = QMix.QMix(env,network,unique_net=False,use_gpu=True,epsilon=1,gamma=.99,lr=5e-4,
        mixer_hidden_layer=8,optimizer='adam',use_mixer=True)

    runner = QMix.QMixRunner(env,qmix,batch_size=64,net_sync_period=100,buffer_size=500)
    # writer = SummaryWriter()
    
    for episode,episode_length,reward,loss in runner.run(N_episodes):
        # if loss != None:
            # writer.add_scalar('loss',loss,episode)
        # if episode % 1000 == 0:
            # print(f'{episode}')
        continue
    # writer.close()


    qmix.epsilon = 0
    for episode,episode_length,reward,loss in runner.run(1):
        print(f'Reward: {reward}')

    # State 1
    actions1 = torch.tensor([0,0,1,1],device=qmix.device).unsqueeze(-1)
    actions2 = torch.tensor([0,1,0,1],device=qmix.device).unsqueeze(-1)
    state = [[-1,0],[-1,1]]
    obs1,obs2 = state[0],state[1]
    obs1 = torch.tensor(obs1,device=qmix.device,dtype=torch.float32).unsqueeze(0).repeat(4,1)
    obs2 = torch.tensor(obs2,device=qmix.device,dtype=torch.float32).unsqueeze(0).repeat(4,1)
    state= torch.tensor(state,device=qmix.device,dtype=torch.float32).flatten().unsqueeze(0).repeat(4,1)

    if qmix.rnn:
        obs1 = obs1.unsqueeze(0)
        obs2 = obs2.unsqueeze(0)
    
    net1 = qmix.agents['agent0']
    Q1,_ = net1(obs1)
    Q1 = Q1.gather(1,actions1)
    net2 = qmix.agents['agent1']
    Q2,_ = net2(obs2)
    Q2 = Q2.gather(1,actions2)
    Q = torch.cat([Q1,Q2],dim=1)
    print(qmix.mixer(Q,state).detach().cpu())



    # State 2A
    actions1 = torch.tensor([0,0,1,1],device=qmix.device).unsqueeze(-1)
    actions2 = torch.tensor([0,1,0,1],device=qmix.device).unsqueeze(-1)
    state = [[0,0],[0,1]]
    obs1,obs2 = state[0],state[1]
    obs1 = torch.tensor(obs1,device=qmix.device,dtype=torch.float32).unsqueeze(0).repeat(4,1)
    obs2 = torch.tensor(obs2,device=qmix.device,dtype=torch.float32).unsqueeze(0).repeat(4,1)
    state= torch.tensor(state,device=qmix.device,dtype=torch.float32).flatten().unsqueeze(0).repeat(4,1)

    if qmix.rnn:
        obs1 = obs1.unsqueeze(0)
        obs2 = obs2.unsqueeze(0)
    
    net1 = qmix.agents['agent0']
    Q1,_ = net1(obs1)
    Q1 = Q1.gather(1,actions1)
    net2 = qmix.agents['agent1']
    Q2,_ = net2(obs2)
    Q2 = Q2.gather(1,actions2)
    Q = torch.cat([Q1,Q2],dim=1)
    print(qmix.mixer(Q,state).detach().cpu())


    # State 2B
    actions1 = torch.tensor([0,0,1,1],device=qmix.device).unsqueeze(-1)
    actions2 = torch.tensor([0,1,0,1],device=qmix.device).unsqueeze(-1)
    state = [[1,0],[1,1]]
    obs1,obs2 = state[0],state[1]
    obs1 = torch.tensor(obs1,device=qmix.device,dtype=torch.float32).unsqueeze(0).repeat(4,1)
    obs2 = torch.tensor(obs2,device=qmix.device,dtype=torch.float32).unsqueeze(0).repeat(4,1)
    state= torch.tensor(state,device=qmix.device,dtype=torch.float32).flatten().unsqueeze(0).repeat(4,1)

    if qmix.rnn:
        obs1 = obs1.unsqueeze(0)
        obs2 = obs2.unsqueeze(0)
    
    net1 = qmix.agents['agent0']
    Q1,_ = net1(obs1)
    Q1 = Q1.gather(1,actions1)
    net2 = qmix.agents['agent1']
    Q2,_ = net2(obs2)
    Q2 = Q2.gather(1,actions2)
    Q = torch.cat([Q1,Q2],dim=1)
    print(qmix.mixer(Q,state).detach().cpu())

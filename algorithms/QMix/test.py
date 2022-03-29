from tanksEnv import twoStepEnv
from tanksEnv.algorithms import QMix
from tanksEnv.algorithms.DQN import Agent
from tanksEnv.utils.networks import FCNetwork
import copy
import torch
from collections import deque


if __name__ == '__main__':
    env = twoStepEnv.Environment()
    network = FCNetwork(env.obs_size,env.n_actions,hidden_layers=[5])
    networks = {name:copy.deepcopy(network) for name in env.agents}
    qmix = QMix.QMix(env,networks)
    runner = QMix.QMixRunner(env,qmix)
    runner.run(2)
from tanksEnv import twoStepEnv
from tanksEnv.algorithms import QMix
from tanksEnv.utils.networks import FCNetwork
import copy


if __name__ == '__main__':
    env = twoStepEnv.Environment()
    network = FCNetwork(env.obs_size,env.n_actions)
    networks = {name:copy.deepcopy(network) for name in env.agents}
    qmix = QMix.Agent(env,network)

from tanksEnv import tanksEnv


env = tanksEnv.Environment()
env.R50 = 1
print(env.Phit(2))
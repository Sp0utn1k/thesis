import numpy as np
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
max_obs = 0
min_obs = 0
env.reset()

pos = np.array([])
speed = np.array([])
angle = np.array([])
omega = np.array([])

for _ in range(1000):
  # env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  max_obs = max(max_obs,*observation)
  min_obs = min(min_obs,*observation)

  pos = np.append(pos,observation[0])
  speed = np.append(speed,observation[1])
  angle = np.append(angle,observation[2])
  omega = np.append(omega,observation[3])

  if done:
    observation = env.reset()


env.close()
print(min_obs)
print(max_obs)

n, bins, patches = plt.hist(x=omega, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.title('Position')
maxfreq = n.max()

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
  
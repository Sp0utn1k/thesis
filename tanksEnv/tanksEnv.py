import numpy as np
import math
import os
from collections import namedtuple
import copy
import pickle
import os
import cv2 as cv
from math import erf, sqrt
import torch

default = {'agents_description':[{'pos0':[0,0],'team': 'blue'},
								{'pos0': [9,9],'team': 'red','policy': None}],
			'size':[5,5],
			'visibility':4,
			'R50':3,
			'obstacles':[],
			'borders_in_obstacles':True,
			'add_graphics':True,
			'N_aim':2,
			'fps':2,
			'max_cycles':-1,
			'remember_aim':True,
			'ammo':-1,
			'im_size':480,
			'observation_mode':'encoded'
			'red_agent':None
			'red_observation_mode':'encoded'
			}
class Agent:
	def __init__(self,team,id=0,policy='user'):
		team = team.lower()
		assert team in ['blue','red'], 'Team must be blue or red'
		self.team = team
		self.id = id
		self.policy = policy
		self.name = self.__str__()

	def __str__(self):
		return f'Agent{self.id} ({self.team})'
	def __repr__(self):
		return self.__str__()

class Environment:
	def __init__(self,**kwargs):

		if 'players_description' in kwargs.keys():
			kwargs['agents_desription'] = kwargs['players_description']
			print('"players_description" deprecated, use "agents_description instead."')

		# for param in kwargs.keys():
		# 	if param not in default.keys():
		# 		raise NameError(f'Unknown parameter \'{param}\'.')
			
		self.observation_mode = kwargs.get('observation_mode',default['observation_mode'])
		self.red_agent = kwargs.get('red_agent',default['red_agent'])
		self.red_observation_mode = kwargs.get('red_observation_mode',default['red_observation_mode'])
		assert self.observation_mode in ['encoded','array','raw'], 'Unknown observation mode.'
		assert self.red_observation_mode in ['encoded','array','raw'], 'Unknown red observation mode.'
		if self.observation_mode == 'encoded' or self.red_observation_mode == 'encoded':
			self.load_encoders()

		self.agents_description = kwargs.get('agents_description',default['agents_description'])
		self.size  = kwargs.get('size',default['size'])
		self.visibility = kwargs.get('visibility',default['visibility'])
		self.R50 = kwargs.get('R50',default['R50'])

		self.obstacles = kwargs.get('obstacles',default['obstacles'])
		if self.observation_mode != 'encoded':
			assert self.obstacles == [] or __name__ == '__main__', 'Using obstacles is only possible in encoder mode.'
		if kwargs.get('borders_in_obstacles',default['borders_in_obstacles']):
			self.add_borders_to_obstacles()

		if kwargs.get('add_graphics',default['add_graphics']):
			self.graphics = Graphics(self,**kwargs)
			self.show_plot = True
			self.twait = 1000.0/kwargs.get('fps',default['fps'])
		else:
			self.graphics = None
			self.show_plot = False

		self.MA = sum([(d['team']=='blue')*d.get('replicas',1) for d in self.agents_description]) > 1
		# if not use_encoder:
		# 	assert not self.MA or __name__ == '__main__', 'Multi-Agents mode is not supported in no_encoder mode.'

		filename = os.path.join(os.getcwd().split('thesis')[-2],'thesis/tanksEnv/los100.pkl')
		with open(filename, 'rb') as file:
			self.los_dict = pickle.load(file)

		self.N_aim = kwargs.get('N_aim',default['N_aim'])
		self.action_names = create_actions_set(self.N_aim)
		self.n_actions = len(self.action_names)
		self.max_cycles = kwargs.get('max_cycles',default['max_cycles'])
		self.remember_aim = kwargs.get('remember_aim',default['remember_aim'])
		self.max_ammo = kwargs.get('ammo',default['ammo'])
		self.reset()

		self.obs_size = len(self.get_observation())
		self.state_size = len(self.get_state())

	def load_encoders(self):
		filename_foe = os.path.join(os.path.dirname(__file__), 'encoders/foes_encoder.pk')
		self.foes_encoder = torch.load(filename_foe).to('cpu')
		self.foes_encoder.eval()

		filename_friend = os.path.join(os.path.dirname(__file__), 'encoders/friends_encoder.pk')
		self.friends_encoder = torch.load(filename_friend).to('cpu')
		self.friends_encoder.eval()

		filename_obstacle = os.path.join(os.path.dirname(__file__), 'encoders/obstacles_encoder.pk')
		self.obstacles_encoder = torch.load(filename_obstacle).to('cpu')
		self.obstacles_encoder.eval()

		filename_foe_state = os.path.join(os.path.dirname(__file__), 'encoders/foes_state_encoder.pk')
		self.foes_state_encoder = torch.load(filename_foe_state).to('cpu')
		self.foes_state_encoder.eval()

		filename_friend_state = os.path.join(os.path.dirname(__file__), 'encoders/friends_state_encoder.pk')
		self.friends_state_encoder = torch.load(filename_friend_state).to('cpu')
		self.friends_state_encoder.eval()

	def add_borders_to_obstacles(self):
		self.obstacles += [[x,-1] for x in range(self.size[0])]
		self.obstacles += [[x,self.size[1]] for x in range(self.size[0])]
		self.obstacles += [[-1,y] for y in range(self.size[1])]
		self.obstacles += [[self.size[0],y] for y in range(self.size[1])]
		
	def init_players(self):
		self.positions = {}
		agents_description = self.agents_description
		self.agents = []
		for description in agents_description:
			team = description['team']
			pos0 = description.get('pos0','random')
			replicas = description.get('replicas',1)
			policy = description.get('policy','user')
			assert team in ['blue','red'], 'Team must be red or blue'
			for _ in range(replicas):
				agent = Agent(team,id=len(self.agents),policy=policy)
				self.agents += [agent]
			if not pos0 == 'random':
				assert replicas == 1, 'Cannot replicate agent with fixed position'
				self.positions[agent] = pos0

	def reset(self):
		self.init_players()
		assert len(self.agents) <= self.N_aim
		self.red_agents = [agent for agent in self.agents if agent.team=="red"]
		self.blue_agents = [agent for agent in self.agents if agent.team=="blue"]
		self.ammo = {agent:self.max_ammo for agent in self.agents}
		self.alive = {}
		self.aim = {}

		for agent in self.agents:
			self.alive[agent] = True
			self.aim[agent] = None
			if not agent in self.positions.keys():
				self.positions[agent] = [np.random.randint(self.size[0]),np.random.randint(self.size[1])]

				while (self.positions[agent] in self.obstacles or
					  self.positions[agent] in [self.positions[p] for p in self.agents 
					  if p in self.positions.keys() and p != agent]):	
					
					self.positions[agent] = [np.random.randint(self.size[0]),np.random.randint(self.size[1])]

		self.current_agent = self.blue_agents[0]
		self.cycle = {agent:0 for agent in self.agents}
		self.n_red = len(self.red_agents)
		self.n_blue = len(self.blue_agents)
		return self.get_observation()

	def get_observation(self):

		agent = self.current_agent
		pos = copy.copy(self.positions[agent])

		observation_mode = self.observation_mode if agent.team=='blue' else self.red_observation_mode

		team = self.blue_agents if agent.team == 'blue' else self.red_agents
		other_team = self.blue_agents if agent.team == 'red' else self.red_agents

		n_team = self.n_blue if agent.team == 'blue' else self.n_red
		n_others = self.n_blue if agent.team == 'red' else self.n_red

		foes = [substract(self.positions[p],pos)+[p.id] for p in other_team 
				if self.is_visible(pos,self.positions[p]) and pos != self.positions[p]]
		friends = [substract(self.positions[p],pos)+[p.id]+[self.ammo[p]] for p in team
				if self.is_visible(pos,self.positions[p]) and pos != self.positions[p]]

		obstacles = [substract(obs,pos) for obs in obstacles if self.is_visible(pos,obstacles)]

		if not foes:
			foes = [[0,0,-100]]
		if not friends:
			friends = [[0,0,-100,0]]
		if not obstacles:
			obstacles = [[0,0]]

		# Create agent_obs
		agent_obs = pos
		agent_obs.append(self.ammo[agent])
		aim = self.aim.get(agent,None)
		if aim == None:
			aim = -1
		else:
			aim = aim.id
		if self.remember_aim:
			agent_obs += [aim]

		if observation_mode == 'encoded':
			with torch.no_grad():
				foes,_ = self.foes_encoder(torch.tensor(foes,dtype=torch.float32))
				foes = list(foes.squeeze(0).numpy())
				observation = agent_obs + foes
				if self.obstacles:
					obstacles,_ = self.obstacles_encoder(torch.tensor(obstacles,dtype=torch.float32))
					obstacles = list(obstacles.squeeze(0).numpy())
					observation += obstacles
				if self.MA:
					friends,_ = self.friends_encoder(torch.tensor(friends,dtype=torch.float32))
					friends = list(friends.squeeze(0).numpy())
					observation += friends

		elif observation_mode == 'array':
			while len(foes) < n_others:
				foes.append([0,0,-100])
			foes = np.array(foes).flatten()
			observation = agent_obs + list(foes)
			if self.obstacles:
				while len(obstacles) < len(self.obstacles):
					obstacles.append([0,0])
				obstacles = np.array(obstacles).flatten()
				observation += list(obstacles)obstacles
			if self.MA:
				while len(friends) < n_team:
					foes.append([0,0,-100,0])
				friends = np.array(friends).flatten()
				observation += list(friends)

		elif observation_mode == 'raw':
			if self.MA:
				observation = agent_obs,foes,friends
			else:
				observation = agent_obs,foes

		return observation

	def get_state(self, team_name='blue'):

		team = self.blue_agents if team_name == 'blue' else self.red_agents
		other_team = self.blue_agents if team_name == 'red' else self.red_agents

		n_team = self.n_blue if team_name == 'blue' else self.n_red
		n_others = self.n_blue if team_name == 'red' else self.n_red

		foes = [self.positions[p]+[p.id] for p in other_team]
		friends = []
		for p in team:
			friend = copy.copy(self.positions[p])
			friend.append(p.id)
			friend.append(self.ammo[p])
			# aim = self.aim.get(p,None)
			# if aim == None:
			# 	aim = -1
			# else:
			# 	aim = aim.id
			# friend.append(aim)
			friends.append(friend)

		obstacles = copy.copy(self.obstacles)

		if not foes:
			foes = [[-10,-10,-100]]
		if not friends:
			friends = [[-10,-10,-100,-1]]

		if not obstacles:
			obstacles = [[-10,-10]]
		
		if self.observation_mode == 'encoded':
			with torch.no_grad():
				foes,_ = self.foes_state_encoder(torch.tensor(foes,dtype=torch.float32))
				foes = list(foes.squeeze(0).numpy())
				observation = foes
				
				friends,_ = self.friends_state_encoder(torch.tensor(friends,dtype=torch.float32))
				friends = list(friends.squeeze(0).numpy())
				observation += friends

		elif self.observation_mode == 'array':
			while len(foes) < n_others:
				foes.append([0,0,-1])
			foes = np.array(foes).flatten()
			observation = list(foes)
			if self.MA:
				while len(friends) < n_team:
					foes.append([0,0,-1,0])
				friends = np.array(friends).flatten()
				observation += list(friends)

		elif self.observation_mode == 'raw':
				observation = foes, friends, obstacles

		return observation


	def next_tile(self,agent,act):
		tile = self.positions[agent]
		x,y = tile
		assert act in ['north','south','west','east'], "Unauthorized movement"
		if act =='north':
			y -= 1
		elif act == 'south':
			y += 1
		elif act == 'west':
			x -= 1
		else:
			x += 1
		return [x,y]

	def is_free(self,tile):
		x,y = tile
		if x < 0 or x >= self.size[0] or y < 0 or y >= self.size[1]:
			return False
		if [x,y] in self.obstacles:
			return False
		if [x,y] in self.positions.values():
			return False
		return True

	def is_valid_action(self,agent,act):
		act = self.action_names[act]
		if act == 'nothing':
			return True
		if act in ['north','south','west','east']:
			return self.is_free(self.next_tile(agent,act))
		if act == 'shoot':
			if self.aim[agent] in self.agents:
				if not self.ammo[agent]:
					return False
				if self.aim[agent].id not in self.visible_targets_id(agent):
					if not self.remember_aim:
						self.aim[agent] = None
				return True
		if 'aim' in act:
			target = int(act[3:])
			if target in self.visible_targets_id(agent):
				return True
		return False

	def action(self,action):
		agent = self.current_agent
		if not self.is_valid_action(agent,action):
			return False
		act = self.action_names[action]
		if act in ['north','south','west','east']:
			self.positions[agent] = self.next_tile(agent,act)
			# print(f'Agent {agent.id} ({agent.team}) goes {act}')
		if act == 'shoot':
			is_hit = self.fire(agent)

			# target = self.aim[agent]
			# print(f'Agent {agent.id} ({agent.team}) shots at agent {target.id} ({target.team})')
			# if is_hit:
			# 	print("hit!")

		if 'aim' in act:
			target_id = int(act[3:])
			self.aim[agent] = self.get_player_by_id(target_id)
			# if self.show_plot:
			# 	target = self.aim[agent]
			# 	print(f'Agent {agent.id} ({agent.team}) aims at agent {target.id} ({target.team})')
			
		return True

	def fire(self,agent):
		if not self.ammo[agent]:
			return False
		self.ammo[agent] -= 1
		target = self.aim[agent]
		distance = norm(self.positions[agent],self.positions[target])
		hit = np.random.rand() < self.Phit(distance)
		if hit:
			self.alive[target] = False
			self.positions[target] = [-1,-1]
			return True
		return False
	
	def Phit(self,r):
		coeffs = [2.4316e-01, 2.4187e-05, 3.9214e-07, -9.2819e-10, 9.5338e-13]
		r = 1.2694e+03*r/self.R50
		std = sum([r**i*coeffs[i] for i in range(len(coeffs))])
		return erf(1/std/sqrt(2))

	def get_reward(self):
		p = self.current_agent
		winner = self.winner()
		if winner == p.team:
			R = 1
		elif winner == None:
			R = -.01
		else:
			R = -1
		return R

	def last(self):
		
		obs = self.get_observation()
		R = self.get_reward()
		done = self.is_done()
		info = None

		return obs,R,done,info

	def play_red_agents(self):
		prev_current = self.current_agent
		for agent in self.red_agents:
			if agent.policy  == None:
				continue
			elif agent.policy == 'random':
				self.current_agent = agent
				available_actions = [action for action in range(len(self.action_names)) 
									if self.is_valid_action(agent,action)]
				action = np.random.choice(available_actions)
				self.action(action)
			else:
				raise AttributeError(f'Policy "{agent.policy}"" is not supported for red agents.')
		self.current_agent = prev_current

	def step(self,action,prompt_action=False):

		if self.MA and self.is_done():
			assert action==None, 'Action must be None when done is True.'
			return 

		agent = self.current_agent
		self.action(action)
		self.cycle[agent] += 1
		if prompt_action:
			print(f'Agent {agent.id} takes action "{self.action_names[action]}".')

		if not self.MA:
			self.play_red_agents()
			self.update_agents()
			R = self.get_reward()
			obs = self.get_observation()
			done = self.is_done()
			info = None
			return obs, R, done, info
		else:
			return

	def get_random_action(self):
		return np.random.choice(list(self.action_names.keys()))

	def agent_iter(self):
		done_agents = []
		user_agents = len([agent for agent in self.agents if agent.policy == 'user'])
		while len(done_agents) < user_agents:
			for i,agent in enumerate(self.agents):
				self.current_agent = agent
				if not self.is_done():
					if agent.policy == 'user':
						yield agent, not i
					elif agent.policy == None:
						continue
					elif agent.policy == 'random':
						available_actions = [action for action in range(len(self.action_names)) 
									if self.is_valid_action(agent,action)]
						action = np.random.choice(available_actions)
						self.action(action)

				elif agent not in done_agents and agent.policy == 'user':
					done_agents.append(agent)
					yield agent, not i
					
		for agent in self.agents:
			if agent not in done_agents and agent.policy == 'user':
				yield agent

	def update_agents(self):
		if not self.MA:
			self.agents = [p for p in self.agents if self.alive[p]]
		# np.random.shuffle(self.agents)
		self.red_agents = [p for p in self.agents if p.team=="red" and self.alive[p] and self.ammo[p]]
		self.blue_agents = [p for p in self.agents if p.team=="blue" and self.alive[p] and self.ammo[p]]

	def get_player_by_id(self,id):
		return [p for p in self.agents if p.id==id][0]

	def is_visible(self,pos1,pos2):
		if pos1 == pos2:
			return True
		dist = norm(pos1,pos2)
		if dist > self.visibility:
			return False
		LOS = self.los(pos1,pos2)
		for pos in LOS:
			if pos in self.obstacles:
				return False
		if not self.in_grid(pos1):
			return False
		if not self.in_grid(pos2):
			return False
		return True

	def in_grid(self,pos):
		if pos[0] < -1 or pos[0] > self.size[0]:
			return False
		if pos[1] < -1 or pos[1] > self.size[1]:
			return False
		return True
			
	def visible_targets_id(self,p1):
		visibles = []
		for p2 in self.agents:
			if p2!=p1 and self.is_visible(self.positions[p1],self.positions[p2]):
				visibles += [p2.id]
		return visibles

	def winner(self):
		self.update_agents()		
		if len(self.blue_agents) == 0:
			winner = 'red'
		elif len(self.red_agents) == 0:
			winner = 'blue'
		else:
			winner = None
		return winner

	def is_done(self):
		agent = self.current_agent
		if self.cycle[agent] >= self.max_cycles:
			return True
		if not self.alive[agent]:
			return True
		if not self.ammo[agent]:
			return True
		if self.winner() != None:
			return True
		return False

	def render(self,twait=-1,save_image=False,filename='render.png'):
		if twait == -1:
			twait = self.twait

		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			if not (x in [-1,self.size[0]] or y in [-1,self.size[1]]):
				self.graphics.set_obstacle(x,y)
		for agent in self.agents:
			idx = agent.id
			team = agent.team	
			pos = self.positions[agent]
			self.graphics.add_agent(idx,team,pos)
		self.graphics.add_grid()
		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		if save_image:				
			cv.imwrite(filename, self.graphics.image) 
		# cv.destroyAllWindows()

	def render_fpv(self,agent,twait=-1,save_image=False,filename='render_fpv.png'):
		if twait == -1:
			twait = self.twait

		if isinstance(agent,int):
			agent = self.get_player_by_id(agent)
		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			if x in range(self.size[0]) and y in range(self.size[1]):
				self.graphics.set_obstacle(x,y)
		for p in self.agents:
			idx = p.id
			team = p.team
			pos = self.positions[p]
			self.graphics.add_agent(idx,team,pos)
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				if not self.is_visible(self.positions[agent],[x,y]):
					self.graphics.delete_pixel(x,y)
		# for [x,y] in los(self.positions[agent],[35,22]):
		# 	self.graphics.set_red(x,y)
		self.graphics.add_grid()
		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		if save_image:				
			cv.imwrite(filename, self.graphics.image) 

	@property
	def N_players(self):
		return len(self.agents)

	def los(self,vect1,vect2):

		if vect2[0] < vect1[0]:
			vect1,vect2 = vect2,vect1

		diff = [vect2[0]-vect1[0],vect2[1]-vect1[1]]
		mirrored = False
		if diff[1] < 0:
			mirrored = True
			diff[1] = -diff[1]

		los = [[i+vect1[0],j*(-1)**mirrored+vect1[1]] for [i,j] in self.los_dict[tuple(diff)]]
		return los

	def get_agents(self):
		return [agent for agent in self.agents if agent.policy=='user']
	
class Graphics:
	def __init__(self,env,**kwargs):
		self.size = kwargs.get('im_size',default['im_size'])
		game_size = env.size
		self.size = (int(np.ceil(self.size*game_size[0]/game_size[1])),self.size)
		self.szi = round(self.size[0]/game_size[0])
		self.background_color = self.to_color([133,97,35])
		self.red = self.to_color([255,0,0])
		self.blue = self.to_color([30,144,255])
		self.obstacles_color = self.to_color([88,86,84])
		self.reset()

	def reset(self):
		self.image = np.full((self.size[1],self.size[0],3),self.background_color,dtype=np.uint8)

	def add_grid(self):
		for i in range(self.szi,self.size[0],self.szi):
			self.image[i,:,:] = 0
		for j in range(self.szi,self.size[1],self.szi):
			self.image[:,j,:] = 0

	def to_color(self,color):
		(r,g,b) = color
		color = (b,g,r)
		return np.array(color,dtype=np.uint8)

	def pixels_in_coord(self,x,y):
		res = []
		szi = self.szi
		for j in range(round(szi*(x)),round(szi*(x+1))):
			for i in range(round(szi*(y)),round(szi*(y+1))):
				yield [i,j]

	def assign_value(self,x,y,val):
		for c in self.pixels_in_coord(x,y):
			self.image[c[0],c[1],:] = val

	def center(self,x,y):
		return (round(x*self.szi),round((y+.92)*self.szi))

	def set_blue(self,x,y):
		self.assign_value(x,y,self.blue)

	def set_red(self,x,y):
		self.assign_value(x,y,self.red)

	def set_obstacle(self,x,y):
		self.assign_value(x,y,self.obstacles_color)

	def delete_pixel(self,x,y):
		self.assign_value(x,y,[0,0,0])

	def add_agent(self,id,team,pos):
		[x,y] = pos
		if team=='red':
			self.set_red(x,y)
		elif team=='blue':
			self.set_blue(x,y)
		cv.putText(self.image,f'{id}',self.center(x,y),cv.FONT_HERSHEY_SIMPLEX,0.6*self.szi/15,(0,0,0),2)

	def erase_tile(self,x,y):
		self.assign_value(x,y,self.background_color)

def norm(vect1,vect2):
	x1,y1 = vect1
	x2,y2 = vect2
	res = (x2-x1)**2 + (y2-y1)**2
	return math.sqrt(res)

def sigmoid(x,l):
	return 1.0/ (1+math.exp(-x*l))

def create_actions_set(N_aim):
	action_names = {1:'north',2:'south',3:'west',4:'east'}
	action_names[0] = 'nothing'
	action_names[5] = 'shoot'
	for i in range(N_aim):
		action_names[6+i] = f'aim{i}'
	return action_names

def substract(pos1,pos2):
	assert len(pos1)==len(pos2), 'unmatched lengths.'
	return [pos1[i]-pos2[i] for i in range(len(pos1))]

def add(pos1,pos2):
	assert len(pos1)==len(pos2), 'unmatched lengths.'
	return [pos1[i]+pos2[i] for i in range(len(pos1))]


if __name__ == '__main__':

	agents_description = [{'pos0':'random','team':'blue'},
						{'pos0':'random','team':'blue'},
						{'pos0':'random','team':'red'},
						{'pos0':'random','team':'red'}
						]

	obstacles = [[x,y] for x in range(9,11) for y in range(2,18)]
	env = Environment(agents_description=agents_description,size=[20,20],use_encoder=False,N_aim=4,im_size=720,
						obstacles=obstacles,visibility = 12)
	images_folder = os.path.abspath('../../images')
	env.render(twait=0,save_image=True,filename = os.path.join(images_folder,'env_render.png'))
	
	# env.agents_description[0]['pos0'] = 'random'
	# env.reset()
	env.render_fpv(0,twait=0,save_image=True,filename = os.path.join(images_folder,'env_render_fpv.png'))
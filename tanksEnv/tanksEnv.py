import numpy as np
import math,time,sys,os
from collections import namedtuple
import copy
import pickle
import os
import torch
from torch import nn, optim
import cv2 as cv
from math import erf

class Player:
	def __init__(self,team,id=0,policy='user'):
		self.team = team
		self.id = id
		self.policy = policy

class Environment:

	def __init__(self,**kwargs):

		self.players_description = kwargs.get('players_description',
			[{'pos0':'random','team':'blue','replicas':1},
			{'pos0':'random','team':'red','replicas':1}])
		self.size  = kwargs.get('size',[5,5])
		self.visibility = kwargs.get('visibility',4)
		self.R50 = kwargs.get('R50',3)

		self.obstacles = kwargs.get('obstacles',[])
		if kwargs.get('borders_in_obstacles',False):
			self.add_borders_to_obstacles()

		if kwargs.get('add_graphics',True):
			self.graphics = Graphics(self,**kwargs)
			self.show_plot = True
			self.twait = 1000.0/kwargs.get('fps',5)
		else:
			self.graphics = None
			self.show_plot = False

		self.MA = sum([(d['team']=='blue')*d.get('replicas',1) for d in self.players_description]) > 1
		filename = os.path.join(os.getcwd().split('tfe')[-2],'tfe/tanksEnv/los100.pkl')
		with open(filename, 'rb') as file:
		    self.los_dict = pickle.load(file)

		self.reset()
		self.action_names = create_actions_set(kwargs.get('N_aim',2))
		self.n_actions = len(self.action_names)
		self.max_cycles = kwargs.get('max_cycles',-1)
		self.remember_aim = kwargs.get('remember_aim',True)
		self.max_ammo = kwargs.get('ammo',-1)

	def add_borders_to_obstacles(self):
		self.obstacles += [[x,-1] for x in range(self.size[0])]
		self.obstacles += [[x,self.size[1]] for x in range(self.size[0])]
		self.obstacles += [[-1,y] for y in range(self.size[1])]
		self.obstacles += [[self.size[0],y] for y in range(self.size[1])]
		
	def init_players(self):
		self.positions = {}
		players_description = self.players_description
		self.players = []
		for d in players_description:
			team = d['team']
			pos0 = d.get('pos0','random')
			replicas = d.get('replicas',1)
			policy = d.get('policy','user')
			assert team in ['blue','red'], 'Team must be red or blue'
			for _ in range(replicas):
				player = Player(team,id=len(self.players),policy=policy)
				self.players += [player]
			if not pos0 == 'random':
				assert replicas == 1, 'Cannot replicate player with fixed position'
				self.positions[player] = pos0

	def reset(self):
		self.init_players()
		self.red_players = [player for player in self.players if player.team=="red"]
		self.blue_players = [player for player in self.players if player.team=="blue"]
		self.ammo = {player:self.max_ammo for player in self.players}
		self.alive = {}
		self.aim = {}

		for player in self.players:
			self.alive[player] = True
			self.aim[player] = None
			if not player in self.positions.keys():
				self.positions[player] = [np.random.randint(self.size[0]),np.random.randint(self.size[1])]

				while (self.positions[player] in self.obstacles or
					  self.positions[player] in [self.positions[p] for p in self.players 
					  if p in self.positions.keys() and self.alive[p] and p != player]):	
					
					self.positions[player] = [np.random.randint(self.size[0]),np.random.randint(self.size[1])]

		if not self.MA:
			self.current_player = self.blue_players[0]
		else:
			self.current_player = self.get_player_by_id(0)

		self.cycle = {player:0 for player in self.players}
		return self.get_state()

	def get_state(self):

		assert not self.MA, 'Multi Agent not properly implemented'
		if not self.MA:
			player = self.current_player
			pos = self.positions[player]
			if len(self.blue_players)!=1:
				print('WARNING: Agent is already dead.')
				return 
			foes = [substract(self.positions[p],pos)+[p.id] for p in self.red_players 
					if self.is_visible(pos,self.positions[p])and pos != self.positions[p]]
			# if not len(foes):
			# 	foes = [[0,0,-1]]
			obstacles = [substract(obstacle,pos) for obstacle in self.obstacles if self.is_visible(pos,obstacle)]

			# Create obs
			obs = copy.copy(pos)
			aim = self.aim.get(player,None)
			if aim == None:
				aim = -1
			else:
				aim = aim.id
			obs += [aim]
			for foe in foes:
				obs += foe
			return obs

	def current_state(self,player):
		pos = self.positions[player]
		ammo = self.ammo[player]
		player_obs = pos + [ammo]

		friends = [substract(self.positions[p],pos)+[p.id] for p in self.blue_players 
			if self.is_visible(pos,self.positions[p]) and pos != self.positions[p]]
		foes = [substract(self.positions[p],pos)+[p.id] for p in self.red_players 
			if self.is_visible(pos,self.positions[p])and pos != self.positions[p]]
		if player.team == 'red':
			friends,foes = foes,friends
		obstacles = [substract(obstacle,pos) for obstacle in self.obstacles if self.is_visible(pos,obstacle)]

		return (player_obs,friends,foes,obstacles)

	def get_list_obstacles(self,player):
		pos = self.positions[player]
		obstacles = [substract(obstacle,pos) for obstacle in self.obstacles if self.is_visible(pos,obstacle)]
		return obstacles

	def next_tile(self,player,act):
		tile = self.positions[player]
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

	def is_valid_action(self,player,act):
		act = self.action_names[act]
		if act == 'nothing':
			return True
		if act in ['north','south','west','east']:
			return self.is_free(self.next_tile(player,act))
		if act == 'shoot':
			if self.aim[player] in self.players:
				if not self.amm0[player]:
					return False
				if self.aim[player].id not in self.visible_targets_id(player):
					if not self.remember_aim:
						self.aim[player] = None
				return True
		if 'aim' in act:
			target = int(act[3:])
			if target in self.visible_targets_id(player):
				return True
		return False

	def action(self,action):
		player = self.current_player
		if not self.is_valid_action(player,action):
			return False
		act = self.action_names[action]
		if act in ['north','south','west','east']:
			self.positions[player] = self.next_tile(player,act)
			# print(f'Player {player.id} ({player.team}) goes {act}')
		if act == 'shoot':
			is_hit = self.fire(player)

			# target = self.aim[player]
			# print(f'Player {player.id} ({player.team}) shots at player {target.id} ({target.team})')
			# if is_hit:
			# 	print("hit!")

		if 'aim' in act:
			target_id = int(act[3:])
			self.aim[player] = self.get_player_by_id(target_id)
			# if self.show_plot:
			# 	target = self.aim[player]
			# 	print(f'Player {player.id} ({player.team}) aims at player {target.id} ({target.team})')
			
		return True

	def fire(self,player):
		if not self.ammo[player]:
			return False
		self.ammo[player] -= 1
		target = self.aim[player]
		distance = norm(self.positions[player],self.positions[target])
		hit = np.random.rand() < self.Phit(distance)
		if hit:
			self.alive[target] = False
			self.positions[target] = [-1,-1]
			return True
		return False
	
	def Phit(self,r):
		return 1/(1+(r/self.R50).^7.5)

	def get_reward(self):
		p = self.current_player
		winner = self.winner()
		if winner == p.team:
			R = 1
		elif winner == None:
			R = -.01
		else:
			R = -1
		return R

	def last(self):
		S = self.observe_state(self.current_player)
		R = self.get_reward(self.current_player)
		done = self.episode_over()
		info = None
		return S_,R,done,info

	def step(self,action,prompt_action=False):
		player = self.current_player
		self.action(action)
		self.update_players()
		self.cycle[player] += 1
		if prompt_action:
			print(f'Agent {player.id} takes action "{self.action_names[action]}".')
		R = self.get_reward()
		S_ = self.get_state()
		done = self.episode_over()
		info = None
		return S_, R, done, info

	def get_random_action(self):
		return np.random.choice(list(self.action_names.keys()))

	def agent_iter(self):
		for p in self.players:
			self.update_players()
			if self.alive[p]:
				self.current_player = p
				yield p.id

	def update_players(self):
		self.players = [p for p in self.players if self.alive[p]]
		# np.random.shuffle(self.players)
		self.red_players = [p for p in self.players if p.team=="red"]
		self.blue_players = [p for p in self.players if p.team=="blue"]

	def get_player_by_id(self,id):
		return [p for p in self.players if p.id==id][0]

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
		for p2 in self.players:
			if p2!=p1 and self.is_visible(self.positions[p1],self.positions[p2]):
				visibles += [p2.id]
		return visibles

	def winner(self):
		if len(self.blue_players) == 0:
			winner = 'red'
		elif len(self.red_players) == 0:
			winner = 'blue'
		else:
			winner = None
		return winner

	def episode_over(self):
		player = self.current_player
		done = self.winner() != None
		done = done or self.cycle[player] == self.max_cycles
		done = done or not self.alive[player]
		return done

	def render(self,twait=1,save_image=False,filename='render.png'):
		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			if not (x in [-1,self.size[0]] or y in [-1,self.size[1]]):
				self.graphics.set_obstacle(x,y)
		for player in self.players:
			id = player.id
			team = player.team	
			pos = self.positions[player]
			self.graphics.add_player(id,team,pos)
		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		if save_image:				
			cv2.imwrite(filename, self.graphics.image) 
		# cv.destroyAllWindows()

	def render_fpv(self,player,twait=1,save_image=False,filename='render_fpv.png'):
		if isinstance(player,int):
			player = self.get_player_by_id(player)
		if not self.show_plot:
			return
		self.graphics.reset()
		for [x,y] in self.obstacles:
			if x in range(self.size[0]) and y in range(self.size[1]):
				self.graphics.set_obstacle(x,y)
		for p in self.players:
			id = p.id
			team = p.team
			pos = self.positions[p]
			self.graphics.add_player(id,team,pos)
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				if not self.is_visible(self.positions[player],[x,y]):
					self.graphics.delete_pixel(x,y)
		# for [x,y] in los(self.positions[player],[35,22]):
		# 	self.graphics.set_red(x,y)

		cv.imshow('image',self.graphics.image)
		cv.waitKey(round(twait))
		if save_image:				
			cv2.imwrite(filename, self.graphics.image) 

	@property
	def N_players(self):
		return len(self.players)

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
	
class Graphics:
	def __init__(self,env,**kwargs):
		self.size = kwargs.get('im_size',480)
		game_size = env.size
		self.size = (int(np.ceil(self.size*game_size[0]/game_size[1])),self.size)
		self.szi = self.size[0]/game_size[0]
		self.background_color = self.to_color([133,97,35])
		self.red = self.to_color([255,0,0])
		self.blue = self.to_color([30,144,255])
		self.obstacles_color = self.to_color([88,86,84])
		self.reset()

	def reset(self):
		self.image = np.full((self.size[1],self.size[0],3),self.background_color,dtype=np.uint8)

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

	def add_player(self,id,team,pos):
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
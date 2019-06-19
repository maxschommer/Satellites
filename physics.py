import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion

import constraint



class RigidBody():
	""" A physical unbending object free to move and rotate in space """
	def __init__(self, I, m, cm_position, init_position=[0,0,0], init_rotation=[1,0,0,0], init_velocity=[0,0,0], init_angularv=[0,0,0]):
		""" I:		      3x3 float array		rotational inertia
			m:		  	  float				    mass
			cm_position:  3 float vector		centre of mass in the mesh coordinate frame
			body_num:     int 					Index of body in solver array
		"""
		self.I = I
		self.I_inv = np.linalg.inv(I)
		self.m = m
		self.cm_position = np.array(cm_position)
		self.body_num = None
		self.environment = None

		self.init_position = np.array(init_position)
		self.init_rotation = Quaternion(init_rotation)
		self.init_velocity = np.array(init_velocity)
		self.init_angularv = np.array(init_angularv)

	def get_position(self, t):
		return self.environment.solution.sol(t)[13*self.body_num:13*self.body_num+3]

	def get_rotation(self, t):
		return Quaternion(self.environment.solution.sol(t)[13*self.body_num+3:13*self.body_num+7])

	def get_velocity(self, t):
		momentum = self.environment.solution.sol(t)[13*self.body_num+7:13*self.body_num+10]
		return momentum/self.m

	def get_angularv(self, t):
		angular_momentum = self.environment.solution.sol(t)[13*self.body_num+10:13*self.body_num+13]
		return self.get_rotation(t).rotate(np.matmul(self.I_inv,self.get_rotation(t).inverse.rotate(angular_momentum)))


class Environment():
	def __init__(self, entities, constraints=[], external_force_generators=[]):
		self.entities = entities
		self.constraints = constraints
		for i, entity in enumerate(entities):
			entity.body_num = i
			entity.environment = self
		self.solution = None
		self.max_t = None

	def solve(self, t0, tf):
		def solvation(t, state):
			state_derivative = []
			for i, entity in enumerate(self.entities):
				pos, rot = state[13*i:13*i+3], Quaternion(state[13*i+3:13*i+7])
				mom, anm = state[13*i+7:13*i+10], state[13*i+10:13*i+13]
				velocity = mom/entity.m
				angularv = rot.rotate(np.matmul(entity.I_inv,rot.inverse.rotate(anm))) # make sure to use the correct ref frame when multiplying by I^-1
				state_derivative.extend([*velocity, *(Quaternion(0,*angularv)*rot/2), 0,0,0, 0,0,0])
			return state_derivative

		initial_state = []
		for e in self.entities:
			initial_state.extend(e.init_position)
			initial_state.extend(e.init_rotation)
			initial_state.extend(e.m*e.init_velocity)
			initial_state.extend(e.init_rotation.rotate(np.matmul(e.I,e.init_rotation.inverse.rotate(e.init_angularv))))
		self.solution = solve_ivp(solvation, [t0, tf], initial_state, dense_output=True)
		self.max_t = tf
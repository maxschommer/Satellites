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
	""" A collection of bodies, constraints, and external forces that manages them all together. """
	def __init__(self, bodies, constraints=[], external_impulsors=[]):
		""" bodies:				[RigidBody]		the list of bodies in the Universe
			constraints:		[Constraint]	the list of constraints between bodies to keep satisfied
			external_impulsors:	[Impulsor]		the list of environmental factors that interact with the system
		"""
		self.bodies = bodies
		self.constraints = constraints
		self.external_impulsors = external_impulsors
		for i, entity in enumerate(bodies):
			entity.body_num = i
			entity.environment = self
		self.solution = None
		self.max_t = None

	def solve(self, t0, tf):
		""" Solve the Universe and save the solution in self.solution.
			t0:	float	the time at which to start solving
			tf:	float	the final time about which we care
		"""
		def solvation(t, state):
			state_derivative = []
			for i, entity in enumerate(self.bodies):
				position, rotation = state[13*i:13*i+3], Quaternion(state[13*i+3:13*i+7])
				momentum, angularm = state[13*i+7:13*i+10], state[13*i+10:13*i+13]
				velocity = momentum/entity.m
				angularv = rotation.rotate(np.matmul(entity.I_inv,rotation.inverse.rotate(angularm))) # make sure to use the correct ref frame when multiplying by I^-1
				force, torque = np.zeros(3), np.zeros(3)
				for imp in self.external_impulsors:
					force += imp.force_on(entity, t, position, rotation, velocity, angularv)
					torque += imp.torque_on(entity, t, position, rotation, velocity, angularv)
				state_derivative.extend([*velocity, *(Quaternion(0,*angularv)*rotation/2), *force, *torque])
			return state_derivative

		initial_state = []
		for b in self.bodies:
			initial_state.extend(b.init_position)
			initial_state.extend(b.init_rotation)
			initial_state.extend(b.m*b.init_velocity)
			initial_state.extend(b.init_rotation.rotate(np.matmul(b.I,b.init_rotation.inverse.rotate(b.init_angularv))))
		self.solution = solve_ivp(solvation, [t0, tf], initial_state, dense_output=True)
		self.max_t = tf


class Impulsor():
	""" An entity that imparts linear and/or angular momentum to the system. """
	def __init__(self):
		pass

	def force_on(self, body, time, position, rotation, velocity, angularv):
		""" Compute the force applied by this Impulsor on the given body, given the time and that body's state.
			body:		RigidBody	the body on which this might be acting
			time:		float		the current time
			position:	3 vector	the current position of body
			rotation:	Quaternion	the current rotation of body
			velocity:	3 vector	the current linear velocity of body
			angularv:	3 vector	the current angular velocity of body
			return		3 vector	the applied force on body
		"""
		return np.zeros(3)

	def torque_on(self, body, time, position, rotation, velocity, angularv):
		""" Compute the force applied by this Impulsor on the given body, given the time and that body's state.
			body:		RigidBody	the body on which this might be acting
			time:		float		the current time
			position:	3 vector	the current position of body
			rotation:	Quaternion	the current rotation of body
			velocity:	3 vector	the current linear velocity of body
			angularv:	3 vector	the current angular velocity of body
			return		3 vector	the applied torque on body
		"""
		return np.zeros(3)


class MagneticDipole(Impulsor):
	""" External torque imparted by a uniform magnetic field on a magnetic body. """
	def __init__(self, body, dipole_moment, external_field):
		""" body:	RigidBody	the body on which the torque is applied
			dipole_moment:	3 vector	the magnetic dipole moment of the body in its reference frame
			external_field:	3 vector	the magnetic flux density of the environment
		"""
		self.body = body
		self.moment = dipole_moment
		self.field = external_field

	def torque_on(self, body, time, position, rotation, velocity, angularv):
		if body is self.body:
			return np.cross(rotation.rotate(self.moment), self.field)
		else:
			return np.zeros(3)

import matplotlib.pyplot as plt
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
		I_inv_rot = np.matmul(np.matmul(self.get_rotation(t).rotation_matrix,self.I_inv),self.get_rotation(t).rotation_matrix.transpose())
		angular_momentum = self.environment.solution.sol(t)[13*self.body_num+10:13*self.body_num+13]
		return np.matmul(I_inv_rot,angular_momentum)


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
		self.max_t = None # TODO: automatically add in displacements and impulses to initially satisfy constraints

	def solve(self, t0, tf):
		""" Solve the Universe and save the solution in self.solution.
			t0:	float	the time at which to start solving
			tf:	float	the final time about which we care
		"""
		def solvation(t, state):
			position, rotation, velocity, angularv, I_inv_rot = [], [], [], [], []
			for i, body in enumerate(self.bodies): # first unpack the state vector
				position.append(state[13*i:13*i+3])
				rotation.append(Quaternion(state[13*i+3:13*i+7]))
				I_inv_rot.append(np.matmul(np.matmul(rotation[i].rotation_matrix,body.I_inv),rotation[i].rotation_matrix.transpose()))
				velocity.append(state[13*i+7:13*i+10]/body.m)
				angularv.append(np.matmul(I_inv_rot[i], state[13*i+10:13*i+13])) # make sure to use the correct ref frame when multiplying by I^-1

			y = np.zeros((7*len(self.bodies))) # now, go through the bodies and get each's part of
			y_dot = np.zeros((7*len(self.bodies))) # the positional components of the state and its derivatives
			y_ddot = np.zeros((7*len(self.bodies)))
			state_dot = np.zeros((13*len(self.bodies))) # as well as the derivative of state, usable by the ODE solver
			for i, body in enumerate(self.bodies):
				force, torque = np.zeros(3), np.zeros(3) # incorporate external forces and moments
				for imp in self.external_impulsors:
					force += imp.force_on(body, t, position[i], rotation[i], velocity[i], angularv[i])
					torque += imp.torque_on(body, t, position[i], rotation[i], velocity[i], angularv[i])
				accelr8n, angulara = force/body.m, np.matmul(I_inv_rot[i], torque)

				angularv_q = 1/2*Quaternion(vector=angularv[i])*rotation[i] # put these vectors in quaternion form
				angulara_q = 1/2*(Quaternion(vector=angulara)*rotation[i] + Quaternion(vector=angularv[i])*angularv_q) # TODO: THIS DOES NOT ACCOUNT FOR THE TUMBLING TERM YET

				y[7*i:7*i+7] = np.concatenate((position[i], [*rotation[i]])) # finally, compile everything into the desired formats
				y_dot[7*i:7*i+7] = np.concatenate((velocity[i], [*angularv_q]))
				y_ddot[7*i:7*i+7] = np.concatenate((accelr8n, [*angulara_q])) # noting that y_ddot is incomplete without constraint forces
				state_dot[13*i:13*i+13] = np.concatenate((velocity[i], [*angularv_q], force, torque))

			num_constraints = sum([c.num_dof for c in self.constraints]) # Now account for constraints!
			J_c = np.zeros((num_constraints, 7*len(self.bodies))) # Sorry if my variable names in this part aren't the most descriptive.
			J_c_dot = np.zeros((num_constraints, 7*len(self.bodies))) # It's because I don't really understand what's going on
			R = np.zeros((7*len(self.bodies), num_constraints))
			j = 0
			for constraint in self.constraints:
				i_a, i_b = constraint.body_a.body_num, constraint.body_b.body_num
				prvω_a = position[i_a], rotation[i_a], velocity[i_a], angularv[i_a]
				prvω_b = position[i_b], rotation[i_b], velocity[i_b], angularv[i_b]
				J_c_j = constraint.constraint_jacobian(*prvω_a, *prvω_b)
				J_c_dot_j = constraint.constraint_derivative_jacobian(*prvω_a, *prvω_b)
				R_j = constraint.force_response(*prvω_a, *prvω_b)
				for k, i in [(0, i_a), (1, i_b)]:
					J_c[j:j+constraint.num_dof, 7*i:7*i+7] = J_c_j[:, 7*k:7*k+7]
					J_c_dot[j:j+constraint.num_dof, 7*i:7*i+7] = J_c_dot_j[:, 7*k:7*k+7]
					R[7*i:7*i+7, j:j+constraint.num_dof] = R_j[7*k:7*k+7, :]
				j += constraint.num_dof

			 # TODO: solve for constraints, add force and torque to state_dot

			return state_dot

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

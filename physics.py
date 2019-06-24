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
		for i, body in enumerate(bodies):
			body.body_num = i
			body.environment = self
		self.solution = None
		self.max_t = None # TODO: automatically add in displacements and impulses to initially satisfy constraints

	def solve(self, t0, tf):
		""" Solve the Universe and save the solution in self.solution.
			t0:	float	the time at which to start solving
			tf:	float	the final time about which we care
		"""
		def solvation(t, state):
			positions, rotations, velocitys, angularvs, I_inv_rots = [], [], [], [], []
			forces, torkes = [], []
			state_dot = [] # as well as the derivative of state, usable by the ODE solver
			for i, body in enumerate(self.bodies): # first unpack the state vector
				positions.append(state[13*i:13*i+3])
				rotations.append(Quaternion(state[13*i+3:13*i+7]))
				I_inv_rots.append(np.matmul(np.matmul(rotations[i].rotation_matrix,body.I_inv),rotations[i].rotation_matrix.transpose()))
				velocitys.append(state[13*i+7:13*i+10]/body.m)
				angularvs.append(np.matmul(I_inv_rots[i], state[13*i+10:13*i+13])) # make sure to use the correct ref frame when multiplying by I^-1
				forces.append(np.zeros(3))
				torkes.append(np.zeros(3))
				for imp in self.external_impulsors: # incorporate external forces and moments
					forces[i] += imp.force_on(body, t, positions[i], rotations[i], velocitys[i], angularvs[i])
					torkes[i] += imp.torke_on(body, t, positions[i], rotations[i], velocitys[i], angularvs[i])
				state_dot.extend([*velocitys[i], *(Quaternion(vector=angularvs[i])*rotations[i]/2), *forces[i], *torkes[i]]) # build the vector to return
			state_dot = np.array(state_dot)
			
			reaction_forces_torkes = self.solve_for_constraints(positions, rotations, velocitys, angularvs, I_inv_rots, forces, torkes) # now account for constraints
			for i, body in enumerate(self.bodies):
				state_dot[13*i+7:13*i+13] += reaction_forces_torkes[i]

			return state_dot # deliver solvation!

		initial_state = []
		for b in self.bodies:
			initial_state.extend(b.init_position)
			initial_state.extend(b.init_rotation)
			initial_state.extend(b.m*b.init_velocity)
			initial_state.extend(b.init_rotation.rotate(np.matmul(b.I,b.init_rotation.inverse.rotate(b.init_angularv))))
		self.solution = solve_ivp(solvation, [t0, tf], initial_state, dense_output=True)
		self.max_t = tf

	def solve_for_constraints(self, positions, rotations, velocitys, angularvs, I_invs, forces=None, torkes=None):
		if forces is None:	forces = [np.zeros(3)]*len(self.bodies)
		if torkes is None:	torkes = [np.zeros(3)]*len(self.bodies)
		# print("Solving for constraints...")

		print("Solving for constraints...")
		y_dot = np.zeros((7*len(self.bodies))) # now, go through the bodies and get each's part of
		y_ddot = np.zeros((7*len(self.bodies))) # the positional components of the state and its derivatives
		for i, body in enumerate(self.bodies):
			xlration, angulara = forces[i]/body.m, np.matmul(I_invs[i], torkes[i])

			angularv_q = 1/2*Quaternion(vector=angularvs[i])*rotations[i] # put these vectors in quaternion form
			angulara_q = 1/2*(Quaternion(vector=angulara)*rotations[i] + Quaternion(vector=angularvs[i])*angularv_q) # TODO: THIS DOES NOT ACCOUNT FOR THE TUMBLING TERM YET

			y_dot[7*i:7*i+7] = np.concatenate((velocitys[i], list(angularv_q)))
			y_ddot[7*i:7*i+7] = np.concatenate((xlration, list(angulara_q))) # this is the acceleration y would see without constraining forces
		print("  Current state derivative: {}".format(y_dot))
		print("  Planned state dderivative: {}".format(y_ddot))

		num_constraints = sum([c.num_dof for c in self.constraints]) # Now account for constraints!
		c = np.zeros((num_constraints))
		c_dot = np.zeros((num_constraints)) # TODO: these two are for debugging porpoises only
		J_c = np.zeros((num_constraints, 7*len(self.bodies))) # Sorry if my variable names in this part aren't the most descriptive.
		J_c_dot = np.zeros((num_constraints, 7*len(self.bodies))) # It's because I don't really understand what's going on.
		R = np.zeros((7*len(self.bodies), num_constraints))
		j = 0
		for constraint in self.constraints:
			i_a, i_b = constraint.body_a.body_num, constraint.body_b.body_num
			prvω_a = positions[i_a], rotations[i_a], velocitys[i_a], angularvs[i_a]
			prvω_b = positions[i_b], rotations[i_b], velocitys[i_b], angularvs[i_b]
			c_j = constraint.constraint_values(*prvω_a, *prvω_b)
			c_dot_j = constraint.constraint_derivative(*prvω_a, *prvω_b)
			J_c_j = constraint.constraint_jacobian(*prvω_a, *prvω_b)
			J_c_dot_j = constraint.constraint_derivative_jacobian(*prvω_a, *prvω_b)
			R_j = constraint.force_response(*prvω_a, *prvω_b)
			for k, i in [(0, i_a), (1, i_b)]:
				c[j:j+constraint.num_dof] = c_j
				c_dot[j:j+constraint.num_dof] = c_dot_j
				J_c[j:j+constraint.num_dof, 7*i:7*i+7] = J_c_j[:, 7*k:7*k+7]
				J_c_dot[j:j+constraint.num_dof, 7*i:7*i+7] = J_c_dot_j[:, 7*k:7*k+7]
				R[7*i:7*i+7, j:j+constraint.num_dof] = R_j[7*k:7*k+7, :]
			j += constraint.num_dof

		f = - np.matmul(np.linalg.inv(np.matmul(J_c, R)), np.matmul(J_c, y_ddot) + np.matmul(J_c_dot, y_dot)) # solve for the constraints!
		print("  Jacobian: {}".format(J_c))
		print("  Jacobian derivative: {}".format(J_c_dot))
		print("  Response: {}".format(R))
		print("  Constraint response: {}".format(np.matmul(J_c, R)))
		print("  Expected constraint derivative: {}".format(np.matmul(J_c, y_dot)))
		print("  Expected constraint dderivative: {}".format(np.matmul(J_c, y_ddot) + np.matmul(J_c_dot, y_dot)))
		print("  Computed constraint effects: {}".format(f))
		print("  Constraint dderivative, accounting for reaction forces: {}".format(np.matmul(J_c, y_ddot+np.matmul(R,f)) + np.matmul(J_c_dot, y_dot)))

		reaction_forces_torkes = [np.zeros(6) for body in self.bodies]
		for constraint in self.constraints: # apply the constraints
			i_a, i_b = constraint.body_a.body_num, constraint.body_b.body_num
			prvω_a = positions[i_a], rotations[i_a], velocitys[i_a], angularvs[i_a]
			prvω_b = positions[i_b], rotations[i_b], velocitys[i_b], angularvs[i_b]
			reaction_forces_torkes[i_a] += constraint.force_torke_on_a(f[:constraint.num_dof], *prvω_a, *prvω_b)
			reaction_forces_torkes[i_b] += constraint.force_torke_on_b(f[:constraint.num_dof], *prvω_a, *prvω_b)
			f = f[constraint.num_dof:]
		print("  Computed forces and torkes: {}".format(reaction_forces_torkes))

		return reaction_forces_torkes


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

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		""" Compute the force applied by this Impulsor on the given body, given the time and that body's state.
			body:		RigidBody	the body on which this might be acting
			time:		float		the current time
			position:	3 vector	the current position of body
			rotation:	Quaternion	the current rotation of body
			velocity:	3 vector	the current linear velocity of body
			angularv:	3 vector	the current angular velocity of body
			return		3 vector	the applied torke on body
		"""
		return np.zeros(3)


class MagneticDipole(Impulsor):
	""" External torke imparted by a uniform magnetic field on a magnetic body. """
	def __init__(self, body, dipole_moment, external_field):
		""" body:	RigidBody	the body on which the torke is applied
			dipole_moment:	3 vector	the magnetic dipole moment of the body in its reference frame
			external_field:	3 vector	the magnetic flux density of the environment
		"""
		self.body = body
		self.moment = dipole_moment
		self.field = external_field

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		if body is self.body:
			return np.cross(rotation.rotate(self.moment), self.field)
		else:
			return np.zeros(3)

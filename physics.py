import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion

import constraint


CONSTRAINT_RECOVERY_TIME = .1 # making this smaller increases the precision with which the constraints are met, but increases the computation time


class RigidBody():
	""" A physical unbending object free to move and rotate in space """
	def __init__(self, I, m, cm_position, init_position=[0,0,0], init_rotation=[1,0,0,0], init_velocity=[0,0,0], init_angularv=[0,0,0]):
		""" I:		      3x3 float array		rotational inertia
			m:		  	  float				    mass
			cm_position:  3 float vector		centre of mass in the mesh coordinate frame
		"""
		self.I = I
		self.I_inv = np.linalg.inv(I)
		self.m = m
		self.cm_position = np.array(cm_position)
		self.body_num = None
		self.environment = None

		self.init_position = np.array(init_position, dtype=np.float)
		self.init_rotation = Quaternion(init_rotation, dtype=np.float)
		self.init_velocity = np.array(init_velocity, dtype=np.float)
		self.init_angularv = np.array(init_angularv, dtype=np.float)

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
		self.max_t = None

		positions, rotations, I_inv_rots = [], [], [] # I would also like to automatically deal with initial velocities that violate constraints
		momentums, angularms = [], [] # luckily, the linearity of time derivatives is such that
		for i, body in enumerate(self.bodies): # I can reuse a bunch of my regular constraint-solving code
			positions.append(body.init_position)
			rotations.append(body.init_rotation)
			I_inv_rots.append(np.matmul(np.matmul(rotations[i].rotation_matrix,body.I_inv),rotations[i].rotation_matrix.transpose()))
			momentums.append(body.m*body.init_velocity)
			angularms.append(body.init_rotation.rotate(np.matmul(body.I,body.init_rotation.inverse.rotate(body.init_angularv)))) # make sure to use the correct ref frame when multiplying by I^-1
		
		reaction_impulses = self.solve_for_constraints(positions, rotations, 0, 0, I_inv_rots, momentums, angularms) # now account for constraints
		for i, body in enumerate(self.bodies):
			body.init_velocity += 1/body.m*reaction_impulses[i][0:3]
			body.init_angularv += np.matmul(I_inv_rots[i], reaction_impulses[i][3:6])
		
	def solve(self, t0, tf):
		""" Solve the Universe and save the solution in self.solution.
			t0:	float	the time at which to start solving
			tf:	float	the final time about which we care
		"""
		def solvation(t, state):
			positions, rotations, velocitys, angularvs = [], [], [], []
			momentums, angularms, I_inv_rots = [], [], []
			forces, torkes = [], []
			state_dot = [] # as well as the derivative of state, usable by the ODE solver
			for i, body in enumerate(self.bodies): # first unpack the state vector
				positions.append(state[13*i:13*i+3])
				rotations.append(Quaternion(state[13*i+3:13*i+7]))
				momentums.append(state[13*i+7:13*i+10])
				angularms.append(state[13*i+10:13*i+13])
				I_inv_rots.append(np.matmul(np.matmul(rotations[i].rotation_matrix,body.I_inv),rotations[i].rotation_matrix.transpose()))
				velocitys.append(momentums[i]/body.m)
				angularvs.append(np.matmul(I_inv_rots[i], angularms[i])) # make sure to use the correct ref frame when multiplying by I^-1
				forces.append(np.zeros(3))
				torkes.append(np.zeros(3))
				for imp in self.external_impulsors: # incorporate external forces and moments
					forces[i] += imp.force_on(body, t, positions[i], rotations[i], velocitys[i], angularvs[i])
					torkes[i] += imp.torke_on(body, t, positions[i], rotations[i], velocitys[i], angularvs[i])
				state_dot.extend([*velocitys[i], *(Quaternion(vector=angularvs[i])*rotations[i]/2), *forces[i], *torkes[i]]) # build the vector to return
			state_dot = np.array(state_dot)
			
			reaction_forces_torkes = self.solve_for_constraints(positions, rotations, velocitys, angularvs, I_inv_rots, forces, torkes) # now account for constraints
			correction_impulses = self.solve_for_constraints(positions, rotations, 0, 0, I_inv_rots, momentums, angularms) # with a velocity-response term as well
			for i, body in enumerate(self.bodies):
				state_dot[13*i+7:13*i+13] += reaction_forces_torkes[i] + correction_impulses[i]/CONSTRAINT_RECOVERY_TIME

			return state_dot # deliver solvation!

		initial_state = []
		for b in self.bodies:
			initial_state.extend(b.init_position)
			initial_state.extend(b.init_rotation)
			initial_state.extend(b.m*b.init_velocity)
			initial_state.extend(b.init_rotation.rotate(np.matmul(b.I,b.init_rotation.inverse.rotate(b.init_angularv))))
		self.solution = solve_ivp(solvation, [t0, tf], initial_state, dense_output=True)
		self.max_t = tf

	def solve_for_constraints(self, positions, rotations, velocitys, angularvs, I_inv_rots, forces, torkes):
		if len(self.constraints) == 0:	return [np.zeros(6) for body in self.bodies]

		if velocitys is 0:	velocitys = [np.zeros(3)]*len(self.bodies)
		if angularvs is 0:	angularvs = [np.zeros(3)]*len(self.bodies)

		y_dot = np.zeros((7*len(self.bodies))) # now, go through the bodies and get each's part of
		y_ddot = np.zeros((7*len(self.bodies))) # the positional components of the state and its derivatives
		for i, body in enumerate(self.bodies):
			xlration, angulara = forces[i]/body.m, np.matmul(I_inv_rots[i], torkes[i]) # TODO: THIS DOES NOT ACCOUNT FOR THE TUMBLING TERM YET

			angularv_q = 1/2*Quaternion(vector=angularvs[i])*rotations[i] # put these vectors in quaternion form
			angulara_q = 1/2*(Quaternion(vector=angulara) - np.dot(angularvs[i],angularvs[i])/2)*rotations[i]

			y_dot[7*i:7*i+7] = np.concatenate((velocitys[i], list(angularv_q)))
			y_ddot[7*i:7*i+7] = np.concatenate((xlration, list(angulara_q))) # this is the acceleration y would see without constraining forces

		num_constraints = sum([c.num_dof for c in self.constraints]) # Now account for constraints!
		J_c = np.zeros((num_constraints, 7*len(self.bodies))) # Sorry if my variable names in this part aren't the most descriptive.
		J_c_dot = np.zeros((num_constraints, 7*len(self.bodies))) # It's because I don't really understand what's going on.
		R = np.zeros((7*len(self.bodies), num_constraints))
		j = 0
		for constraint in self.constraints:
			i_a, i_b = constraint.body_a.body_num, constraint.body_b.body_num
			prvω_a = positions[i_a], rotations[i_a], velocitys[i_a], angularvs[i_a]
			prvω_b = positions[i_b], rotations[i_b], velocitys[i_b], angularvs[i_b]
			J_c_j = constraint.constraint_jacobian(*prvω_a, *prvω_b)
			J_c_dot_j = constraint.constraint_derivative_jacobian(*prvω_a, *prvω_b)
			R_j = constraint.response(*prvω_a, *prvω_b, I_inv_rot_a=I_inv_rots[i_a], I_inv_rot_b=I_inv_rots[i_b])
			for k, i in [(0, i_a), (1, i_b)]:
				J_c[j:j+constraint.num_dof, 7*i:7*i+7] = J_c_j[:, 7*k:7*k+7]
				J_c_dot[j:j+constraint.num_dof, 7*i:7*i+7] = J_c_dot_j[:, 7*k:7*k+7]
				R[7*i:7*i+7, j:j+constraint.num_dof] = R_j[7*k:7*k+7, :]
			j += constraint.num_dof

		f = - np.matmul(np.linalg.inv(np.matmul(J_c, R)), np.matmul(J_c, y_ddot) + np.matmul(J_c_dot, y_dot)) # solve for the constraints!

		reaction_forces_torkes = [np.zeros(6) for body in self.bodies]
		for constraint in self.constraints: # apply the constraints
			i_a, i_b = constraint.body_a.body_num, constraint.body_b.body_num
			prvω_a = positions[i_a], rotations[i_a], velocitys[i_a], angularvs[i_a]
			prvω_b = positions[i_b], rotations[i_b], velocitys[i_b], angularvs[i_b]
			reaction_forces_torkes[i_a] += constraint.force_torke_on_a(f[:constraint.num_dof], *prvω_a, *prvω_b)
			reaction_forces_torkes[i_b] += constraint.force_torke_on_b(f[:constraint.num_dof], *prvω_a, *prvω_b)
			f = f[constraint.num_dof:]

		return reaction_forces_torkes

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution
from pyquaternion import Quaternion

import constraint


CONSTRAINT_RECOVERY_TIME = .03 # making this smaller increases the precision with which the constraints are met, but increases the computation time


class Environment():
	""" A collection of bodies, constraints, and external forces that manages them all together. """
	def __init__(self, bodies, constraints=[], sensors=[], external_impulsors=[], events=[],
			magnetic_field=[0,0,0], solar_flux=[0,0,0], air_velocity=[0,0,0], air_density=0):
		""" bodies:				[RigidBody]		the list of bodies in the Universe
			constraints:		[Constraint]	the list of constraints between bodies to keep satisfied
			sensors:			[Sensor]		the list of sensors that are entitled to certain information
			external_impulsors:	[Impulsor]		the list of environmental factors that interact with the system
		"""
		self.bodies = bodies
		for i, body in enumerate(bodies):
			body.body_num = i
			body.environment = self
		self.constraints = constraints
		self.sensors = sensors
		for sen in sensors:
			sen.environment = self
		self.external_impulsors = external_impulsors
		for imp in external_impulsors:
			imp.environment = self
		self.events = events

		self.magnetic_field = magnetic_field if hasattr(magnetic_field, 'get_value') else np.array(magnetic_field)
		self.solar_flux     = solar_flux if hasattr(solar_flux, 'get_value') else np.array(solar_flux)
		self.air_velocity   = air_velocity if hasattr(air_velocity, 'get_value') else np.array(air_velocity)
		self.air_density = air_density

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
		t = t0 # start at time t0
		for body in self.bodies: # put all of the init_ variables into more generic ones
			body.position, body.rotation = body.init_position, body.init_rotation
			body.momentum = body.m*body.init_velocity # these state field will be used at every breakpoint
			body.angularm = body.rotation.rotate(np.matmul(body.I,body.rotation.inverse.rotate(body.init_angularv)))

		full_ts, full_interpolants = [], []
		while True: # now repeatedly
			initial_state = [] # build a new "initial" state for the solver
			for body in self.bodies:
				if body.active:
					initial_state.extend([*body.position, *body.rotation, *body.momentum, *body.angularm])
				else:
					initial_state.extend([-9000,0,0, 1,0,0,0, 0,0,0, 0,0,0])
			step_solution = solve_ivp(self.ode_func, [t, tf], initial_state, dense_output=True, events=self.events) # solve
			full_ts.extend(step_solution.sol.ts[:-1]) # save the results to our full solution
			full_interpolants.extend(step_solution.sol.interpolants)
			final_state = step_solution.y[:,-1] # unpack the "final" state
			for body in self.bodies:
				body.position, body.rotation = final_state[0:3], Quaternion(final_state[3:7])
				body.momentum, body.angularm = final_state[7:10], final_state[10:13]
				final_state = final_state[13:]
			t = step_solution.t[-1]

			if step_solution.status == 1: # if an event was hit
				for i, event in enumerate(self.events):
					if event.terminal and len(step_solution.t_events[i]) > 0:
						triggered_event = event # find that event
						break
				triggered_event.happen() # and activate its effects
				self.events.remove(triggered_event) # each event only happens once
			else: # if the end of the tspan or an error was hit
				break # end this cycle of death

		self.solution = OdeSolution(full_ts+[t], full_interpolants)
		self.max_t = tf

	def ode_func(self, t, state):
		""" The state evolution vector to pass into Python's ODE solver. """
		positions, rotations, velocitys, angularvs = [], [], [], []
		momentums, angularms, I_inv_rots = [], [], []
		state_dot = [] # as well as the derivative of state, usable by the ODE solver
		for i, body in enumerate(self.bodies): # first unpack the state vector
			positions.append(state[13*i:13*i+3])
			rotations.append(Quaternion(state[13*i+3:13*i+7]))
			momentums.append(state[13*i+7:13*i+10])
			angularms.append(state[13*i+10:13*i+13])
			I_inv_rots.append(np.matmul(np.matmul(rotations[i].rotation_matrix,body.I_inv),rotations[i].rotation_matrix.transpose()))
			velocitys.append(momentums[i]/body.m)
			angularvs.append(np.matmul(I_inv_rots[i], angularms[i])) # make sure to use the correct ref frame when multiplying by I^-1

		for sen in self.sensors: # feed those results to all sensors
			sen.sense(t, positions, rotations, velocitys, angularvs)

		forces, torkes = [], []
		for i, body in enumerate(self.bodies): # then resolve the effects of external forces and torkes
			forces.append(np.zeros(3))
			torkes.append(np.zeros(3))
			for imp in self.external_impulsors:
				forces[i] += imp.force_on(body, t, positions[i], rotations[i], velocitys[i], angularvs[i])
				torkes[i] += imp.torke_on(body, t, positions[i], rotations[i], velocitys[i], angularvs[i])
			state_dot.extend([*velocitys[i], *(Quaternion(vector=angularvs[i])*rotations[i]/2), *forces[i], *torkes[i]]) # build the vector to return
		state_dot = np.array(state_dot)
		
		reaction_forces_torkes = self.solve_for_constraints(positions, rotations, velocitys, angularvs, I_inv_rots, forces, torkes) # now account for constraints
		correction_impulses = self.solve_for_constraints(positions, rotations, 0, 0, I_inv_rots, momentums, angularms) # with a velocity-response term as well
		for i, body in enumerate(self.bodies):
			state_dot[13*i+7:13*i+13] += reaction_forces_torkes[i] + correction_impulses[i]/CONSTRAINT_RECOVERY_TIME

		return state_dot # deliver solvation!

	def solve_for_constraints(self, positions, rotations, velocitys, angularvs, I_inv_rots, forces, torkes):
		if len(self.constraints) == 0:	return [np.zeros(6) for body in self.bodies]

		if velocitys is 0:	velocitys = [np.zeros(3)]*len(self.bodies)
		if angularvs is 0:	angularvs = [np.zeros(3)]*len(self.bodies)

		y_dot = np.empty((7*len(self.bodies))) # now, go through the bodies and get each's part of
		y_ddot = np.empty((7*len(self.bodies))) # the positional components of the state and its derivatives
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

	def get_magnetic_field(self, t):
		if hasattr(self.magnetic_field, 'get_value'):
			return self.magnetic_field.get_value(t)
		else:
			return self.magnetic_field

	def get_solar_flux(self, t):
		if hasattr(self.solar_flux, 'get_value'):
			return self.solar_flux.get_value(t)
		else:
			return self.solar_flux

	def get_air_velocity(self, t):
		if hasattr(self.air_velocity, 'get_value'):
			return self.air_velocity.get_value(t)
		else:
			return self.air_velocity

	def get_air_density(self, t):
		if hasattr(self.air_density, 'get_value'):
			return self.air_density.get_value(t)
		else:
			print(e)
			return self.air_density

	def shell(self):
		""" Strip away all of the things that don't fit in the pickle jar. """
		self.constraints = None
		self.external_impulsors = None
		self.events = None


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

		self.active = True # an inactive body will neither solve nor render

	def get_position(self, t):
		return self.environment.solution(t)[13*self.body_num:13*self.body_num+3]

	def get_rotation(self, t):
		return Quaternion(self.environment.solution(t)[13*self.body_num+3:13*self.body_num+7])

	def get_velocity(self, t):
		momentum = self.environment.solution(t)[13*self.body_num+7:13*self.body_num+10]
		return momentum/self.m

	def get_angularv(self, t):
		I_inv_rot = np.matmul(np.matmul(self.get_rotation(t).rotation_matrix,self.I_inv),self.get_rotation(t).rotation_matrix.transpose())
		angular_momentum = self.environment.solution(t)[13*self.body_num+10:13*self.body_num+13]
		return np.matmul(I_inv_rot,angular_momentum)

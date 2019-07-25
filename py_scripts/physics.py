import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion

import constraint


CONSTRAINT_RECOVERY_TIME = .03 # making this smaller increases the precision with which the constraints are met, but increases the computation time


class Environment():
	""" A collection of bodies, constraints, and external forces that manages them all together. """
	def __init__(self, bodies, sensors={}, constraints=[], external_impulsors=[], events=[],
			magnetic_field=[0,0,0], solar_flux=[0,0,0], air_velocity=[0,0,0], air_density=0):
		""" bodies:				{str:RigidBody}		the dict of bodies in the Universe
			sensors:			{str:Sensor}		the dict of sensors that are entitled to certain information
			constraints:		[Constraint]		the list of constraints between bodies to keep satisfied
			external_impulsors:	[Impulsor]			the list of environmental factors that interact with the system
			events:				[Event]				the list of events that might require the simulation to break
			magnetic_field:		3 vector or Table	the value of the Earth's B field in T
			solar_flux:			3 vector or Table	the value of the solar constant in W/m^2
			air_velocity:		3 vector or Table	the value of the atmosphere's relative velocity in m/s
			air_density:		3 vector or Table	the value of the atmospheric density in kg/m^3
		"""
		self.bodies = bodies
		for i, (key, body) in enumerate(bodies.items()):
			body.id = key
			body.num = i
			body.environment = self
		self.sensors = sensors
		for key, sen in sensors.items():
			sen.environment = self
		self.constraints = constraints
		self.external_impulsors = external_impulsors
		for imp in external_impulsors:
			imp.environment = self
		self.events = events

		self.magnetic_field = magnetic_field if hasattr(magnetic_field, 'get_value') else np.array(magnetic_field)
		self.solar_flux     = solar_flux if hasattr(solar_flux, 'get_value') else np.array(solar_flux)
		self.air_velocity   = air_velocity if hasattr(air_velocity, 'get_value') else np.array(air_velocity)
		self.air_density    = air_density

		self.solution = None
		self.max_t = None

		positions, rotations, I_inv_rots = [], [], [] # I would like to automatically deal with initial velocities that violate constraints
		momentums, angularms = [], [] # luckily, the linearity of time derivatives is such that
		for body in self.bodies.values(): # I can reuse a bunch of my regular constraint-solving code
			positions.append(body.init_position)
			rotations.append(body.init_rotation)
			I_inv_rots.append(np.matmul(np.matmul(rotations[-1].rotation_matrix,body.I_inv),rotations[-1].rotation_matrix.transpose()))
			momentums.append(body.m*body.init_velocity)
			angularms.append(body.init_rotation.rotate(np.matmul(body.I,body.init_rotation.inverse.rotate(body.init_angularv)))) # make sure to use the correct ref frame when multiplying by I^-1
		reaction_impulses = self.solve_for_constraints(positions, rotations, 0, 0, I_inv_rots, momentums, angularms) # now account for constraints
		for body in self.bodies.values():
			body.init_velocity += 1/body.m*reaction_impulses[body.num][0:3]
			body.init_angularv += np.matmul(I_inv_rots[-1], reaction_impulses[-1][3:6])
		
	def solve(self, t0, tf, method='RK54', num_data=20000):
		""" Solve the Universe and save the solution in self.solution.
			t0:			float	the time at which to start solving
			tf:			float	the final time about which we care
			method:		str		the code for the method to pass to solve_ivp
			num_data:	int		the number of times at which to store the value of the solution
		"""
		t = t0 # start at time t0
		for body in self.bodies.values(): # put all of the init_ variables into more generic ones
			body.position, body.rotation = body.init_position, body.init_rotation
			body.momentum = body.m*body.init_velocity # these state field will be used at every breakpoint
			body.angularm = body.rotation.rotate(np.matmul(body.I,body.rotation.inverse.rotate(body.init_angularv)))

		full_ts, full_ys = [], []
		while True: # now repeatedly
			initial_state = [] # build a new "initial" state for the solver
			for body in self.bodies.values():
				if body.active:
					initial_state.extend([*body.position, *body.rotation, *body.momentum, *body.angularm])
				else:
					initial_state.extend([-9000,0,0, 1,0,0,0, 0,0,0, 0,0,0]) # hide the inactive bodies at -9000
			step_solution = solve_ivp(self.ode_func, [t, tf], initial_state, t_eval=np.linspace(t, tf, num_data), events=self.events, method=method, rtol=1e-6) # solve
			full_ts.extend(step_solution.t[:-1]) # save the results to our full solution
			full_ys.extend(step_solution.y[:,:-1].transpose())
			t = step_solution.t[-1]
			y = step_solution.y[:,-1] # unpack the "final" state
			for i, body in enumerate(self.bodies.values()):
				body.position, body.rotation = y[13*i+0:13*i+3], Quaternion(y[13*i+3:13*i+7])
				body.momentum, body.angularm = y[13*i+7:13*i+10], y[13*i+10:13*i+13]

			if step_solution.status == 1: # if an event was hit
				for i, event in enumerate(self.events):
					if event.terminal and len(step_solution.t_events[i]) > 0:
						triggered_event = event # find that event
						break
				triggered_event.happen() # and activate its effects
				self.events.remove(triggered_event) # each event only happens once
			else: # if the end of the tspan or an error was hit
				break # end this cycle of death

		self.solution = (np.array(full_ts+[t]), np.array(full_ys+[y]))
		self.max_t = tf

	def ode_func(self, t, state):
		""" The state evolution vector to pass into Python's ODE solver. """
		positions, rotations, velocitys, angularvs = [], [], [], []
		momentums, angularms, I_inv_rots = [], [], []
		angularv_qs = [] # this one is the derivative of the rotation in quaternion space
		state_dot = [] # as well as the derivative of state, usable by the ODE solver
		for body in self.bodies.values(): # first unpack the state vector
			idx = 13*body.num
			positions.append(state[idx+0:idx+3]) # into a group of lists
			rotations.append(Quaternion(state[idx+3:idx+7]))
			momentums.append(state[idx+7:idx+10])
			angularms.append(state[idx+10:idx+13])
			rotation_matrix = rotations[-1].normalised.rotation_matrix
			I_inv_rots.append(np.matmul(np.matmul(rotation_matrix, body.I_inv), rotation_matrix.transpose()))
			velocitys.append(momentums[-1]/body.m)
			angularvs.append(np.matmul(I_inv_rots[-1], angularms[-1])) # make sure to use the correct ref frame when multiplying by I^-1
			angularv_qs.append(Quaternion(scalar=(1-rotations[-1].magnitude)/10, vector=angularvs[-1])*rotations[-1]/2)

		for sen in self.sensors.values(): # feed those results to all sensors
			sen.sense(t, positions, rotations, velocitys, angularvs)

		forces, torkes = [], []
		for i in range(len(self.bodies)): # then resolve the effects of external forces and torkes
			forces.append(np.zeros(3))
			torkes.append(np.zeros(3))
			for imp in self.external_impulsors:
				forces[i] += imp.force_on(i, t, positions[i], rotations[i], velocitys[i], angularvs[i])
				torkes[i] += imp.torke_on(i, t, positions[i], rotations[i], velocitys[i], angularvs[i])
			state_dot.extend([*velocitys[i], *angularv_qs[i], *forces[i], *torkes[i]]) # build the vector to return
		state_dot = np.array(state_dot)
		
		reaction_forces_torkes = self.solve_for_constraints(positions, rotations, velocitys, angularvs, I_inv_rots, forces, torkes) # now account for constraints
		correction_impulses = self.solve_for_constraints(positions, rotations, 0, 0, I_inv_rots, momentums, angularms) # with a velocity-response term as well
		for i in range(len(self.bodies)):
			state_dot[13*i+7:13*i+13] += reaction_forces_torkes[i] + correction_impulses[i]/CONSTRAINT_RECOVERY_TIME

		return state_dot # deliver solvation!

	def solve_for_constraints(self, positions, rotations, velocitys, angularvs, I_inv_rots, forces, torkes):
		if len(self.constraints) == 0:	return [np.zeros(6) for body in self.bodies]

		if velocitys is 0:	velocitys = [np.zeros(3)]*len(self.bodies)
		if angularvs is 0:	angularvs = [np.zeros(3)]*len(self.bodies)

		y_dot = np.empty((7*len(self.bodies))) # now, go through the bodies and get each's part of
		y_ddot = np.empty((7*len(self.bodies))) # the positional components of the state and its derivatives
		for body in self.bodies.values():
			i = body.num
			xlration, angulara = forces[i]/body.m, np.matmul(I_inv_rots[i], torkes[i]) # TODO: Is there a tumbling term here?
			angularv_q = 1/2*Quaternion(vector=angularvs[i])*rotations[i] # put these vectors in quaternion form
			angulara_q = 1/2*(Quaternion(vector=angulara) - np.dot(angularvs[i],angularvs[i])/2)*rotations[i]
			y_dot[7*i:7*i+7] = np.concatenate((velocitys[i], list(angularv_q)))
			y_ddot[7*i:7*i+7] = np.concatenate((xlration, list(angulara_q))) # this is the acceleration y would see without constraining forces

		num_constraints = sum(c.num_dof for c in self.constraints) # Now account for constraints!
		J_c = np.zeros((num_constraints, 7*len(self.bodies))) # Sorry if my variable names in this part aren't the most descriptive.
		J_c_dot = np.zeros((num_constraints, 7*len(self.bodies))) # It's because I don't really understand what's going on.
		R = np.zeros((7*len(self.bodies), num_constraints))
		j = 0
		for constraint in self.constraints:
			i_a, i_b = constraint.body_a.num, constraint.body_b.num
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

		reaction_forces_torkes = [np.zeros(6) for b in self.bodies]
		for constraint in self.constraints: # apply the constraints
			i_a, i_b = constraint.body_a.num, constraint.body_b.num
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
			return self.air_density

	def global_cm(self, t):
		cm = np.zeros(3)
		for body in self.bodies.values():
			cm += body.get_position(t)
		return cm

	def shell(self):
		""" Strip away all of the things that don't fit in the pickle jar. """
		self.constraints = None
		self.external_impulsors = None
		self.events = None

	def reduce(self):
		""" Approximate the solution to make it fit in a jar. """
		self.solution = (sol_component[::2] for sol_component in self.solution)


class RigidBody():
	""" A physical unbending object free to move and rotate in space """
	def __init__(self, m, cm_position, tensor=None, axes=None, moments=None,
			init_position=[0,0,0], init_rotation=[1,0,0,0], init_velocity=[0,0,0], init_angularv=[0,0,0]):
		""" m:				float				mass
			cm_position:	3 float vector		centre of mass in the mesh coordinate frame
			tensor:			3x3 float matrix	rotational inertia tensor (axes and moments will be ignored if this is present)
			axes:			[3 float vector]	principal axes of inertia (moments must be present if this is)
			moments:		[float]				principal moments of inertia (axes must be present if this is)
		"""
		self.m = m
		self.cm_position = np.array(cm_position)

		if tensor is not None: # if a tensor is provided
			self.I = np.array(tensor) # take it
			self.I_inv = np.linalg.inv(self.I)
		else: # if values and axes are provided
			basis = np.array(axes).transpose()
			values = np.identity(3)/moments # use them to construct the inverse
			self.I_inv = np.matmul(np.matmul(basis, values), np.linalg.inv(basis))
			self.I_inv = (self.I_inv + self.I_inv.transpose())/2
			self.I = np.linalg.inv(self.I_inv)

		self.init_position = np.array(init_position, dtype=np.float)
		self.init_rotation = Quaternion(init_rotation, dtype=np.float)
		self.init_velocity = np.array(init_velocity, dtype=np.float)
		self.init_angularv = np.array(init_angularv, dtype=np.float)

		self.num = None
		self.environment = None
		self.active = True # an inactive body will neither solve nor render

	def get_position(self, t):
		tp, yp = self.environment.solution
		return np.array([np.interp(t, tp, yp[:,13*self.num+i]) for i in range(0, 3)])

	def get_rotation(self, t):
		tp, yp = self.environment.solution
		return Quaternion([np.interp(t, tp, yp[:,13*self.num+i]) for i in range(3, 7)])

	def get_velocity(self, t):
		tp, yp = self.environment.solution
		momentum = np.array([np.interp(t, tp, yp[:,13*self.num+i]) for i in range(7, 10)])
		return momentum/self.m

	def get_angularv(self, t):
		tp, yp = self.environment.solution
		I_inv_rot = np.matmul(np.matmul(self.get_rotation(t).rotation_matrix,self.I_inv),self.get_rotation(t).rotation_matrix.transpose())
		angularm = np.array([np.interp(t, tp, yp[:,13*self.num+i]) for i in range(10, 13)])
		return np.matmul(I_inv_rot, angularm)

	def __eq__(self, other):
		return self.num == other or (hasattr(other,'num') and self.num == other.num)

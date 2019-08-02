import numpy as np


class Impulsor():
	""" An entity that imparts linear and/or angular momentum to the system. """
	def __init__(self):
		self.environment = None

	def force_on(self, body, time, position, rotation, velocity, angularv):
		""" Compute the force applied by this Impulsor on the given body, given the time and that body's state.
			body:		int			the id of the body on which this might be acting
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
			body:		int			the id of the body body on which this might be acting
			time:		float		the current time
			position:	3 vector	the current position of body
			rotation:	Quaternion	the current rotation of body
			velocity:	3 vector	the current linear velocity of body
			angularv:	3 vector	the current angular velocity of body
			return		3 vector	the applied torke on body
		"""
		return np.zeros(3)


class Magnetorker(Impulsor):
	""" External torke imparted by a uniform magnetic field on a 3-axis variable-magnitude dipole. """
	def __init__(self, body, moment):
		""" body:		RigidBody			the body on which this is mounted
			moment:		(float) -> float	the magnetic dipole moment vector at a given time
		"""
		self.body = body
		self.moment = moment

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		if body == self.body:
			return np.cross(
				rotation.rotate(self.moment(time)), self.environment.get_magnetic_field(time))
		else:
			return np.zeros(3)


class PermanentMagnet(Magnetorker):
	""" External torke imparted by a uniform magnetic field on a magnetic dipole. """
	def __init__(self, body, dipole_moment):
		""" body:			RigidBody			the body on which the torke is applied
			dipole_moment:	3 vector			the magnetic dipole moment of the body in its reference frame
		"""
		super().__init__(body, lambda t: dipole_moment)


class GimballedThruster(Impulsor):
	""" External force imparted by a variable-magnitude variable-direction thrust. """
	def __init__(self, body, lever_arm, thrust):
		""" body:		RigidBody			the body on which the thruster is mounted
			lever_arm:	3 vector			the thruster's position on the body, in the body frame
			thrust:		(float) -> 3 vector	the thrust vector at a given time, in the body frame
		"""
		self.body = body
		self.lever_arm = lever_arm - self.body.cm_position
		self.thrust = thrust

	def force_on(self, body, time, position, rotation, velocity, angularv):
		if body == self.body:
			return rotation.rotate(self.thrust(time))
		else:
			return np.zeros(3)

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		if body == self.body:
			return rotation.rotate(np.cross(self.lever_arm, self.thrust(time)))
		else:
			return np.zeros(3)


class Thruster(GimballedThruster):
	""" External force imparted by a variable-magnitude thrust. """
	def __init__(self, body, lever_arm, direction, magnitude):
		""" body:		RigidBody			the body on which the thruster is mounted
			lever_arm:	3 vector			the thruster's position on the body, in the body frame
			direction:	3 vector			the direction the thruster points in the body frame
			magnitude:	(float) -> float	the thrust magnitude at a given time
		"""
		self.direction = np.array(direction, dtype=float)/np.linalg.norm(direction)
		super().__init__(body, lever_arm, lambda t: self.direction*magnitude(t))


class Drag(Impulsor):
	""" External force and torke imparted by collision with the atmosphere. """
	def __init__(self, bodies, areas, cp_positions):
		""" bodies:			{str:RigidBody}	the set of all bodies on which this will act
			areas:			[3 vector]		the effective area of the body on each axis with the drag coefficients multiplied in
			cp_positions:	[3 vector]		the position of the centre of pressure in the body frame
		"""
		self.areas = [np.ones(3)*area for area in areas] # the area matrices are to be multiplied by the wind direction
		self.cp_positions = [cp_position - b.cm_position for cp_position, b in zip(cp_positions, bodies.values())]

	def force_on(self, body, time, position, rotation, velocity, angularv):
		wind_velocity = self.environment.get_air_velocity(time)
		area_vec = self.areas[body if type(body) is int else body.num]
		v_times_a = np.sum(np.abs(wind_velocity*rotation.rotate(area_vec))) # effective area times velocity magnitude
		return 1/2*v_times_a*self.environment.get_air_density(time)*wind_velocity

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		cp_position = self.cp_positions[body if type(body) is int else body.num] # this torke is pretty straightforward
		r_cross_f_torke = np.cross(rotation.rotate(cp_position), self.force_on(body, time, position, rotation, velocity, angularv))
		# area_vec = self.areas[body if type(body) is int else body.num] # this one is rough and super small and I actually just removed it to make it run faster
		# area_eff = np.sum(area_vec - np.abs(normalized(angularv)*rotation.rotate(area_vec))) # effective area times velocity magnitude
		ω_squared_torke = 0#-1/2*area_eff**(2.5)*self.environment.get_air_density(time)*np.linalg.norm(angularv)*angularv
		return r_cross_f_torke + ω_squared_torke # combine torque due to offset center of pressure and against rotation of body


class Parachute(Impulsor):
	""" External force and torke imparted by a massless tethered sphere. """
	def __init__(self, body, area, position):
		""" body:		RigidBody	the set of all bodies on which this will act
			area:		float		the effective area of the dragging object with the drag coefficients multiplied in
			position:	[3 vector]	the position of the centre of pressure in the body frame
		"""
		self.body = body
		self.area = area
		self.cp_position = position - body.cm_position

	def force_on(self, body, time, position, rotation, velocity, angularv):
		if body == self.body:
			v = self.environment.get_air_velocity(time)
			return 1/2*self.area*self.environment.get_air_density(time)*np.linalg.norm(v)*v
		else:
			return np.zeros(3)

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		if body == self.body:
			return np.cross(rotation.rotate(self.cp_position), self.force_on(body, time, position, rotation, velocity, angularv))
		else:
			return np.zeros(3)


def normalized(vec):
	mag = np.linalg.norm(vec)
	if mag == 0:
		base = np.zeros(vec.shape)
		base[0] = 1
		return base
	else:
		return vec / mag

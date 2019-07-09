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


class MagneticDipole(Magnetorker):
	""" External torke imparted by a uniform magnetic field on a magnetic body. """
	def __init__(self, body, dipole_moment):
		""" body:			RigidBody			the body on which the torke is applied
			dipole_moment:	3 vector			the magnetic dipole moment of the body in its reference frame
		"""
		self.moment = np.array(dipole_moment)
		super().__init__(body, lambda t: self.moment)


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
	def __init__(self, area, cp_position):
		""" lever_arm:		3 vector		the thruster's position on the body
			area:			float			the effective area of the body with the drag coefficient multiplied in
			cp_position:	3 vector		the position of the centre of pressure in the body frame
		""" # TODO: account for orientation- and body-dependent cD and cP
		self.area = area
		self.cp_position = np.array(cp_position)

	def force_on(self, body, time, position, rotation, velocity, angularv):
		velocity = self.environment.get_air_velocity(time)
		return 1/2*self.area*self.environment.get_air_density(time)*np.linalg.norm(velocity)*velocity

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		return 1e3*np.cross(rotation.rotate(self.cp_position), self.force_on(body, time, position, rotation, velocity, angularv)) -\
			1/2*self.area**(2.5)*self.environment.get_air_density(time)*np.linalg.norm(angularv)*angularv # combine torque due to offset center of pressure and against to rotation of body

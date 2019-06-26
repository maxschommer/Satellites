import numpy as np


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


class Magnetorker(Impulsor):
	""" External torke imparted by a uniform magnetic field on a variable-magnitude dipole. """
	def __init__(self, body, direction, magnitude, external_field=[0, 35e-6, 0]):
		""" body:			RigidBody			the body on which this is mounted
			direction:		3 vector			the direction of the magnetic moment this produces
			magnitude:		(float) -> float	the magnitude of the magnetic moment at a given time
			external_field:	3 vector			the value of the ambient magnetic flux density
		"""
		self.body = body
		self.direction = np.array(direction)/np.linalg.norm(direction)
		self.magnitude = magnitude
		self.field = np.array(external_field)

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		if body is self.body:
			return np.cross(rotation.rotate(self.direction)*self.magnitude(time), self.field)
		else:
			return np.zeros(3)


class MagneticDipole(Magnetorker):
	""" External torke imparted by a uniform magnetic field on a magnetic body. """
	def __init__(self, body, dipole_moment, external_field):
		""" body:			RigidBody	the body on which the torke is applied
			dipole_moment:	3 vector	the magnetic dipole moment of the body in its reference frame
			external_field:	3 vector	the magnetic flux density of the environment
		"""
		self.dipole_moment = np.array(dipole_moment)
		super().__init__(
			body, self.dipole_moment/np.linalg.norm(self.dipole_moment),
			lambda t: np.linalg.norm(self.dipole_moment), external_field)


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
		if body is self.body:
			return rotation.rotate(self.thrust(time))
		else:
			return np.zeros(3)

	def torke_on(self, body, time, position, rotation, velocity, angularv):
		if body is self.body:
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
import numpy as np
from pyquaternion import Quaternion


class Sensor:
	""" A component that gathers and stores information about the environment. """
	def __init__(self):
		self.environment = None

	def sense(self, time, positions, rotations, velocitys, angularvs):
		""" Store the current reading. """
		self.last_value = self.reading(time, positions, rotations, velocitys, angularvs)

	def all_readings(self, ts, states):
		""" Get readings in bulk and output them in a list. """
		readings = []
		for t, state in zip(ts, states):
			positions, rotations, velocitys, angularvs = [], [], [], []
			for body in self.environment.bodies.values():
				idx = 13*body.num
				positions.append(state[idx+0:idx+3])
				rotations.append(Quaternion(state[idx+3:idx+7]))
				momentum = state[idx+7:idx+10]
				angularm = state[idx+10:idx+13]
				I_inv_rot = np.matmul(np.matmul(rotations[-1].rotation_matrix,body.I_inv),rotations[-1].rotation_matrix.transpose())
				velocitys.append(momentum/body.m)
				angularvs.append(np.matmul(I_inv_rot, angularm))
			readings.append(self.reading(t, positions, rotations, velocitys, angularvs))
		return readings


	def reading(self, time, positions, rotations, velocitys, angularvs):
		""" Get the current value seen by the sensor. """
		raise NotImplementedError("Subclasses should override.")


class Photodiode(Sensor):
	""" A daylight detector. """
	SUN_SIZE = np.radians(.5)
	def __init__(self, body, direction):
		""" body:		RigidBody	the body to which this is mounted
			direction:	3 vector	the direction this diode faces (it cannot detect light coming from any other direction)
		"""
		self.body = body
		self.direction = np.array(direction)/np.linalg.norm(direction)

	def reading(self, time, positions, rotations, velocitys, angularvs):
		solar_flux = self.environment.get_solar_flux(time)
		if np.dot(rotations[self.body.num].rotate(self.direction), solar_flux) >= 0:
			return np.linalg.norm(solar_flux)
		else:
			return 0


class Magnetometer(Sensor):
	""" A sensor that just gets the magnetic field in it's body's reference frame, as well as the first derivative. """
	def __init__(self, body):
		""" body:		RigidBody	the body to which this is mounted
			direction:	3 vector	the direction this diode faces (it cannot detect light coming from any other direction)
		"""
		self.body = body

	def reading(self, time, positions, rotations, velocitys, angularvs):
		B = self.environment.get_magnetic_field(time)
		B_dot = (B - self.environment.get_magnetic_field(time-1.))/1.
		B_rot = rotations[self.body.num].inverse.rotate(B)
		B_dot_rot = rotations[self.body.num].inverse.rotate(B_dot) -\
			np.cross(rotations[self.body.num].inverse.rotate(angularvs[self.body.num]), B_rot)
		return np.concatenate((B_rot, B_dot_rot))


class BangBangBdot():
	""" A control loop for a Magnetorker, using a Magnetometer, to orient a RigidBody with respect to a magnetic field. """
	def __init__(self, sensor, max_moment, axes=[1,1,1], precision=1):
		""" sensor:		Magnetometer	the sensor from which to obtain feedback
			axis:		3 vector		the desired direction of the external magnetic field in the body frame
			max_moment:	float			the highest moment it will ever suggest on an axis (i.e. the Magnetorker's operating limit)
			precision:	float			the minimum period of the bangs before it will switch to a linear response
		"""
		self.sensor = sensor
		self.axes = np.array(axes)
		self.max_moment = max_moment
		self.gain = 2.2e-3/((3e-5)**2*precision)

	def __call__(self, time):
		""" Compute the magnetic dipole moment to exert. """
		Bx, By, Bz, Bx_dot, By_dot, Bz_dot = self.sensor.last_value
		return -np.array([
			np.clip(self.gain*Bx_dot, -self.max_moment, self.max_moment),
			np.clip(self.gain*By_dot, -self.max_moment, self.max_moment),
			np.clip(self.gain*Bz_dot, -self.max_moment, self.max_moment)])*self.axes

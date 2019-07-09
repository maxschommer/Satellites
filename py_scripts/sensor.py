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
		B_rot = rotations[self.body.num].inverse.rotate(self.environment.get_magnetic_field(time))
		B_dot_rot = -np.cross(rotations[self.body.num].inverse.rotate(angularvs[self.body.num]), B_rot)
		return np.concatenate((B_rot, B_dot_rot))

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
			for i, body in enumerate(self.environment.bodies):
				positions.append(state[13*i:13*i+3])
				rotations.append(Quaternion(state[13*i+3:13*i+7]))
				momentum = state[13*i+7:13*i+10]
				angularm = state[13*i+10:13*i+13]
				I_inv_rot = np.matmul(np.matmul(rotations[i].rotation_matrix,body.I_inv),rotations[i].rotation_matrix.transpose())
				velocitys.append(momentum/body.m)
				angularvs.append(np.matmul(I_inv_rot, angularm))
			readings.append(self.reading(t, positions, rotations, velocitys, angularvs))
		return readings


	def reading(self, time, positions, rotations, velocitys, angularvs):
		""" Get the current value seen by the sensor. """
		raise NotImplementedError("Subclasses should override.")


class Photodiode(Sensor):
	""" A daylight detector. """
	def __init__(self, body, direction):
		""" body:		RigidBody	the body to which this is mounted
			direction:	3 vector	the direction this diode faces (it cannot detect light coming from any other direction)
		"""
		self.body = body
		self.direction = np.array(direction)

	def reading(self, time, positions, rotations, velocitys, angularvs):
		if np.dot(rotations[self.body.body_num].rotate(self.direction), self.environment.get_solar_flux(time)) < 0:
			return np.linalg.norm(self.environment.get_solar_flux(time))
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
		B_rot = rotations[self.body.body_num].inverse.rotate(self.environment.get_magnetic_field(time))
		B_dot_rot = -np.cross(rotations[self.body.body_num].inverse.rotate(angularvs[self.body.body_num]), B_rot)
		return np.concatenate((B_rot, B_dot_rot))

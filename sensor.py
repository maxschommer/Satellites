import numpy as np
from pyquaternion import Quaternion


class Sensor:
	""" A component that gathers and stores information about the environment. """
	def __init__(self):
		self.environment = None

	def sense(self, positions, rotations, velocitys, angularvs):
		self.last_value = self.reading(positions, rotations, velocitys, angularvs)

	def all_readings(self, states):
		readings = []
		for state in states:
			positions, rotations, velocitys, angularvs = [], [], [], []
			for i, body in enumerate(self.environment.bodies):
				positions.append(state[13*i:13*i+3])
				rotations.append(Quaternion(state[13*i+3:13*i+7]))
				momentum = state[13*i+7:13*i+10]
				angularm = state[13*i+10:13*i+13]
				I_inv_rot = np.matmul(np.matmul(rotations[i].rotation_matrix,body.I_inv),rotations[i].rotation_matrix.transpose())
				velocitys.append(momentum/body.m)
				angularvs.append(np.matmul(I_inv_rot, angularm))
			readings.append(self.reading(positions, rotations, velocitys, angularvs))
		return readings


	def reading(self, positions, rotations, velocitys, angularvs):
		raise NotImplementedError("Subclasses should override.")


class Photodiode(Sensor):
	""" A daylight detector. """
	def __init__(self, body, direction):
		""" body:		RigidBody	the body to which this is mounted
			direction:	3 vector	the direction this diode faces (it cannot detect light coming from any other direction)
		"""
		self.body = body
		self.direction = np.array(direction)

	def reading(self, positions, rotations, velocitys, angularvs):
		if np.dot(rotations[self.body.body_num].rotate(self.direction), self.environment.solar_flux) < 0:
			return np.linalg.norm(self.environment.solar_flux)
		else:
			return 0

import numpy as np


class Magnetostabilisation():
	""" A control loop for a Magnetorker, using a Magnetometer, to orient a RigidBody with respect to a magnetic field. """
	def __init__(self, sensor, axis, max_moment=float('inf')):
		""" sensor:		Magnetometer	the sensor from which to obtain feedback
			axis:		3 vector		the desired direction of the external magnetic field in the body frame
			max_moment:	float			the highest moment it will ever suggest on an axis (i.e. the Magnetorker's operating limit)
		"""
		self.sensor = sensor
		self.axis = np.array(axis)
		self.max_moment = max_moment

	def __call__(self, time):
		""" Compute the magnetic dipole moment to exert. """
		return self.axis

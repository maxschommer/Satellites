# constraint.py
# Contains constraint-solving code

import numpy as np
from pyquaternion import Quaternion


BASIS_VECTORS = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
BASIS_QUATERNIONS = [Quaternion(vector=vector) for vector in BASIS_VECTORS]


class Constraint(object):
	""" A law that applies internal forces and torkes to two RigidBodys. """
	def __init__(self, body_a, body_b):
		self.body_a = body_a
		self.body_b = body_b

	def constraint_values(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		""" Compute the vector of values that need to go to zero.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	n vector	the current value of the constraint vector
		"""
		raise NotImplementedError("Subclasses should override (can you call it 'overriding' in Python, or is that Java-specific?)")

	def constraint_derivative(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		""" Compute the time derivative of the constraint vector.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	n vector	the current time derivative of the constraint vector
		"""
		raise NotImplementedError("Subclasses should override.")

	def constraint_jacobian(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		""" Compute the Jacobian matrix of the constraint vector.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	nx14 matrix	the current derivative of the constraint vector with respect to the state vector
		"""
		raise NotImplementedError("Subclasses should override.")

	def constraint_derivative_jacobian(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		""" Compute the time derivative of the Jacobian.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	nx14 matrix	the current derivative of the time derivative of the constraint vector with respect to the state vector
		"""
		raise NotImplementedError("Subclasses should override.")

	def force_response(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		""" Compute the force response matrix.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	nx12 matrix	the matrix that converts the constrainting parameter vector to the force-torke vectors
		"""
		raise NotImplementedError("Subclasses should override.")

	def response(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b, I_inv_rot_a=None, I_inv_rot_b=None):
		""" Compute the second derivative response matrix.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	nx14 matrix	the matrix that converts the constraining parameter vector to the state second derivative contribution
		"""
		force_torke_response = self.force_response(position_a, rotation_a, velocity_a, angularv_a, position_b, rotation_b, velocity_b, angularv_b)
		pos_a_response = force_torke_response[0:3,:]/self.body_a.m
		pos_b_response = force_torke_response[6:9,:]/self.body_b.m
		quaternion_matrix_a = 1/2*np.array([list(BASIS_QUATERNIONS[i]*rotation_a) for i in range(3)]).transpose()
		quaternion_matrix_b = 1/2*np.array([list(BASIS_QUATERNIONS[i]*rotation_b) for i in range(3)]).transpose()
		rot_a_response = np.matmul(quaternion_matrix_a, np.matmul(I_inv_rot_a, force_torke_response[3:6,:]))
		rot_b_response = np.matmul(quaternion_matrix_b, np.matmul(I_inv_rot_b, force_torke_response[9:12,:]))
		return np.vstack((pos_a_response, rot_a_response, pos_b_response, rot_b_response))

	def force_torke_on_a(self, vector,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		""" Compute the force-torke vector on body_a.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return  nx14 matrix the matrix that converts the f
		"""
		return np.matmul(
			self.force_response(position_a, rotation_a, velocity_a, angularv_a, position_b, rotation_b, velocity_b, angularv_b)[0:6,:],
			vector)

	def force_torke_on_b(self, vector,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		""" Compute the force-torke vector on body_b.
			pos_a	3 vector	the current position of body_a
			rot_a	Quaternion	the current orientation of body_a
			vel_a	3 vector	the current linear velocity of body_a
			omg_a	3 vector	the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return  nx14 matrix the matrix that converts the f
		"""
		return np.matmul(
			self.force_response(position_a, rotation_a, velocity_a, angularv_a, position_b, rotation_b, velocity_b, angularv_b)[6:12,:],
			vector)


class BallJoint(Constraint):
	""" Restricts two points on two RigidBodies to be fixed relative to each other. """
	def __init__(self, body_a, body_b, point_a, point_b):
		""" body_a:		RigidBody		the first body of the joint
			body_b: 	RigidBody		the second body of the joint
			point_a:	3 float vector	the pin location in the first body's coordinate frame
			point_b:	3 float vector	the pin location in the second body's coordinate frame
		"""
		super().__init__(body_a, body_b)
		self.point_a = np.array(point_a) - body_a.cm_position
		self.point_b = np.array(point_b) - body_b.cm_position # all internal calculations measure point_a and point_b from the centre of mass
		self.num_dof = 3 # x force on a, y force on a, z force on a

	def constraint_values(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		return position_a + rotation_a.rotate(self.point_a) - position_b - rotation_b.rotate(self.point_b)

	def constraint_derivative(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		return velocity_a + np.cross(angularv_a,rotation_a.rotate(self.point_a)) - velocity_b - np.cross(angularv_b, rotation_b.rotate(self.point_b))

	def constraint_jacobian(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		pos_a_dependency = np.identity(3)
		pos_b_dependency = -np.identity(3)
		rot_a_dependency = np.zeros((3, 4))
		rot_b_dependency = np.zeros((3, 4))
		for i in range(3):
			rot_a_dependency[i,:] = list(-2*BASIS_QUATERNIONS[i]*rotation_a*Quaternion(vector=self.point_a))
			rot_b_dependency[i,:] = list( 2*BASIS_QUATERNIONS[i]*rotation_b*Quaternion(vector=self.point_b))
		return np.hstack((pos_a_dependency, rot_a_dependency, pos_b_dependency, rot_b_dependency))

	def constraint_derivative_jacobian(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		pos_a_dependency = np.zeros((3, 3))
		pos_b_dependency = -np.zeros((3, 3))
		rot_a_dependency = np.zeros((3, 4))
		rot_b_dependency = np.zeros((3, 4))
		for i in range(3):
			rot_a_dependency[i,:] = list(-BASIS_QUATERNIONS[i]*Quaternion(vector=angularv_a)*rotation_a*Quaternion(vector=self.point_a))
			rot_b_dependency[i,:] = list( BASIS_QUATERNIONS[i]*Quaternion(vector=angularv_b)*rotation_b*Quaternion(vector=self.point_b))
		return np.hstack((pos_a_dependency, rot_a_dependency, pos_b_dependency, rot_b_dependency))

	def force_response(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		force_a_response = np.identity(3)
		force_b_response = -np.identity(3)
		torke_a_response = cross_matrix(rotation_a.rotate(self.point_a))
		torke_b_response = -cross_matrix(rotation_b.rotate(self.point_b))
		return np.vstack((force_a_response, torke_a_response, force_b_response, torke_b_response))


class Parallel(Constraint):
	""" Restricts two points on two RigidBodies to have a particular axis always parallel. """
	def __init__(self, body_a, body_b, axis_a, axis_b):
		""" body_a:		RigidBody		the first body of the joint
			body_b: 	RigidBody		the second body of the joint
			axis_a:		3 float vector	the hinge axis in the first body's coordinate frame
			axis_b:		3 float vector	the hinge axis in the second body's coordinate frame
		"""
		super().__init__(body_a, body_b)
		axis_a = np.array(axis_a)/np.linalg.norm(axis_a) # normalise the primary axis
		if axis_a[1] != 0 or axis_a[2] != 0:
			self.axis_a1 = np.cross(axis_a, [1,0,0]) # pick an orthogonal secondary axis
		else:
			self.axis_a1 = np.cross(axis_a, [0,1,0])
		self.axis_a1 /= np.linalg.norm(self.axis_a1)
		assert np.dot(axis_a, self.axis_a1) == 0
		self.axis_a2 = np.cross(axis_a, self.axis_a1) # and define a second secondary axis based on that

		self.axis_b = np.array(axis_b)/np.linalg.norm(axis_b)

		self.num_dof = 2 # axis_1 torke, axis_2 torke

	def constraint_values(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		return np.array([
			np.dot(rotation_b.rotate(self.axis_b), rotation_a.rotate(self.axis_a1)),
			np.dot(rotation_b.rotate(self.axis_b), rotation_a.rotate(self.axis_a1))])

	def constraint_derivative(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		axis_a1_r = rotation_a.rotate(self.axis_a1)
		axis_a2_r = rotation_a.rotate(self.axis_a2)
		axis_b_r = rotation_b.rotate(self.axis_b)
		return np.array([
			np.dot(np.cross(angularv_b, axis_b_r), axis_a1_r) + np.dot(axis_b_r, np.cross(angularv_a, axis_a1_r)),
			np.dot(np.cross(angularv_b, axis_b_r), axis_a2_r) + np.dot(axis_b_r, np.cross(angularv_a, axis_a2_r))])

	def constraint_jacobian(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		rot_b_dependency = np.zeros((3, 4)) # these are the jacobians of qvq* specifically
		for i in range(3):
			rot_b_dependency[i,:] = list(2*BASIS_QUATERNIONS[i]*rotation_b*Quaternion(vector=self.axis_b))

		jacob = np.zeros((2, 14))
		for j, axis_a in enumerate([self.axis_a1, self.axis_a2]):
			rot_a_dependency = np.zeros((3, 4)) # these are the jacobians of qvq* specifically
			for i in range(3):
				rot_a_dependency[i,:] = list(2*BASIS_QUATERNIONS[i]*rotation_a*Quaternion(vector=axis_a))
			jacob[j,3:7] = np.matmul(rotation_b.rotate(self.axis_b), rot_a_dependency)
			jacob[j,10:14] = np.matmul(rotation_a.rotate(axis_a), rot_b_dependency)

		return jacob

	def constraint_derivative_jacobian(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		rot_b_dependency, rot_b_dot_dependency = np.zeros((3, 4)), np.zeros((3, 4)) # these are the jacobians of qvq* specifically
		for i in range(3):
			rot_b_dependency[i,:] = list(2*BASIS_QUATERNIONS[i]*rotation_b*Quaternion(vector=self.axis_b))
			rot_b_dot_dependency[i,:] = list(2*BASIS_QUATERNIONS[i]*rotation_b*Quaternion(vector=self.axis_b))

		jacob = np.zeros((2, 14))
		for j, axis_a in enumerate([self.axis_a1, self.axis_a2]):
			rot_a_dependency, rot_a_dot_dependency = np.zeros((3, 4)), np.zeros((3, 4)) # these are the jacobians of qvq* specifically
			for i in range(3):
				rot_a_dependency[i,:] = list(2*BASIS_QUATERNIONS[i]*rotation_a*Quaternion(vector=axis_a))
				rot_a_dot_dependency[i,:] = list(2*BASIS_QUATERNIONS[i]*rotation_a*Quaternion(vector=axis_a))
			jacob[j,3:7] = np.matmul(rotation_b.rotate(self.axis_b), rot_a_dot_dependency) +\
				np.matmul(np.cross(angularv_b, rotation_b.rotate(self.axis_b)), rot_a_dependency)
			jacob[j,10:14] = np.matmul(rotation_a.rotate(axis_a), rot_b_dependency) +\
				np.matmul(np.cross(angularv_a, rotation_a.rotate(axis_a)), rot_a_dependency)

		return jacob

	def force_response(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		force_a_response = np.zeros((3, 2))
		force_b_response = np.zeros((3, 2))
		torke_a_response = np.stack((rotation_a.rotate(self.axis_a1), rotation_a.rotate(self.axis_a2)), axis=1)
		torke_b_response = -torke_a_response
		return np.vstack((force_a_response, torke_a_response, force_b_response, torke_b_response))


class Hinge(Constraint):
	""" Restricts two line segments on two RigidBodies to be fixed relative to each other. """
	def __init__(self, body_a, body_b, point_a, point_b, axis_a, axis_b):
		""" body_a:		RigidBody		the first body of the joint
			body_b: 	RigidBody		the second body of the joint
			point_a:	3 float vector	the hinge centre in the first body's coordinate frame
			point_b:	3 float vector	the hinge centre in the second body's coordinate frame
			axis_a:		3 float vector	the direction of the hinge in the first body's coordinate frame
			axis_b:		3 float vector	the direction of the hinge in the second body's coordinate frame
		"""
		super().__init__(body_a, body_b)
		self.sub_constraint_α = BallJoint(body_a, body_b, np.array(point_a), np.array(point_b))
		self.sub_constraint_β = Parallel(body_a, body_b, axis_a, axis_b)
		self.num_dof = 5

	def constraint_values(self, *args):
		return np.concatenate((self.sub_constraint_α.constraint_values(*args), self.sub_constraint_β.constraint_values(*args)))

	def constraint_derivative(self, *args):
		return np.concatenate((self.sub_constraint_α.constraint_derivative(*args), self.sub_constraint_β.constraint_derivative(*args)))

	def constraint_jacobian(self, *args):
		return np.vstack((self.sub_constraint_α.constraint_jacobian(*args), self.sub_constraint_β.constraint_jacobian(*args)))

	def constraint_derivative_jacobian(self, *args):
		return np.vstack((self.sub_constraint_α.constraint_derivative_jacobian(*args), self.sub_constraint_β.constraint_derivative_jacobian(*args)))

	def force_response(self, *args):
		return np.hstack((self.sub_constraint_α.force_response(*args), self.sub_constraint_β.force_response(*args)))


def cross_matrix(v):
	""" Returns this vector cross product as a matrix multiplication. """
	return np.array([
		[    0, -v[2],  v[1]],
		[ v[2],     0, -v[0]],
		[-v[1],  v[0],     0]])

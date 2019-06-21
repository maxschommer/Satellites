# constraint.py
# Contains constraint-solving code

import numpy as np
from pyquaternion import Quaternion


BASIS_VECTORS = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
BASIS_QUATERNIONS = [Quaternion(1,0,0,0), Quaternion(0,1,0,0), Quaternion(0,0,1,0), Quaternion(0,0,0,1)]


class Constraint(object):
	""" A law that applies internal forces and torques to two RigidBodys. """
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
			return	nx14 matrix	the matrix that converts the force vector to the staet second derivative contribution
		"""
		raise NotImplementedError("Subclasses should override.")



class BallJointConstraint(Constraint):
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
		self.num_dof = 3

		# body_b.position = body_a.position + body_a.rotation.rotate(point_a) - body_b.rotation.rotate(point_b) # adjust starting conditions such
		# body_b.velocity = body_a.velocity + np.cross(body_a.angularv,body_a.rotation.rotate(point_a)) -       # that constraint is initially met
		# 		np.cross(body_b.angularv,body_b.rotation.rotate(point_b))

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
		for i in range(4):
			rot_a_dependency[:,i] = (rotation_a*BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_a) +
					BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_a)*rotation_a).vector
			rot_b_dependency[:,i] = (-rotation_b*BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_b) +
					BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_b)*rotation_b).vector
		return np.hstack((pos_a_dependency, rot_a_dependency, pos_b_dependency, rot_b_dependency))

	def constraint_derivative_jacobian(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		pos_a_dependency = np.zeros((3, 3))
		pos_b_dependency = -np.zeros((3, 3))
		rot_a_dependency = np.zeros((3, 4))
		rot_b_dependency = np.zeros((3, 4))
		for i in range(4):
			rot_a_dependency[:,i] = np.cross(angularv_a, (rotation_a*BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_a) +
					BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_a)*rotation_a).vector)
			rot_b_dependency[:,i] = -np.cross(angularv_b, (rotation_b*BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_b) +
					BASIS_QUATERNIONS[i]*Quaternion(vector=self.point_b)*rotation_b).vector)
		return np.hstack((pos_a_dependency, rot_a_dependency, pos_b_dependency, rot_b_dependency))

	def force_response(self,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		pos_a_response = np.identity(3)/self.body_a.m
		pos_b_response = -np.identity(3)/self.body_b.m
		rot_a_response = np.zeros((4, 3))
		rot_b_response = np.zeros((4, 3))
		for i in range(3):
			I_inv_rot_a = np.matmul(np.matmul(rotation_a.rotation_matrix,self.body_a.I_inv),rotation_a.rotation_matrix.transpose())
			I_inv_rot_b = np.matmul(np.matmul(rotation_b.rotation_matrix,self.body_b.I_inv),rotation_b.rotation_matrix.transpose())
			rot_a_response[:,i] = [*1/2*Quaternion(vector=np.matmul(I_inv_rot_a, np.cross(rotation_a.rotate(self.point_a), BASIS_VECTORS[i])))*rotation_a]
			rot_b_response[:,i] = [*-1/2*Quaternion(vector=np.matmul(I_inv_rot_b, np.cross(rotation_b.rotate(self.point_b), BASIS_VECTORS[i])))*rotation_b]
		return np.vstack((pos_a_response, rot_a_response, pos_b_response, rot_b_response))

	def force_torque_on_a(self, vector,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		return [*(vector), *(np.cross(rotation_a.rotate(self.point_a), vector))]

	def force_torque_on_b(self, vector,
			position_a, rotation_a, velocity_a, angularv_a,
			position_b, rotation_b, velocity_b, angularv_b):
		return [*(-vector), *(np.cross(rotation_b.rotate(self.point_b), -vector))]


class HingeJointConstraint(Constraint):
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
		self.sub_constraint_α = BallJointConstraint(body_a, body_b, np.array(point_a)+axis_a, np.array(point_b)+axis_b)
		self.sub_constraint_β = BallJointConstraint(body_a, body_b, np.array(point_a)-axis_a, np.array(point_b)-axis_b)

	def force_and_torque_on_a(self, pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b):
		self.sub_constraint_β.frc_and_trq_on_a(pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b) # TODO: incorporate both
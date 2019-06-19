# constraint.py
# Contains constraint-solving code

import numpy as np


class Constraint(object):
	""" A law that applies internal forces and torques to two RigidBodys. """
	def __init__(self, body_a, body_b):
		self.body_a = body_a
		self.body_b = body_b

	def force_and_torque_on_a(self, pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b):
		""" Compute the constraint-satisfying restorative force and torque on body_a.
			pos_a	3 float vector			the current position of body_a
			rot_a	Quaternion				the current orientation of body_a
			vel_a	3 float vector			the current linear velocity of body_a
			omg_a	3 float vector			the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	(3 vector, 3 vector)	the applied force and torque on body_a
		"""
		raise NotImplementedError("Subclasses should override (can you call it 'overriding' in Python, or is that Java-specific?)")

	def force_and_torque_on_b(self, pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b):
		""" Compute the constraint-satisfying restorative force and torque on body_b.
			pos_a	3 float vector			the current position of body_a
			rot_a	Quaternion				the current orientation of body_a
			vel_a	3 float vector			the current linear velocity of body_a
			omg_a	3 float vector			the current angular velocity of body_a
			..._b	Do I need to specify the rest?
			return	(3 vector, 3 vector)	the applied force and torque on body_b
		"""
		f_on_a, τ_on_a = self.force_on_a(pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b)
		return -f_on_a, -τ_on_a


class BallJointConstraint(Constraint):
	""" Restricts two points on two RigidBodies to be fixed relative to each other. """
	def __init__(self, body_a, body_b, point_a, point_b):
		""" body_a:		RigidBody		the first body of the joint
			body_b: 	RigidBody		the second body of the joint
			point_a:	3 float vector	the pin location in the first body's coordinate frame
			point_b:	3 float vector	the pin location in the second body's coordinate frame
		"""
		super().__init__(body_a, body_b)
		self.point_a = np.array(point_a)
		self.point_b = np.array(point_b)

	def force_and_torque_on_a(self, pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b):
		return np.array([0,0,0]), np.array([0,0,0]) # TODO: this


class HingeJointConstraint(Constraint):
	""" Restricts two line segments on two RigidBodies to be fixed relative to each other. """
	def __init__(self, body_a, body_b, point_a, point_b, axis_a):
		""" body_a:		RigidBody		the first body of the joint
			body_b: 	RigidBody		the second body of the joint
			point_a:	3 float vector	the hinge centre in the first body's coordinate frame
			point_b:	3 float vector	the hinge centre in the second body's coordinate frame
			axis_a:		3 float vector	the direction of the hinge in the first body's coordinate frame
			axis_b:		3 float vector	the direction of the hinge in the second body's coordinate frame
		"""
		super().__init__(body_a, body_b)
		axis_b = body_b.rot.rotate(body_a.rot.conjugate.rotate(axis_a))
		self.sub_constraint_α = BallJointConstraint(body_a, body_b, np.array(point_a)+axis_a, np.array(point_b)+axis_b)
		self.sub_constraint_β = BallJointConstraint(body_a, body_b, np.array(point_a)-axis_a, np.array(point_b)-axis_b)

	def force_and_torque_on_a(self, pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b):
		self.sub_constraint_β.frc_and_trq_on_a(pos_a, rot_a, vel_a, omg_a, pos_b, rot_b, vel_b, omg_b) # TODO: incorporate both
import numpy as np
from pyquaternion import Quaternion
import time
import ratcave as rc



X_AX_ROTATION = Quaternion(axis=[1,0,0], degrees=90)
Y_AX_ROTATION = Quaternion(axis=[0,1,0], degrees=90)
Z_AX_ROTATION = Quaternion(axis=[0,0,1], degrees=90)

UNIT_SCALE = 9e-4
VELOCITY_SCALE = .02
ANGULARV_SCALE = .002


class Stage():
	""" A group of actors with functions for updating and displaying them """
	def __init__(self, actors, environment, speed=1, t_start=0, t_end=None):
		""" actors:			[Actor]		the list of Actors visible on this stage
			environment:	Environment	the Environment of bodies, so that we know about the physical solution we're showing
			speed:			float		the factor of realtime at which to animate
			t_start:		float		the first time to show (defaults to t=0)
			t_end:			float		the final time to show (defaults to last time simulated)
		"""
		self.actors = actors
		self.environment = environment
		self.speed = speed
		self.t = t_start
		self.t_start = t_start
		self.t_end = t_end if t_end is not None else environment.max_t
		self.started = False

		for actor in self.actors:
			actor.stage = self
			actor.load_resources()

	def update(self, dt):
		""" Move all Actors into updated positions.
			dt:	float	the number of seconds of realtime that have proressed
		"""
		if dt and not self.started: # ignore the first call to update
			self.started = True # because that's when pyglet tries to skip past the first 3 seconds
		else:
			self.t = self.t + self.speed*dt
			if self.t > self.t_end:
				self.t = self.t_start

			for a in self.actors:
				a.update(self.t)


class Actor():
	""" A visual representation of some entity in physical space. """
	def __init__(self, model, scale=1):
		""" model:			str						"{file address of mesh file}->{internal address of mesh form}"
			mesh_readers:	{str:WavefrontReader}	a dictionary of existing mesh readers for different files
		"""
		self.model = model
		self.scale = scale
		self.stage = None

	def load_resources(self, mesh_readers={}):
		""" Summon the datum that are inconvenient to summon during __init__. """
		model_directory, model_name = self.model.split("->")
		if model_directory not in mesh_readers:
			mesh_readers[model_directory] = rc.WavefrontReader("../Meshes/{}.obj".format(model_directory))
		self.mesh = mesh_readers[model_directory].get_mesh(model_name, scale=self.scale*UNIT_SCALE, mean_center=False)
		self.mesh.rotation = rc.coordinates.RotationQuaternion(1, 0, 0, 0)

	def update(self, t):
		""" Move the mesh for a new time if necessary.
			t	float	the current time in seconds
		"""
		pass


class BodyActor(Actor):
	""" An Actor that represents the physical form of a RigidBody. """
	def __init__(self, body, model, scale=1):
		""" body	RigidBody or str	the body to depict
			model	str					"{file address of mesh file}->{internal address of mesh form}"
		"""
		super().__init__(model, scale=scale)
		self.body = body

	def load_resources(self, **kwargs):
		super().load_resources(**kwargs)
		if type(self.body) == str:
			self.body = self.stage.environment.bodies[self.body]

	def update(self, t):
		position = self.body.get_position(t)
		rotation = self.body.get_rotation(t)
		self.mesh.position = position - rotation.rotate(self.body.cm_position)
		assign_wxyz(self.mesh.rotation, rotation)


class VectorActor(Actor):
	""" An Actor that represents a vector quantity. """
	def __init__(self, body, quantity, model, mounting_point=[0,0,0]):
		""" body		RigidBody or str	the body posessing the quantity to depict
			quantity	str				the name of the quantity to depict
			model		str				"{file address of mesh file}->{internal address of mesh form}"
		"""
		super().__init__(model)
		self.body = body
		self.quantity = quantity
		self.mounting_point = np.array(mounting_point)

	def load_resources(self, **kwargs):
		super().load_resources(**kwargs)
		if type(self.body) == str:
			self.body = self.stage.environment.bodies[self.body]

	def update(self, t):
		position = self.body.get_position(t)
		rotation = self.body.get_rotation(t)
		self.mesh.position = position + rotation.rotate(self.mounting_point)

		if self.quantity == "xaxis":
			assign_wxyz(self.mesh.rotation, self.body.get_rotation(t)*Z_AX_ROTATION.inverse)
		elif self.quantity == "yaxis":
			assign_wxyz(self.mesh.rotation, self.body.get_rotation(t))
		elif self.quantity == "zaxis":
			assign_wxyz(self.mesh.rotation, self.body.get_rotation(t)*X_AX_ROTATION)
		elif self.quantity == "velocity":
			v = self.body.get_velocity(t)
			if v[0] == 0 and v[2] == 0:
				assign_wxyz(self.mesh.rotation, [1,0,0,0] if ω[1] > 0 else [0,1,0,0])
			else:
				assign_wxyz(self.mesh.rotation, Quaternion(axis=[v[2],0,-v[0]], angle=np.arccos(v[1]/np.linalg.norm(v))))
			self.mesh.scale.y = max(1e-6, VELOCITY_SCALE*np.linalg.norm(v))
		elif self.quantity == "angularv":
			ω = self.body.get_angularv(t)
			if ω[0] == 0 and ω[2] == 0:
				assign_wxyz(self.mesh.rotation, [1,0,0,0] if ω[1] > 0 else [0,1,0,0])
			else:
				assign_wxyz(self.mesh.rotation, Quaternion(axis=[ω[2],0,-ω[0]], angle=np.arccos(ω[1]/np.linalg.norm(ω))))
			self.mesh.scale.y = max(1e-6, ANGULARV_SCALE*np.linalg.norm(ω))
		else:
			raise ValueError("Unrecognised vector quantity: {}".format(self.quantity))


class VectorFieldActor(Actor):
	""" An Actor that represents a vector field. """
	def __init__(self, field, model, position=[0,0,0]):
		""" field		3 vector or Table				the vector field to depict
			model		str								"{file address of mesh file}->{internal address of mesh form}"
			position	3 vector or RigidBody or str	the place to put the tail of this arrow
		"""
		super().__init__(model)
		self.field = field
		self.position = position
		v0 = field.get_value(0) if hasattr(field, 'get_value') else field
		self.unit = UNIT_SCALE/np.linalg.norm(v0)

	def load_resources(self, **kwargs):
		super().load_resources(**kwargs)
		if type(self.position) == str:
			self.position = self.stage.environment.bodies[self.position]

	def update(self, t):
		if hasattr(self.position, 'get_position'):
			self.mesh.position = self.position.get_position(t)
		else:
			self.mesh.position = self.position

		v = self.field.get_value(t) if hasattr(self.field, 'get_value') else self.field

		if v[0] == 0 and v[2] == 0:
			assign_wxyz(self.mesh.rotation, [1,0,0,0] if ω[1] > 0 else [0,1,0,0])
		else:
			assign_wxyz(self.mesh.rotation, Quaternion(axis=[v[2],0,-v[0]], angle=np.arccos(v[1]/np.linalg.norm(v))))
		self.mesh.scale.y = max(1e-6, self.unit*np.linalg.norm(v))


def assign_wxyz(ratcave_garbage, actual_quaternion):
	""" Assign a Quaternion value to a ractave.coordinates.RotationQuaternion without creating a new object.
		ratcave_garbage:	RotationQuaternion	the object that needs to have its fields reset despite having no convenient methods for doing so
		actual_quaternion:	Quaternion			the mostly decent object that has the four values to be transferred
	"""
	ratcave_garbage.w, ratcave_garbage.x, ratcave_garbage.y, ratcave_garbage.z = actual_quaternion

import numpy as np
from pyquaternion import Quaternion
import time
import ratcave as rc



X_AX_ROTATION = Quaternion(axis=[1,0,0], degrees=90)
Y_AX_ROTATION = Quaternion(axis=[0,1,0], degrees=90)
Z_AX_ROTATION = Quaternion(axis=[0,0,1], degrees=90)

UNIT_SCALE = .001
VELOCITY_SCALE = .02
ANGULARV_SCALE = .002


class Actor():
	def __init__(self, model, mesh_readers={}):
		model_directory, model_name = model.split("->")
		if model_directory not in mesh_readers:
			mesh_readers[model_directory] = rc.WavefrontReader("Meshes/{}.obj".format(model_directory))
		self.mesh = mesh_readers[model_directory].get_mesh(model_name, scale=UNIT_SCALE, mean_center=False)
		self.mesh.rotation = rc.coordinates.RotationQuaternion(1, 0, 0, 0)

	def update(self, t):
		pass


class BodyActor(Actor):
	def __init__(self, body, model):
		super().__init__(model)
		self.body = body

	def update(self, t):
		position = self.body.get_position(t)
		rotation = self.body.get_rotation(t)
		self.mesh.position = position - rotation.rotate(self.body.cm_position)
		assign_wxyz(self.mesh.rotation, rotation)


class VectorActor(Actor):
	def __init__(self, body, model, quantity, mounting_point=[0,0,0]):
		super().__init__(model)
		self.body = body
		self.quantity = quantity
		self.mounting_point = np.array(mounting_point)

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
				assign_wxyz(self.mesh.rotation, [0,0,0,0])
			else:
				assign_wxyz(self.mesh.rotation, Quaternion(axis=[v[2],0,-v[0]], angle=np.arccos(v[1]/np.linalg.norm(v))))
			self.mesh.scale.y = VELOCITY_SCALE*np.linalg.norm(v)
		elif self.quantity == "angularv":
			ω = self.body.get_angularv(t)
			if ω[0] == 0 and ω[2] == 0:
				assign_wxyz(self.mesh.rotation, [0,0,0,0])
			else:
				assign_wxyz(self.mesh.rotation, Quaternion(axis=[ω[2],0,-ω[0]], angle=np.arccos(ω[1]/np.linalg.norm(ω))))
			self.mesh.scale.y = ANGULARV_SCALE*np.linalg.norm(ω)
		else:
			raise ValueError("Unrecognised vector quantity: {}".format(self.quantity))


class Stage():
	def __init__(self, actors, environment, speed=1):
		self.actors = actors
		self.environment = environment
		self.t = 0
		self.update(0)

	def update(self, dt):
		self.t = self.t + dt
		if self.t > self.environment.max_t:
			return

		for a in self.actors:
			a.update(self.t)



def assign_wxyz(ratcave_garbage, actual_quaternion):
	ratcave_garbage.w, ratcave_garbage.x, ratcave_garbage.y, ratcave_garbage.z = actual_quaternion

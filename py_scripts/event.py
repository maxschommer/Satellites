import numpy as np

from physics import RigidBody


class Event():
	""" I made a class because Numpy's system is dumb. """
	def __init__(self, terminal=False, direction=0):
		""" terminal:	Boolean		whether to terminate integration if this event occurs
			direction:	float		direction of a zero crossing
		"""
		self.terminal = terminal
		self.direction = direction
		self.environment = None

	def __call__(self, t, state):
		""" A continuous float function that goes to zero when the event occurs. """
		raise NotImplementedError("Subclasses should override.")

	def happen(self):
		""" Execute the changes to the environment that accompany this event. """
		pass


class Launch(Event):
	""" An event set to happen at a specific time that spawns a new body from an existing one. """
	def __init__(self, t, launcher, payload, launch_site, launch_velocity, launch_angularv):
		""" main_body:		RigidBody	the initial body that executes the launch
			ejection_mass:	float		the mass of the body that gets launched from the main one
			t:				float		the time at which to execute the launch
		"""
		super().__init__(True, direction=-1)
		self.time = t
		self.launcher = launcher
		self.payload = payload
		self.launch_position = launch_site - launcher.cm_position
		self.launch_velocity = np.array(launch_velocity)
		self.launch_angularv = np.array(launch_angularv)
		self.payload.active = False

	def __call__(self, t, state):
		return self.time - t

	def happen(self):
		print("		launch!")
		launcher_velocity = self.launcher.momentum/self.launcher.m
		launcher_angularv = self.launcher.rotation.rotate(np.matmul(self.launcher.I_inv, self.launcher.rotation.inverse.rotate(self.launcher.angularm)))
		self.payload.position = self.launcher.position + self.launcher.rotation.rotate(self.launch_position)
		self.payload.momentum = self.payload.m*(launcher_velocity + self.launcher.rotation.rotate(self.launch_velocity))
		self.payload.angularm = np.matmul(self.payload.I, launcher_angularv + self.launcher.rotation.rotate(self.launch_angularv))
		self.launcher.m -= self.payload.m
		self.launcher.I -= self.payload.I
		self.launcher.I_inv = np.linalg.inv(self.launcher.I)
		self.launcher.momentum -= self.payload.momentum
		self.launcher.angularm -= self.payload.angularm
		self.payload.active = True

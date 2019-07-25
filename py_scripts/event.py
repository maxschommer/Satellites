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
	def __init__(self, t, launcher, payload, launch_site, launch_velocity, launch_angularv, actually_do_it=True):
		""" t:					float		the time at which to execute the launch
			launcher:			RigidBody	the body from which the payload is launched
			payload:			RigidBody	the body that the launcher launches
			launch_site:		3 vector	the location whence the payload is launched in the launcher's body frame
			launch_velocity:	3 vector	the relative velocity with which the payload is launched
			launch_angularv:	3 vector	the relative angular velocity with which the payload is launched
			actually_do_it:		bool		if set to False, the launcher will be affected, but the payload will not be simulated
		"""
		super().__init__(True, direction=-1)
		self.time = t
		self.launcher = launcher
		self.payload = payload
		self.launch_position = launch_site - launcher.cm_position
		self.launch_velocity = np.array(launch_velocity)
		self.launch_angularv = np.array(launch_angularv)
		self.payload.active = False
		self.actually_do_it = actually_do_it

	def __call__(self, t, state):
		return self.time - t

	def happen(self):
		print("		launch!")
		launcher_velocity = self.launcher.momentum/self.launcher.m
		launcher_angularv = self.launcher.rotation.rotate(np.matmul(self.launcher.I_inv, self.launcher.rotation.inverse.rotate(self.launcher.angularm)))
		self.payload.position = self.launcher.position + self.launcher.rotation.rotate(self.launch_position)
		self.payload.rotation = self.launcher.rotation * self.payload.rotation
		self.payload.momentum = self.payload.m*(launcher_velocity + self.launcher.rotation.rotate(self.launch_velocity))
		payload_angularv = launcher_angularv + self.launcher.rotation.rotate(self.launch_angularv)
		self.payload.angularm = self.payload.rotation.rotate(np.matmul(self.payload.I, self.payload.rotation.inverse.rotate(payload_angularv)))
		self.launcher.m -= self.payload.m
		self.launcher.I -= self.payload.I
		self.launcher.I_inv = np.linalg.inv(self.launcher.I)
		self.launcher.momentum -= self.payload.momentum
		self.launcher.angularm -= self.payload.angularm - np.cross(self.launcher.rotation.rotate(self.launch_position), self.payload.momentum)
		if self.actually_do_it:
			self.payload.active = True

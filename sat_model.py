import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
import pyglet
from pyglet.gl import *
from pyglet.window import key
import ratcave as rc


dipole_moment = np.array([0, 0, 0]) # A*m^2
B_earth = np.array([35e-6, 0, 0]) # T

window = pyglet.window.Window()
scene = rc.Scene()

class RidgidBody():
	""" A physical unbending object free to move and rotate in space """
	def __init__(
			self, I, m, CM,  mesh,
			init_pos=[0,0,0], init_rot=[1,0,0,0], init_vel=[0,0,0], init_omg=[0,0,0]):
		""" I:		3x3 float array		rotational inertia
			m:		float				mass
			CM:		3 float vector		centre of mass
			mesh:	ractave mesh		shape of body
			init_pos:	3 float vector	initial linear displacement in [x, y, z]
			init_rot:	4 float vector	initial rotational displacement in [a, b, c, d]
			init_vel:	3 float vector	initial linear velocity in [x, y, z]
			init_omg:	3 float vector	initial angular velocity in [x, y, z]
		"""
		self.I = I
		self.I_inv = np.linalg.inv(I)
		self.m = m
		self.CM = np.array(CM)
		self.mesh = mesh
		self.mesh.rotation = rc.coordinates.RotationQuaternion(1, 0, 0, 0)

		self.pos = np.array(init_pos) + self.CM # position
		self.rot = Quaternion(*init_rot) # rotation
		self.mom = m*np.array(init_vel) # momentum
		self.anm = self.rot.rotate(np.matmul(I,self.rot.conjugate.rotate(np.array(init_omg)))) # angular momentum
		self.t = 0

		self.update(0)


	def update(self, del_t, f_ext=None, τ_ext=None):
		""" Update using an actual ODE solver.
			t_end:	float															the time of the desired solution
			f_ext:	3 vector of (float, 3 vector, Quaternion, 3 vector, 3 vector)	the external force at a given time and state
			τ_ext:	3 vector of (float, 3 vector, Quaternion, 3 vector, 3 vector)	the external force at a given time
		"""
		if f_ext is None:	f_ext = lambda *args: np.zeros(3)
		if τ_ext is None:	τ_ext = lambda *args: np.zeros(3)

		def state_derivative(t, state):
			pos, rot, mom, anm = state[0:3], Quaternion(state[3:7]), state[7:10], state[10:13]
			vel = mom/self.m
			omg = rot.rotate(np.matmul(self.I_inv,rot.conjugate.rotate(anm))) # make sure to use the correct ref frame when multiplying by I^-1
			return [*vel, *(1/2*Quaternion(0,*omg)*rot), *f_ext(t, pos, rot, mom, anm), *τ_ext(t, pos, rot, mom, anm)]

		state = [*self.pos, *self.rot, *self.mom, *self.anm]
		sol = solve_ivp(state_derivative, [self.t, self.t+del_t], state)
		state = sol.y[:,-1]
		self.pos, self.rot, self.mom, self.anm = state[0:3], Quaternion(state[3:7]), state[7:10], state[10:13]

		self.mesh.position = self.pos - self.rot.rotate(self.CM)
		self.mesh.rotation.w, self.mesh.rotation.x, self.mesh.rotation.y, self.mesh.rotation.z = self.rot
		self.t += del_t


class Environment():
	""" A space of objects with methods to render them """
	def __init__(self, objects):
		self.curr_t = 0
		self.keys = key.KeyStateHandler()
		window.push_handlers(self.keys)
		self.objects = objects
		scene.meshes = [o.mesh for o in objects]
		scene.camera.position.xyz = 0, 0, 2
		scene.bgColor = 1, 1, 1

		pyglet.clock.schedule(self.move_camera)
		pyglet.clock.schedule(self.update)

	@window.event
	def on_draw():
		with rc.default_shader:
			scene.draw()

	def update(self, dt):
		# print(curr_t)
		self.curr_t = self.curr_t + dt

		for obj in self.objects:
			obj.update(dt,
					τ_ext=lambda t,pos,rot,mom,anm: np.cross(rot.rotate(dipole_moment), B_earth)) # torque from a constant magnetic dipole


	def move_camera(self, dt):
		  camera_speed = 20
		  if self.keys[key.LEFT]:
			  scene.camera.rotation.y -= camera_speed * dt
		  if self.keys[key.RIGHT]:
			  scene.camera.rotation.y += camera_speed * dt



if __name__ == '__main__':
	# f = 1

	# def integrand(t, y0):

	# 	a = f / m * t
	# 	v = a * t + y0[0]
	# 	p = v * t + y0[1]
	# 	return p

	# y0 = (0,0)

	# I = RK45(integrand, 0, y0, 2)
	# I.step()

	# f_dense = I.dense_output()

	# t = np.linspace(0, 20)
	# # print(t)
	# print(f_dense(t))
	# plt.plot(t, f_dense(t)[0])
	# # plt.plot(t, integrand(t, (a,b)))
	# plt.show()



	# Insert filename into WavefrontReader.
	obj_filename = rc.resources.obj_primitives
	sat_reader = rc.WavefrontReader("Meshes/ThinSatFrame.obj")
	obj_reader = rc.WavefrontReader(obj_filename)
	# Create Mesh
	sat_mesh = sat_reader.get_mesh("Frame", position=(0, 0, 0), scale=.01)

	I = [[9.759e-5,  -4.039e-6, -1.060e-7],
		 [-4.039e-6,  7.858e-5,  7.820e-9],
		 [ -1.060e-7, 7.820e-9,  1.743e-4]] # kg*m^2
	m = 0.05223934 # kg
	CM = [0.00215328, -0.00860001, -0.00038142] # m --> check coordinates
	v0 = [-.02, .04, -.06] # m/s
	w0 = [.6, -.4, .2] # rad/s

	satellite = RidgidBody(I, m, CM, sat_mesh, init_vel=v0, init_omg=w0)




	# Create Scene

	env = Environment([satellite])



	pyglet.app.run()

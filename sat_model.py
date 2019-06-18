import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
import pyglet
from pyglet.gl import *
from pyglet.window import key
import ratcave as rc


# Make a 1d position solver
# Make a 3d position solver
# Make a 1d angle solver
# Make a 3d angle solver
# Integrate them
t_span = 20

window = pyglet.window.Window()
scene = rc.Scene()

class RidgidBody():
	""" A physical unbending object free to move and rotate in space """
	def __init__(self, I, m, CM,  mesh, init_pos=[0,0,0], init_rot=[1,0,0,0], init_vel=[0,0,0], init_omg=[0,0,0]):
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
		# print(I)
		# print(np.diag(I))
		# self.local_axis = np.linalg.eig(np.diag(I))
		# print(self.local_axis)
		self.m = m
		self.CM = CM
		self.mesh = mesh
		self.pos = np.array(init_pos) # position
		self.rot = Quaternion(*init_rot) # rotation
		self.mom = m*np.array(init_vel) # momentum
		self.anm = np.matmul(I,np.array(init_omg)) # angular momentum

	def update(self, del_t, f_ext, τ_ext): # TODO: implement an actual ODE solver
		""" Update using Euler's method.
			del_t:	float	the amount of time to jump forward
		"""
		rot_mat = self.rot.rotation_matrix # I was looking into using Euler's equations to try to avoid this step, but this PhD from Intel seems to think this is the best way
		I_inv_rot = np.matmul(np.matmul(rot_mat, self.I_inv), rot_mat.transpose()) # TODO: see if I really need to do these matrix multiplications, or if there's a faster way
		self.mom = self.mom + del_t*f_ext
		vel = self.mom/m
		self.anm = self.anm + del_t*τ_ext
		omg = np.matmul(I_inv_rot,self.anm)
		omg_norm = np.linalg.norm(omg)

		self.pos = self.pos + del_t*vel
		self.rot = self.rot + del_t*1/2*Quaternion(0,*omg)*self.rot

		self.mesh.position = self.pos
		self.mesh.rotation = rc.coordinates.RotationQuaternion(*self.rot) # TODO: is there a way to update these without instantiating an object each time?

	def solve(self, t_end, f_ext, τ_ext): # TODO: implement this
		""" Compute the position at time t_end given force and torque profiles.
			t_end:	float						the time of the desired solution
			f_ext:	3 float vector of (float)	the external force at a given time
			τ_ext:	3 float vector of (float)	the external force at a given time
		"""
		raise NotImplementedError()


# def f(t, A):
# 	# return 0
# 	return np.array([-.5*A[0], -.5*A[2], -.5*A[4]])
# 	# if y[0] < 0:
# 	# 	return -.5*y[0]
# 	# else:
# 	# 	return -.5*y[0]

# def oscillator(t, A):
# 	# dydt = [[x', x''/m],
# 	#         [y', y''/m],
# 	#         [z', z''/m]
# 	f_ext = f(t, A)
# 	dydt = [[A[1], f_ext[0]/m],
# 			[A[3], f_ext[1]/m],
# 			[A[5], f_ext[2]/m]]
# 	return dydt
# m = 1
# init = np.asarray([.2, -.1, 0, .1, 0, 0])

# t_span = 20
# sol = solve_ivp(oscillator, [0, t_span], init,
# 				vectorized=True, 
# 				 dense_output=True)
# t = np.linspace(0, t_span, 200)


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
		if (self.curr_t > t_span):
			self.curr_t = 0

		for obj in self.objects:
			obj.update(dt, f_ext=np.array([0,0,0]), τ_ext=np.array([0,0,0]))


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

	# kg*m^2
	I = [[9.759e-5,  -4.039e-6, -1.060e-7],
		 [-4.039e-6,  7.858e-5,  7.820e-9],
		 [ -1.060e-7, 7.820e-9,  1.743e-4]]

	# kg
	m = 0.05223934 
	# m --> check coordinates
	CM = [0.00215328, -0.00860001, -0.00038142]

	satellite = RidgidBody(I, m, CM, sat_mesh, init_vel=[-.02, .04, -.06], init_omg=[.6, -.4, .2])




	# Create Scene

	env = Environment([satellite])



	pyglet.app.run()

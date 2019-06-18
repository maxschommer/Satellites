import sklearn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.window import key
import ratcave as rc


# Make a 1d position solver
# Make a 3d position solver
# Make a 1d angle solver
# Make a 3d angle solver
# Integrate them
window = pyglet.window.Window()
scene = rc.Scene()

class RidgidBody():
	def __init__(self, I, m, CM,  mesh, init_pos=[[0,0,0,0,0,0],[0,0,0,0,0,0]]):
		self.I = I
		# print(I)
		# print(np.diag(I))
		# self.local_axis = np.linalg.eig(np.diag(I))
		# print(self.local_axis)
		self.m = m
		self.CM = CM
		self.mesh = mesh
		# init 

	def update(self, curr_t):
		# print(sol.sol(curr_t))
		self.mesh.position.x = sol.sol(curr_t)[0]  # dt is the time between frames
		self.mesh.position.y = sol.sol(curr_t)[2]
		self.mesh.position.z = sol.sol(curr_t)[4]


def f(t, A):
	# return 0
	return np.array([-.5*A[0], -.5*A[2], -.5*A[4]])
	# if y[0] < 0:
	# 	return -.5*y[0]
	# else:
	# 	return -.5*y[0]

def oscillator(t, A):
	# dydt = [[x', x''/m],
	#         [y', y''/m],
	#         [z', z''/m]
	f_ext = f(t, A)
	dydt = [[A[1], f_ext[0]/m],
			[A[3], f_ext[1]/m],
			[A[5], f_ext[2]/m]]
	return dydt
m = 1
init = np.asarray([.2, -.1, 0, .1, 0, 0])

t_span = 20
sol = solve_ivp(oscillator, [0, t_span], init,
				vectorized=True, 
				 dense_output=True)
t = np.linspace(0, t_span, 200)


class Environment():
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
			obj.update(self.curr_t)


	def move_camera(self, dt):
		  camera_speed = 20
		  if self.keys[key.LEFT]:
			  scene.camera.rotation.y -= camera_speed * dt
		  if self.keys[key.RIGHT]:
			  scene.camera.rotation.y += camera_speed * dt



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

I_inv = np.linalg.inv(I)

# kg
m = 0.05223934 
# m --> check coordinates
CM = [0.00215328, -0.00860001, -0.00038142]

satellite = RidgidBody(I, m, CM,  sat_mesh)




# Create Scene

env = Environment([satellite])



pyglet.app.run()
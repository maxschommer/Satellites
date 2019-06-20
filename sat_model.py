import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet.window import key
import ratcave as rc

from constraint import BallJointConstraint, HingeJointConstraint
from physics import RigidBody, Environment, MagneticDipole
from rendering import BodyActor, VectorActor, Stage



WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

dipole_moment = np.array([0, 0, 2]) # A*m^2
B_earth = np.array([35e-6, 0, 0]) # T



if __name__ == '__main__':

	I = [[9.759e-5,  -4.039e-6, -1.060e-7],
		 [-4.039e-6,  7.858e-5,  7.820e-9],
		 [ -1.060e-7, 7.820e-9,  1.743e-4]] # kg*m^2
	# I = [[1,0,0],
	# 	 [0,1,0],
	# 	 [0,0,1]]
	m = 0.05223934 # kg
	cm = [0.00215328, -0.00860001, -0.00038142] # m --> check coordinates
	# cm = [0, 0, .1]
	v0 = [.01,.005,-.02] # m/s
	ω0 = [-.3,.6, .1] # rad/s

	satellite_l = RigidBody(I, m, cm, init_position=[-.1,0,0], init_velocity=v0, init_angularv=ω0)
	satellite_c = RigidBody(I, m, cm, init_position=[0,0,0], init_velocity=v0, init_angularv=ω0)
	satellite_r = RigidBody(I, m, cm, init_position=[.1,0,0], init_velocity=v0, init_angularv=ω0)

	environment = Environment(
			bodies=[
				satellite_l,
				satellite_c,
				satellite_r,
			],
			constraints=[
				HingeJointConstraint(satellite_l, satellite_c, [.5,0,0], [-.5,0,0], [0,1,0]),
				HingeJointConstraint(satellite_c, satellite_r, [.5,0,0], [-.5,0,0], [0,1,0]),
			],
			external_impulsors=[
				MagneticDipole(satellite_l, dipole_moment, B_earth),
			])
	environment.solve(0, 15)

	stage = Stage([
		BodyActor(satellite_l, "ThinSatFrame->Frame"),
		BodyActor(satellite_c, "ThinSatFrame->Frame"),
		BodyActor(satellite_r, "ThinSatFrame->Frame"),
		# VectorActor(satellite_l, "xaxis", "Resources/arrow->Arrow"),
		# VectorActor(satellite_l, "yaxis", "Resources/arrow->Arrow"),
		# VectorActor(satellite_l, "zaxis", "Resources/arrow->Arrow"),
		# VectorActor(satellite_l, "velocity", "Resources/arrow->Arrow"),
		VectorActor(satellite_l, "angularv", "Resources/arrow->Arrow"),
	], environment)


	window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
	scene = rc.Scene(
			meshes=[a.mesh for a in stage.actors],
			camera=rc.Camera(position=(0, 0, .4)),
			bgColor=(1, 1, .9))

	@window.event
	def on_draw():
		with rc.default_shader:
			scene.draw()

	pyglet.clock.schedule(stage.update)

	pyglet.app.run()

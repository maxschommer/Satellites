import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet.window import key
import ratcave as rc

from constraint import Hinge
from locomotion import MagneticDipole, Thruster
from physics import Environment, RigidBody
from rendering import BodyActor, VectorActor, Stage
from sensor import Photodiode



WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


if __name__ == '__main__':

	I = [[9.759e-5,  -4.039e-6, -1.060e-7],
		 [-4.039e-6,  7.858e-5,  7.820e-9],
		 [ -1.060e-7, 7.820e-9,  1.743e-4]] # kg*m^2
	# I = [[1e-5,0,0],
	# 	 [0,1e-5,0],
	# 	 [0,0,1e-5]]
	m = 0.05223934 # kg
	cm = [0.00215328, -0.00860001, -0.00038142] # m --> check coordinates
	# cm = [0,0,0]
	# cm = [0, 0, .1]
	# v0 = [.01,.005,-.02] # m/s
	# ω0 = [-.05,.2, -.4] # rad/s
	v0 = [0, 0, 0]
	ω0 = [0, 0, 0]

	satellite_l = RigidBody(I, m, cm, init_position=[-.05,0,.05], init_velocity=v0, init_angularv=[0,0,0], init_rotation=[np.sqrt(.5), 0, np.sqrt(.5), 0])
	satellite_c = RigidBody(I, m, cm, init_position=[0,0,0], init_velocity=v0, init_angularv=ω0)
	satellite_r = RigidBody(I, m, cm, init_position=[.1,0,0], init_velocity=v0, init_angularv=ω0)

	environment = Environment(
		bodies=[
			satellite_l,
			satellite_c,
			satellite_r,
		],
		constraints=[
			# Hinge(satellite_l, satellite_c, [.05,0,-.005], [-.05,0,-.005], [0,1,0], [0,1,0]),
			# Hinge(satellite_c, satellite_r, [.05,0, .005], [-.05,0, .005], [0,1,0], [0,1,0]),
		],
		sensors=[
			Photodiode(satellite_l, [0,0,1]),
		],
		external_impulsors=[
			MagneticDipole(satellite_l, [0, 0, 1]),
			Thruster(satellite_r, [ .05,.05,0], [-1, 0, 0], lambda t: .002 if int(t)%3==0 else 0),
			Thruster(satellite_r, [-.05,.05,0], [ 1, 0, 0], lambda t: .002 if int(t)%3==1 else 0),
		],
		magnetic_field=[0, 0, 35e-6], # T
		solar_flux=[0, -1.361, 0], # W/m^2
	)
	environment.solve(0, 10)

	for sen in environment.sensors:
		plt.plot(environment.solution.t, sen.all_readings(environment.solution.y.transpose()))
	plt.show()

	stage = Stage([
		BodyActor(satellite_l, "ThinSatFrame->Frame"),
		BodyActor(satellite_c, "ThinSatFrame->Frame"),
		BodyActor(satellite_r, "ThinSatFrame->Frame"),
		# VectorActor(satellite_l, "xaxis", "Resources/arrow->Arrow"),
		# VectorActor(satellite_l, "yaxis", "Resources/arrow->Arrow"),
		# VectorActor(satellite_l, "zaxis", "Resources/arrow->Arrow"),
		VectorActor(satellite_l, "angularv", "Resources/arrow->Arrow"),
		VectorActor(satellite_c, "angularv", "Resources/arrow->Arrow"),
		VectorActor(satellite_r, "angularv", "Resources/arrow->Arrow"),
	], environment, speed=1)


	window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
	scene = rc.Scene(
			meshes=[a.mesh for a in stage.actors],
			camera=rc.Camera(position=(.1, .1, .5), rotation=(-12, 9, 3)),
			bgColor=(1, 1, .9))

	@window.event
	def on_draw():
		with rc.default_shader:
			scene.draw()

	pyglet.clock.schedule(stage.update)

	pyglet.app.run()

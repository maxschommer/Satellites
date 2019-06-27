import sklearn
import numpy as np
import pickle
import ratcave as rc

from constraint import Hinge
from control import Magnetostabilisation
from event import Launch
from locomotion import MagneticDipole, Magnetorker, Thruster, GimballedThruster, Drag
from physics import Environment, RigidBody
from rendering import BodyActor, VectorActor, Stage
from sensor import Photodiode, Magnetometer


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
	acetone = RigidBody(1.25e-7*np.identity(3), 5e-3, [0,0,0])
	magnetometer = Magnetometer(satellite_l)

	environment = Environment(
		bodies=[
			satellite_l,
			satellite_c,
			satellite_r,
			acetone,
		],
		constraints=[
			# Hinge(satellite_l, satellite_c, [.05,0,-.005], [-.05,0,-.005], [0,1,0], [0,1,0]),
			# Hinge(satellite_c, satellite_r, [.05,0, .005], [-.05,0, .005], [0,1,0], [0,1,0]),
		],
		sensors=[
			# Photodiode(satellite_l, [0,0,1]),
			magnetometer,
		],
		external_impulsors=[
			Magnetorker(satellite_l, Magnetostabilisation(magnetometer, [0, 0, 1])),
			Thruster(satellite_r, [ .05,.05,0], [-1, 0, 0], lambda t: .004 if int(t)%3==0 else 0),
			Thruster(satellite_r, [-.05,.05,0], [ 1, 0, 0], lambda t: .004 if int(t)%3==1 else 0),
			Drag(satellite_c, .001, [0,0,0])
		],
		events=[
			Launch(3, satellite_c, acetone, [0,0,0], [0,-.003,.5], [0,12.6,0])
		],
		magnetic_field=[0, 0, 35e-6], # T
		solar_flux=[0, -1.361, 0], # W/m^2
		air_velocity=[-7.8e3, 0, 0], # m/s
		air_density=1e-14, # kg/m^3
	)
	environment.solve(0, 15)

	stage = Stage([
		BodyActor(satellite_l, "ThinSatFrame->Frame"),
		BodyActor(satellite_c, "ThinSatFrame->Frame"),
		BodyActor(satellite_r, "ThinSatFrame->Frame"),
		BodyActor(acetone, "Justin->Justin", scale=30),
		# VectorActor(satellite_l, "xaxis", "Resources/arrow->Arrow"),
		# VectorActor(satellite_l, "yaxis", "Resources/arrow->Arrow"),
		# VectorActor(satellite_l, "zaxis", "Resources/arrow->Arrow"),
		VectorActor(satellite_l, "angularv", "Resources/arrow->Arrow"),
		VectorActor(satellite_c, "angularv", "Resources/arrow->Arrow"),
		VectorActor(satellite_r, "angularv", "Resources/arrow->Arrow"),
	], environment, speed=1)

	environment.shell()

	with open('../data.pkl', 'wb') as f:
		stage = pickle.dump(stage, f)
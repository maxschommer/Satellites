import sklearn
import numpy as np
import pickle
import ratcave as rc

from constraint import Hinge
from control import Magnetostabilisation
from event import Launch
from gmat_integration import SunTable, VelocityTable, AtmosphericTable
from locomotion import MagneticDipole, Magnetorker, Thruster, GimballedThruster, Drag
import matplotlib.pyplot as plt
from physics import Environment, RigidBody
from rendering import BodyActor, VectorActor, Stage
from sensor import Photodiode, Magnetometer

np.random.seed(0)


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
	q0 = np.random.randn(4)
	q0 = q0/np.linalg.norm(q0)
	v0 = [1, 0, 0] # m/s
	# ω0 = [-.05,.2, -.4] # rad/s
	ω0 = np.random.normal(0., 1., 3)

	# satellite_l = RigidBody(I, m, cm, init_position=[-.05,0,.05], init_velocity=v0, init_angularv=ω0, init_rotation=q0)#, init_rotation=[np.sqrt(.5), 0, np.sqrt(.5), 0])
	# satellite_c = RigidBody(I, m, cm, init_position=[0,0,0], init_velocity=v0, init_angularv=ω0, init_rotation=q0)
	# satellite_r = RigidBody(I, m, cm, init_position=[.1,0,0], init_velocity=v0, init_angularv=ω0, init_rotation=q0)
	# acetone = RigidBody(1.25e-7*np.identity(3), 5e-3, [0,0,0])
	h = .1
	r = h/12
	arrow = RigidBody([[(3*r**2+h**2)/12, 0, 0], [0, r**2/2, 0], [0, 0, (3*r**2+h**2)/12]], 1, [0, h/2, 0], init_rotation=[np.sqrt(.5),np.sqrt(.5),0,0])
	# magnetometer = Magnetometer(arrow)

	environment = Environment(
		bodies=[
			# satellite_l,
			# satellite_c,
			# satellite_r,
			# acetone,
			arrow,
		],
		constraints=[
			# Hinge(satellite_l, satellite_c, [.05,0,-.005], [-.05,0,-.005], [0,1,0], [0,1,0]),
			# Hinge(satellite_c, satellite_r, [.05,0, .005], [-.05,0, .005], [0,1,0], [0,1,0]),
		],
		sensors=[
			# Photodiode(satellite_l, [0,0,1]),
			# Photodiode(satellite_l, [0,0,-1]),
			# magnetometer,
		],
		external_impulsors=[
			# Magnetorker(satellite_l, Magnetostabilisation(magnetometer, [0, 1, 0], max_moment=.02)),
			# Thruster(satellite_r, [ .05,.05,0], [-1, 0, 0], lambda t: .004 if int(t)%3==0 else 0),
			# Thruster(satellite_r, [-.05,.05,0], [ 1, 0, 0], lambda t: .004 if int(t)%3==1 else 0),
			# Drag(satellite_c, .001, [0,0,0])
		],
		events=[
			# Launch(3, satellite_c, acetone, [0,0,0], [0,-.003,.5], [0,12.6,0])
		],
		magnetic_field=[0, 0, 35e-6], # T
		solar_flux=SunTable('../gmat_scripts/sunrise_sunset_table.txt'), # W/m^2
		air_velocity=VelocityTable('../gmat_scripts/ReportFile1.txt'), # m/s
		air_density=AtmosphericTable('../gmat_scripts/ReportFile1.txt'), # kg/m^3
	)
	environment.solve(0, 1000)

	stage = Stage([
		# BodyActor(satellite_l, "ThinSatFrame->Frame"),
		# BodyActor(satellite_c, "ThinSatFrame->Frame"),
		# BodyActor(satellite_r, "ThinSatFrame->Frame"),
		# BodyActor(acetone, "Justin->Justin", scale=30),
		# VectorActor(satellite_l, "angularv", "Resources/arrow->Arrow"),
		# VectorActor(satellite_c, "angularv", "Resources/arrow->Arrow"),
		# VectorActor(satellite_r, "angularv", "Resources/arrow->Arrow"),
		BodyActor(arrow, "Resources/arrow->Arrow"),
		# VectorActor(arrow, "angularv", "Resources/arrow->Arrow")
	], environment)

	environment.shell()

	with open('../saves/drag.pkl', 'wb') as f:
		stage = pickle.dump(stage, f)

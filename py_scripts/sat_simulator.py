import sklearn
import numpy as np
import pickle
import ratcave as rc

from constraint import Hinge
from event import Launch
from gmat_integration import MagneticTable, SunTable, VelocityTable, AtmosphericTable
from locomotion import MagneticDipole, Magnetorker, Thruster, GimballedThruster, Drag
import matplotlib.pyplot as plt
from physics import Environment, RigidBody
from sensor import Photodiode, Magnetometer, Magnetostabilisation


FILENAME = 'stabl0.pkl' # place to save

WINDOW_WIDTH = 800 # window properties
WINDOW_HEIGHT = 600

I_1 = [[9.759e-5,  -4.039e-6, -1.060e-7], # mass properties
	 [-4.039e-6,  7.858e-5,  7.820e-9],
	 [ -1.060e-7, 7.820e-9,  1.743e-4]] # kg*m^2
m_1 = 0.05223934 # kg
cm_1 = [0.00215328, -0.00860001, -0.00038142] # m
I_3 = [[3.166e-4, 7.726e-6, 1.088e-6],
	 [7.726e-6, 2.362e-3,-1.253e-6],
	 [1.088e-6,-1.253e-6, 2.675e-3]] # kg*m^2
m_3 = .318 # kg
cm_3 = [-1.4e-3, 1.0e-3, -2.2e-3] # m

np.random.seed(0) # make it reproduceable 


if __name__ == '__main__':
	q0 = np.random.randn(4) # pick some initial conditions
	q0 = q0/np.linalg.norm(q0)
	v0 = [1, 0, 0] # m/s
	ω0 = np.random.normal(0., 1., 3)

	bodies = { # declare all of the things
		# 'left_sat':  RigidBody(I_1, m_1, cm_1, init_position=[0,0, .01], init_angularv=[0,0,0], init_rotation=[1,0,0,0]),
		# 'center_sat':RigidBody(I_1, m_1, cm_1, init_position=[0,0,0], init_angularv=[0,0,0], init_rotation=[0,0,1,0]),
		# 'right_sat': RigidBody(I_1, m_1, cm_1, init_position=[0,0,-.01], init_angularv=[0,0,0], init_rotation=[1,0,0,0]),
		# 'acetone':   RigidBody(1.25e-7*np.identity(3), 5e-3, [0,0,0]),
		'satellites':RigidBody(I_3, m_3, cm_3, init_angularv=ω0, init_rotation=q0),
	}
	sensors = {
		'photo_0':Photodiode(bodies['satellites'], [0, 0, 1]),
		'photo_1':Photodiode(bodies['satellites'], [0, 1, 0]),
		'photo_2':Photodiode(bodies['satellites'], [0, 0,-1]),
		'photo_3':Photodiode(bodies['satellites'], [0,-1, 0]),
		'magnet':Magnetometer(bodies['satellites']),
	}
	constraints = [
		# Hinge(bodies['left_sat'],  bodies['center_sat'], [-.05,0,-.005], [.05,0,-.005], [0,1,0], [0,1,0]),
		# Hinge(bodies['center_sat'], bodies['right_sat'], [-.05,0, .005], [.05,0, .005], [0,1,0], [0,1,0]),
	]
	external_impulsors = [
		Magnetorker(bodies['satellites'], Magnetostabilisation(sensors['magnet'], max_moment=.02, axes=[1,1,0])),
		Drag(.003, [0,0,0]),
		# Thruster(bodies['left_sat'], [0, .05,0], [-1, 0, 0], lambda t: [.001,0,-.001,-.001,0,.001][int(t)%6]),
		# Thruster(bodies['left_sat'], [0,-.05,0], [ 1, 0, 0], lambda t: [.001,0,-.001,-.001,0,.001][int(t)%6]),
	]
	events = [
		# Launch(3, bodies['center_sat'], bodies['acetone'], [0,0,0], [0,-.003,.5], [0,12.6,0])
	]

	environment = Environment( # put them together in an Environment
		bodies, sensors, constraints, external_impulsors, events,
		magnetic_field=MagneticTable("../gmat_scripts/ReportFile1.txt"), # T
		solar_flux=SunTable("../gmat_scripts/sunrise_sunset_table.txt"), # W/m^2
		air_velocity=VelocityTable("../gmat_scripts/ReportFile1.txt"), # m/s
		air_density=AtmosphericTable("../gmat_scripts/ReportFile1.txt"), # kg/m^3
	)

	environment.solve(0, 180*60, method='LSODA') # run the simulation

	environment.shell() # strip away the unpicklable parts

	with open("../simulations/{}".format(FILENAME), 'wb') as f: # save the simulation with pickle
		pickle.dump(environment, f)

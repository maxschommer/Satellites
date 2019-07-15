import sklearn
import numpy as np
import pickle
import ratcave as rc

from constraint import Hinge
from event import Launch
from gmat_integration import MagneticTable, SunTable, VelocityTable, AtmosphericTable
from locomotion import PermanentMagnet, Magnetorker, Thruster, GimballedThruster, Drag
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

I_3 = [[ 445.e-6,   37.e-6,   23.e-6],
       [  37.e-6, 4119.e-6,    1.e-6],
       [  23.e-6,    1.e-6, 4555.e-6]] # kg*m^2
m_3 = .529 # kg
cm_3 = [-19.43e-3,-0.65e-3,-0.76e-3] # m


if __name__ == '__main__':
	for drag_coef in [1, .1, .01, 10]: # A*m^2
		print("c_D = {:.0e} mA m^2".format(drag_coef))
		for seed in range(0, 3):
			print("	i = {:02d}".format(seed))
			print("		setting up environment...")

			FILENAME = 'orient-{:.0e}-{:01d}.pkl'.format(drag_coef, seed)
			np.random.seed(seed) # make it reproduceable 

			q0 = np.random.randn(4) # pick some initial conditions
			q0 = q0/np.linalg.norm(q0)
			v0 = [1, 0, 0] # m/s
			ω0 = np.random.normal(0., 1., 3)

			bodies = { # declare all of the things
				# 'left_sat':  RigidBody(I_1, m_1, cm_1, init_position=[0,0, .01], init_angularv=[0,0,0], init_rotation=[1,0,0,0]),
				# 'center_sat':RigidBody(I_1, m_1, cm_1, init_position=[0,0,0], init_angularv=[0,0,0], init_rotation=[0,0,1,0]),
				# 'right_sat': RigidBody(I_1, m_1, cm_1, init_position=[0,0,-.01], init_angularv=[0,0,0], init_rotation=[1,0,0,0]),
				# 'vapor':     RigidBody(1.25e-7*np.identity(3), 5e-3, [0,0,0]),
				# 'dipole':    RigidBody([[1.981e-9,0,0],[.013e-9,0,0],[1.981e-9,0,0]], 1.053e-4, )
				'satellites':RigidBody(I_3, m_3, cm_3, init_angularv=ω0, init_rotation=q0),
			}
			sensors = {
				# 'photo_0':Photodiode(bodies['satellites'], [0, 0, 1]),
				# 'photo_1':Photodiode(bodies['satellites'], [0, 1, 0]),
				# 'photo_2':Photodiode(bodies['satellites'], [0, 0,-1]),
				# 'photo_3':Photodiode(bodies['satellites'], [0,-1, 0]),
				'magnet':Magnetometer(bodies['satellites']),
			}
			constraints = [
				# Hinge(bodies['left_sat'],  bodies['center_sat'], [-.05,0,-.005], [.05,0,-.005], [0,1,0], [0,1,0]),
				# Hinge(bodies['center_sat'], bodies['right_sat'], [-.05,0, .005], [.05,0, .005], [0,1,0], [0,1,0]),
			]
			external_impulsors = [
				Magnetorker(bodies['satellites'], Magnetostabilisation(sensors['magnet'], max_moment=.02, axes=[1,1,1])),
				# PermanentMagnet(bodies['satellites'], [0, 0, moment]),
				Drag(bodies, [drag_coef*.1*.01], [[0,0,0]]),
				# Thruster(bodies['left_sat'], [0, .05,0], [-1, 0, 0], lambda t: [.001,0,-.001,-.001,0,.001][int(t)%6]),
				# Thruster(bodies['left_sat'], [0,-.05,0], [ 1, 0, 0], lambda t: [.001,0,-.001,-.001,0,.001][int(t)%6]),
			]
			events = [
				# Launch(10200+1*i, bodies['satellites'], bodies['dipole'], [.02,-.040+.008*i,0], [0,0,.6], [63.,0,0]) for i in range(10),
				# Launch(30000, bodies['satellites'], bodies['vapor'], [0,0,0], [0,0,.1], [0,0,0]),
				# Launch(35000, bodies['satellites'], bodies['vapor'], [0,0,0], [0,0,.1], [0,0,0]),
			]

			environment = Environment( # put them together in an Environment
				bodies, sensors, constraints, external_impulsors, events,
				magnetic_field=MagneticTable("../gmat_scripts/ReportFile1.txt"), # T
				solar_flux=SunTable("../gmat_scripts/sunrise_sunset_table.txt"), # W/m^2
				air_velocity=VelocityTable("../gmat_scripts/ReportFile1.txt"), # m/s
				air_density=AtmosphericTable("../gmat_scripts/ReportFile1.txt"), # kg/m^3
			)

			print("		solving...")
			environment.solve(0, 180*60, method='LSODA') # run the simulation
			
			print("		saving as {}...".format(FILENAME))
			environment.shell() # strip away the unpicklable parts
			while True:
				try:
					with open("../simulations/{}".format(FILENAME), 'wb') as f: # save the simulation with pickle
						pickle.dump(environment, f, protocol=4)
				except MemoryError:
					print("		reducing...")
					environment.reduce()
				else:
					break

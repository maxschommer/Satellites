import sklearn
import numpy as np
import pickle
import ratcave as rc

from constraint import Hinge
from event import Launch
from gmat_integration import MagneticTable, SunTable, VelocityTable, AtmosphericTable
from locomotion import PermanentMagnet, Magnetorker, Thruster, GimballedThruster, Drag, Parachute
import matplotlib.pyplot as plt
from physics import Environment, RigidBody
from sensor import Photodiode, Magnetometer, BangBangBdot


FILENAME = 'test.pkl' # place to save

m_1 = 0.05223934 # kg
cm_1 = [0.00215328, -0.00860001, -0.00038142] # m
I_1 = [[9.759e-5,  -4.039e-6, -1.060e-7],
       [-4.039e-6,  7.858e-5,  7.820e-9],
       [ -1.060e-7, 7.820e-9,  1.743e-4]] # kg*m^2

m_3 = .529 # kg
cm_3 = [-19.26e-3, -5.96e-3, -0.86e-3] # m
v_3 = [[ 1.00, 0.09, 0.02],
	   [-0.09, 1.00,-0.01],
	   [-0.02, 0.00, 1.00]]
λ_3 = [0.26661841e-3, 2.15589278e-3, 2.41089260e-3] # kg*m^2

m_half = 1.003e-3 # kg
cm_half = [3.102e-3, -0.487e-3, -0.003e-3] # m
I_half = [[1.169e+1, 1.946e+0, 2.295e-4],
	      [1.946e+0, 5.316e+1,-8.537e-3],
	      [2.295e-4,-8.537e-3, 5.333e+1]] # kg*m^2


if __name__ == '__main__':
	for num_magnets in [0, 8, 1]:
		for seed in [17, 19, 21, 23, 25, 27]:
			print("i = {:02d}".format(seed))
			for drag_coef in [.1, 1, 10]:
				print("	c_D = {:.0e}".format(drag_coef))
				print("		setting up environment...")

				FILENAME = 'magnet_{:.0e}_{:02d}_{:1d}.pkl'.format(drag_coef, seed, num_magnets)
				np.random.seed(seed) # make it reproduceable 

				q0 = np.random.randn(4) # pick some initial conditions
				q0 = q0/np.linalg.norm(q0)
				v0 = [0, 0, 0] # m/s
				ω0 = np.random.normal(0., .5, 3) if seed != 17 else np.zeros(3)

				bodies = { # declare all of the things
					# 'left_sat':  RigidBody(m_1, cm_1, I_1, init_position=[0,0, .01], init_angularv=[0,0,0], init_rotation=[1,0,0,0]),
					# 'center_sat':RigidBody(m_1, cm_1, I_1, init_position=[0,0,0], init_angularv=[0,0,0], init_rotation=[0,0,1,0]),
					# 'right_sat': RigidBody(m_1, cm_1, I_1, init_position=[0,0,-.01], init_angularv=[0,0,0], init_rotation=[1,0,0,0]),
					# 'vapor':     RigidBody(5e-3, [0,0,0], 1.25e-7*np.identity(3)),
					# 'dipole':    RigidBody(1.053e-3, [0,0,0], [[1.981e-8,0,0],[0,.013e-8,0],[0,0,1.981e-8]]),
					'satellites':RigidBody(m_3, cm_3, axes=v_3, moments=λ_3, init_angularv=ω0, init_rotation=q0),
					# 'half-barrel':RigidBody(m_half, cm_half, I_half, init_angularv=[126,0,0]),
				}
				sensors = {
					# 'photo_0': Photodiode(bodies['satellites'], [0, 0, 1]),
					# 'photo_1': Photodiode(bodies['satellites'], [0, 1, 0]),
					# 'photo_2': Photodiode(bodies['satellites'], [0, 0,-1]),
					# 'photo_3': Photodiode(bodies['satellites'], [0,-1, 0]),
					'magnetic':Magnetometer(bodies['satellites']),
				}
				constraints = [
					# Hinge(bodies['left_sat'],  bodies['center_sat'], [-.05,0,-.005], [.05,0,-.005], [0,1,0], [0,1,0]),
					# Hinge(bodies['center_sat'], bodies['right_sat'], [-.05,0, .005], [.05,0, .005], [0,1,0], [0,1,0]),
				]
				external_impulsors = [
					Magnetorker(bodies['satellites'], BangBangBdot(sensors['magnetic'], max_moment=.2, axes=[1,1,1])),
					PermanentMagnet(bodies['satellites'], [0, 0, num_magnets*54.1*(1/8/2)**2*(1/8)]), # put diameter and height in inches into paretheses
					Drag(bodies, [drag_coef*np.array([.1*.01, .3*.01, .1*.3])], [[0,0,0]]),
					Parachute(bodies['satellites'], drag_coef*.01, [-.15, 0, -.005])
					# Drag(bodies, [[8.9*9.6e-6, 8.9*26.7e-6, 9.6*26.7e-6]], [[0,0,0]]),
				]
				events = [
					# *[Launch(6600+1*i, bodies['satellites'], bodies['dipole'], [.02,-.040+.008*i,0], [0,0,.6], [63.,0,0], actually_do_it=False) for i in range(1)],
					# Launch(30000, bodies['satellites'], bodies['vapor'], [0,0,0], [0,0,.1], [0,0,0]),
					# Launch(35000, bodies['satellites'], bodies['vapor'], [0,0,0], [0,0,.1], [0,0,0]),
				]

				environment = Environment( # put them together in an Environment
					bodies, sensors, constraints, external_impulsors, events,
					magnetic_field=MagneticTable("../gmat_scripts/ReportFile1.txt"), # T
					solar_flux=SunTable("../gmat_scripts/sunrise_sunset_table.txt"), # W/m^2
					air_velocity=VelocityTable("../gmat_scripts/ReportFile1.txt"), # m/s
					air_density=AtmosphericTable("../gmat_scripts/ReportFile1.txt"), # kg/m^3
					verbose=False,
				)

				print("		solving...")
				environment.solve(0, 20000, method='LSODA') # run the simulation
				
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

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pyquaternion import Quaternion
import seaborn as sns
sns.set_style('whitegrid')


FILENAME = 'magnet-{:.0e}-{:01d}.pkl'


if __name__ == '__main__':
	ωs = []
	Es = []
	ϴs = []
	param = .1
	for seed in [5, 4, 3, 2, 1, 0]:
		print('loading', param, seed)
		with open('../simulations/{}'.format(FILENAME.format(param, seed)), 'rb') as f:
			env = pickle.load(f)

		T, Y = env.solution

		key = 'satellites'
		i = 13*env.bodies[key].num
		ωs.append([])
		Es.append([])
		ϴs.append([])
		for t, y in zip(T, Y):
			q = Quaternion(y[i+3:i+7])
			z_prime = q.rotate([0,0,1])
			R = q.rotation_matrix
			I_inv = np.matmul(np.matmul(R, env.bodies[key].I_inv), R.transpose())
			v = -env.get_air_velocity(t)
			ω = np.matmul(I_inv, y[i+10:i+13])
			ωs[-1].append(np.linalg.norm(ω))
			Es[-1].append(1/2*np.matmul(np.matmul(y[i+10:i+13], I_inv), y[i+10:i+13]))
			ϴs[-1].append(np.arccos(np.dot(z_prime, v/np.linalg.norm(v))))
			# ϴs[-1].append(np.arccos(np.dot(q.rotate([1,0,0]), ω/np.linalg.norm(ω))))

	ωs = np.array(ωs)
	Es = np.array(Es)
	ϴs = np.array(ϴs)
	env = None # clear up some memory

	order = np.argsort(Es[:,0])
	# order = np.arange(0, len(Es[:,0]))

	plt.figure()
	plt.title("Rotational kinetic energy (c_D = {:n})".format(param))
	for z, i in enumerate(order):
		plt.semilogy(T/3600, Es[i,:], linewidth=.7, zorder=10-z)
	plt.ylabel("Energy (J)")
	plt.xlabel("Time (hr)")
	# plt.legend()

	plt.figure()
	plt.title("Angular velocity magnitude (c_D = {:n})".format(param))
	for z, i in enumerate(order):
		plt.semilogy(T/3600, ωs[i,:]/(2*np.pi)*60, linewidth=.7, zorder=10-z)
	plt.ylabel("Angular velocity (rpm)")
	plt.xlabel("Time (hr)")
	# plt.legend()

	plt.figure()
	plt.title("Angle between launcher axis and orbital path (c_D = {:n})".format(param))
	for z, i in enumerate(order):
		plt.plot(T/3600, np.degrees(ϴs[i,:]), linewidth=.7, zorder=10-z)
	plt.ylabel("Angle (°)")
	plt.xlabel("Time (hr)")
	plt.yticks(np.linspace(0, 180, 7))
	# plt.legend()

	# plt.figure()
	# for i in range(4):
	# 	plt.plot(T/3600, environment.sensors['photo_{:d}'.format(i)].all_readings(T, Y), label="Sensor {:d}".format(i))
	# plt.ylabel("Photodiode reading (W/m^2)")
	# plt.xlabel("Time (hr)")
	# plt.legend()

	plt.show()
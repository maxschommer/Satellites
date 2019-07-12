import matplotlib.pyplot as plt
import numpy as np
import pickle
from pyquaternion import Quaternion
import seaborn as sns
sns.set_style('whitegrid')


FILENAME = 'stabl{}.pkl'


if __name__ == '__main__':
	ωs = []
	Es = []
	ϴs = []
	for j in range(6):
		with open('../simulations/{}'.format(FILENAME.format(j)), 'rb') as f:
			env = pickle.load(f)

		T = np.linspace(0, env.max_t, 2000)
		Y = env.solution(T)

		key = 'satellites'
		i = 13*env.bodies[key].num
		ωs.append([])
		Es.append([])
		ϴs.append([])
		for t, y in zip(T, Y.transpose()):
			q = Quaternion(y[i+3:i+7])
			z_prime = q.rotate([0,0,1])
			R = q.rotation_matrix
			I_inv = np.matmul(np.matmul(R, env.bodies[key].I_inv), R.transpose())
			v = -env.get_air_velocity(t)
			ωs[-1].append(np.linalg.norm(np.matmul(I_inv, y[i+10:i+13])))
			Es[-1].append(1/2*np.matmul(np.matmul(y[i+10:i+13], I_inv), y[i+10:i+13]))
			ϴs[-1].append(np.arccos(np.dot(z_prime, v/np.linalg.norm(v))))

	ωs = np.array(ωs)
	Es = np.array(Es)
	ϴs = np.array(ϴs)
	env = None # clear up some memory

	order = np.argsort(Es[:,0])

	# plt.figure()
	# plt.title("Energy decay")
	# for z, i in enumerate(order):
	# 	plt.semilogy(T/3600, Es[i,:], linewidth=.7, zorder=10-z)
	# plt.ylabel("Rotational energy (J)")
	# plt.xlabel("Time (hr)")

	plt.figure()
	plt.title("Rotational deceleration")
	for z, i in enumerate(order):
		plt.semilogy(T/3600, ωs[i,:]/(2*np.pi)*60, linewidth=.7, zorder=10-z)
	plt.ylabel("Angular velocity (rpm)")
	plt.xlabel("Time (hr)")

	plt.figure()
	plt.title("Angle stabilization")
	for z, i in enumerate(order):
		plt.plot(T/3600, np.degrees(ϴs[i,:]), linewidth=.7, zorder=10-z)
	plt.ylabel("Launcher angle (°)")
	plt.xlabel("Time (hr)")

	# plt.figure()
	# for i in range(4):
	# 	plt.plot(T/3600, environment.sensors['photo_{:d}'.format(i)].all_readings(T, Y), label="Sensor {:d}".format(i))
	# plt.ylabel("Photodiode reading (W/m^2)")
	# plt.xlabel("Time (hr)")
	# plt.legend()

	plt.show()
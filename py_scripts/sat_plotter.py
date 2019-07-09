import matplotlib.pyplot as plt
import numpy as np
import pickle
from pyquaternion import Quaternion
import seaborn as sns
sns.set_style('whitegrid')


FILENAME = 'stabl0.pkl'


if __name__ == '__main__':
	with open('../simulations/{}'.format(FILENAME), 'rb') as f:
		environment = pickle.load(f)

	T = np.linspace(0, environment.max_t, 100)
	Y = np.array([environment.solution(t) for t in T])

	key = 'satellites'
	i = 13*environment.bodies[key].num
	E = []
	ϴ = []
	for t, y in zip(T, Y):
		q = Quaternion(y[i+3:i+7])
		z_prime = q.rotate([0,0,1])
		R = q.rotation_matrix
		I_inv = np.matmul(np.matmul(R, environment.bodies[key].I_inv), R.transpose())
		v = -environment.get_air_velocity(t)
		E.append(1/2*np.matmul(np.matmul(y[10:13], I_inv), y[10:13]))
		ϴ.append(np.arccos(np.dot(z_prime, v/np.linalg.norm(v))))

	# plt.figure()
	# plt.semilogy(T/3600, E*1e-3)
	# plt.ylabel("Rotational energy (mJ)")
	# plt.xlabel("Time (hr)")

	plt.figure()
	plt.plot(T/3600, np.degrees(ϴ))
	plt.ylabel("Launcher angle (°)")
	plt.xlabel("Time (hr)")

	plt.figure()
	for i in range(4):
		plt.plot(T/3600, environment.sensors['photo_{:d}'.format(i)].all_readings(T, Y), label="Sensor {:d}".format(i))
	plt.ylabel("Photodiode reading (W/m^2)")
	plt.xlabel("Time (hr)")
	plt.legend()

	plt.show()
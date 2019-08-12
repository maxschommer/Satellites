import matplotlib.pyplot as plt
import numpy as np
import pickle
from pyquaternion import Quaternion
import seaborn as sns
sns.set_style('whitegrid')


FILENAME = 'magnet_{:.0e}_{:02d}_{:1d}.pkl'


if __name__ == '__main__':
	ωs = []
	Es = []
	ϴs = []
	As = []
	param = 10
	num_magnets = 1
	for seed in [17, 19, 21, 23, 25, 27]:
		print('loading', param, seed)
		with open('../simulations/{}'.format(FILENAME.format(param, seed, num_magnets)), 'rb') as f:
			env = pickle.load(f)

		T, Y = env.solution

		key = 'satellites'
		i = 13*env.bodies[key].num
		ωs.append([])
		Es.append([])
		ϴs.append([])
		As.append([])
		for t, y in zip(T, Y):
			q = Quaternion(y[i+3:i+7])
			z_prime = q.rotate([0,0,1])
			R = q.rotation_matrix
			I_inv = np.matmul(np.matmul(R, env.bodies[key].I_inv), R.transpose())
			v = -env.get_air_velocity(t)
			# ω = np.matmul(I_inv, y[i+10:i+13])
			# ωs[-1].append(np.linalg.norm(ω))
			Es[-1].append(1/2*np.matmul(np.matmul(y[i+10:i+13], I_inv), y[i+10:i+13]))
			ϴs[-1].append(np.arccos(np.dot(z_prime, v/np.linalg.norm(v))))
			As[-1].append(np.arccos(np.dot(z_prime, env.get_down(t))))
			# ϴs[-1].append(np.arccos(np.dot(q.rotate([1,0,0]), ω/np.linalg.norm(ω))))

	ωs = np.array(ωs)
	Es = np.array(Es)
	ϴs = np.array(ϴs)
	As = np.array(As)

	order = np.argsort(-Es[:,0])
	# order = np.arange(0, len(Es[:,0]))

	sns.set_palette(sns.cubehelix_palette(6, start=.15, rot=-.40, light=.7, dark=.2))

	# plt.figure()
	# plt.title("Rotational kinetic energy (c_D = {:n})".format(param))
	# for z, i in enumerate(order):
	# 	plt.semilogy(T/3600, Es[i,:], linewidth=.9, zorder=10-z)
	# plt.ylabel("Energy (J)")
	# plt.xlabel("Time (hr)")
	# # plt.legend()

	# plt.figure()
	# plt.title("Angular velocity magnitude (c_D = {:n})".format(param))
	# for z, i in enumerate(order):
	# 	plt.semilogy(T/3600, ωs[i,:]/(2*np.pi)*60, linewidth=.9, zorder=10-z)
	# plt.ylabel("Angular velocity (rpm)")
	# plt.xlabel("Time (hr)")
	# # plt.legend()

	plt.figure()
	plt.title("Angle between launcher axis and orbital (c_D = {:n})".format(param))
	for z, i in enumerate(order):
		plt.plot(T/3600, np.degrees(ϴs[i,:]), linewidth=.9, zorder=10-z)
	plt.ylabel("Angle (°)")
	plt.xlabel("Time (hr)")
	plt.yticks(np.linspace(0, 180, 7))
	# plt.legend()

	plt.figure()
	plt.title("Angle between launcher axis and vertical (c_D = {:n})".format(param))
	for z, i in enumerate(order):
		plt.plot(T/3600, np.degrees(As[i,:]), linewidth=.9, zorder=10-z)
	plt.ylabel("Angle (°)")
	plt.xlabel("Time (hr)")
	plt.yticks(np.linspace(0, 180, 7))
	# plt.legend()

	# for inds in [(np.abs(T-2264)<20), (np.abs(T-4384)<20)]:
	# 	plt.figure()
	# 	sensor_readings = [env.sensors['photo_{:d}'.format(i)].all_readings(T[inds], Y[inds]) for i in range(4)]
	# 	for i in range(4):
	# 		plt.plot(T[inds], sensor_readings[i], label=["Top", "Left", "Bottom", "Right"][i])
	# 	plt.ylabel("Photodiode reading (W/m^2)")
	# 	plt.xlabel("Time (s)")
	# 	plt.legend()

	plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pyglet
from pyglet.window import key
from pyquaternion import Quaternion
import ratcave as rc


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


def look_at(target, source=[0, 0, -1], roll=0):
	""" Return a Ratcave Rotation object that makes the camera look in the given direction. """
	target = np.array(target)/np.linalg.norm(target)
	source = np.array(source)/np.linalg.norm(source)
	q = Quaternion(axis=np.cross(source, target), angle=np.arccos(np.dot(target, source)))
	q = Quaternion(axis=target, degrees=roll)*q
	return rc.RotationQuaternion(*q).to_euler(units='deg')


if __name__ == '__main__':
	with open('../saves/rect5.pkl', 'rb') as f:
		stage = pickle.load(f)

	stage.load_resources()
	stage.speed = 1

	# T = np.linspace(0, stage.environment.max_t, 216)
	# Y = np.array([stage.environment.solution(t) for t in T])
	# plt.figure()
	# plt.plot(T, np.sqrt(Y[:,3]**2+Y[:,4]**2+Y[:,5]**2+Y[:,6]**2))
	# plt.show()
	# plt.figure()
	# for sen in stage.environment.sensors:
	# 	plt.plot(T, sen.all_readings(T, Y))
	# plt.show()

	# stage.environment.max_t = 5.77

	scene = rc.Scene(
		meshes=[a.mesh for a in stage.actors],
		camera=rc.Camera(position=(-.5, -.1, .1), rotation=look_at([5, 1, -1], roll=79)),
		# camera=rc.Camera(position=(-1.5, 0, 0), rotation=(0, -90, 0), projection=rc.PerspectiveProjection(fov_y=15, aspect=WINDOW_WIDTH/WINDOW_HEIGHT)),
		light=rc.Light(position=(0., 5., 1.)),
		bgColor=(1, 1, .9))

	window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

	@window.event
	def on_draw():
		with rc.default_shader:
			scene.draw()

	pyglet.clock.schedule(stage.update)

	pyglet.app.run()

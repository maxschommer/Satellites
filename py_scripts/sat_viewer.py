import matplotlib.pyplot as plt
import numpy as np
import pickle
import pyglet
from pyglet.window import key
import ratcave as rc


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


with open('../data.pkl', 'rb') as f:
	stage = pickle.load(f)

stage.load_resources()

# print(stage.environment.solution)
# T = np.linspace(0, stage.environment.max_t)
# Y = np.array([stage.environment.solution(t) for t in T])
# plt.plot(T, Y)
# plt.show()

# for sen in environment.sensors:
# 	plt.plot(environment.solution.t, sen.all_readings(environment.solution.y.transpose()))
# plt.show()

scene = rc.Scene(
	meshes=[a.mesh for a in stage.actors],
	# camera=rc.Camera(position=(.1, .1, .5), rotation=(-12, 9, 3)),
	camera=rc.Camera(position=(.0, .0, 1.5), rotation=(0, 0, 0), projection=rc.PerspectiveProjection(fov_y=15, aspect=WINDOW_WIDTH/WINDOW_HEIGHT)),
	light=rc.Light(position=(0., 2., 1.)),
	bgColor=(1, 1, .9))

window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

@window.event
def on_draw():
	with rc.default_shader:
		scene.draw()

pyglet.clock.schedule(stage.update)

pyglet.app.run()
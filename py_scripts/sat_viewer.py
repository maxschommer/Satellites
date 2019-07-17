import numpy as np
import pickle
import pyglet
from pyglet.window import key
from pyquaternion import Quaternion
import ratcave as rc

from rendering import Stage, BodyActor, VectorActor, VectorFieldActor


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

FILENAME = 'magnet-1e-01-1.pkl'


def look_at(target, source=[0, 0, -1], roll=0):
	""" Return a Ratcave Rotation object that makes the camera look in the given direction. """
	target = np.array(target)/np.linalg.norm(target)
	source = np.array(source)/np.linalg.norm(source)
	q = Quaternion(axis=np.cross(source, target), angle=np.arccos(np.dot(target, source)))
	q = Quaternion(axis=target, degrees=roll)*q
	return rc.RotationQuaternion(*q).to_euler(units='deg')


if __name__ == '__main__':
	with open('../simulations/{}'.format(FILENAME), 'rb') as f: # load the desired save
		environment = pickle.load(f)

	stage = Stage([ # construct the stage with which to render the simulation
		# BodyActor('left_sat', "ThinSatFrame->Frame"),
		# BodyActor('center_sat', "ThinSatFrame->Frame"),
		# BodyActor('right_sat', "ThinSatFrame->Frame"),
		# BodyActor('acetone', "Justin->Justin", scale=30),
		BodyActor('satellites', "ThinSatAsm->ThinSatAsm"),
		# VectorActor('left_sat', "angularv", "Resources/arrow->Arrow"),
		# VectorActor('center_sat', "angularv", "Resources/arrow->Arrow"),
		# VectorActor('right_sat', "angularv", "Resources/arrow->Arrow"),
		# VectorActor('satellites', "angularv", "Resources/arrow->Arrow"),
		VectorFieldActor(environment.air_velocity, "Resources/arrow->Arrow", 'satellites'),
	], environment, speed=300)

	scene = rc.Scene( # build the ratcave scene
		meshes=[a.mesh for a in stage.actors],
		camera=rc.Camera(position=(-.5, -.1, .1), rotation=look_at([5, 1, -1], roll=79)),
		light=rc.Light(position=(0., 5., 1.)),
		bgColor=(1, 1, .9))

	window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT) # open the pyglet window

	@window.event # queue up the drawing process
	def on_draw():
		with rc.default_shader:
			scene.draw()
	pyglet.clock.schedule(stage.update)

	def move_camera(dt): # optionally, queue up a camera moving process
		origin = stage.environment.global_cm(stage.t)
		scene.camera.position.xyz = origin + [-.5,-.1, .1]
	pyglet.clock.schedule(move_camera)

	pyglet.app.run() # start the viewer!

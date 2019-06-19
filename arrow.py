import pyglet
from pyglet.gl import *
import numpy as np
def coordSysArrow(x1, y1, z1, x2, y2, z2):
	# print("Hi")
	glPushMatrix()
	glPushAttrib( GL_POLYGON_BIT ) # includes GL_CULL_FACE
	glDisable(GL_CULL_FACE) # draw from all sides

	# Size of cone in arrow:
	coneFractionAxially = 0.2
	coneFractionRadially = 0.1

	# Calculate cone parameters:
	v = np.array((x2-x1, y2-y1, z2-z1))
	norm_of_v = np.sqrt( np.dot(v,v) )
	coneHgt = coneFractionAxially * norm_of_v
	coneRadius = coneFractionRadially * norm_of_v
	vConeLocation = (1.0-coneFractionAxially) * v

	# Construct transformation matrix
	mat44 = np.eye(4)
	normalized_v = v/norm_of_v

	mat44[0,0] = normalized_v[0]
	mat44[1,1] = normalized_v[1]
	mat44[2,2] = normalized_v[2]

	# -----------------------
	#   Draw line + cone
	# -----------------------
	# Draw single line:
	glBegin(GL_LINES)
	glVertex3f(x1, y1, z1) # from
	glVertex3f(x2, y2, z2) # to
	glEnd() # GL_LINES

	# Move and rotate in position:
	glTranslatef( *vConeLocation )
	if 0: # turn on/off here
		#glLoadIdentity()
		glMultMatrixf( mat44 ) #  <===== PROBLEM HERE?!?! WHAT?

	# Make a cone!
	cone_obj = gluNewQuadric();
	# gluCylinder(gluNewQuadr, Radius_base, Radius_top,
	#               height, slices, stacks)
	gluCylinder(cone_obj, 0, coneRadius,\
		coneHgt, 8, 1);

	glPopAttrib() # GL_CULL_FACE
	glPopMatrix()


window = pyglet.window.Window()
label = pyglet.text.Label('Hello, world',
					  font_name='Times New Roman',
					  font_size=36,
					  x=window.width//2, y=window.height//2,
					  anchor_x='center', anchor_y='center')
@window.event
def on_draw():
	
	window.clear()
	coordSysArrow(0,0,0,-1,-1,-1)
	# label.draw()
pyglet.app.run()
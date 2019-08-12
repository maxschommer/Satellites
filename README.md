# Satellites

Scripts, optimizations and simulations for ThinSats Cosmic DC, Lt. Surge, and Fallout Boy

The simulation is `sat_simulator.py`. It creates an `Environment` object, which takes a `dict` of `RigidBody` objects, a dict of `Sensor` objects, a `list` of `Constraint` objects, a `list` of `Impulsor` objects, and a `list` of `Event` objects:
~~~~python
environment = Environment(
	bodies={
		'bod0':RigidBody(...),
		'bod1':RigidBody(...),
	},
	sensors={
		'sen0':Magnetometer(...),
	},
	constraints=[
		Hinge(...),
	],
	external_impulsors=[
		Thruster(...),
	],
	events=[
		Launch(...),
	],
)
~~~~
Values or `Tables` can also be passed in for environmental quantities like magnetic field and air density. See the class definition for more details.

Once the `Environment` exists, the solution is obtained by simply calling
~~~~python
environment.solve(0, 15)
~~~~
or however many seconds you want. The solution is saved by pickling the `Environment`.
~~~~python
with open("path/to/your.file", 'wb') as f: # save the simulation with pickle
	pickle.dump(environment, f)
~~~~

The `sat_viewer.py` script then displays that solution. Pass in a matching filename at the top, and then customize the `Stage` object to determine how things are rendered. `Stage` takes a list of `Actors` and the `Environment`. Each `Actor` represents one visual object on screen: a rendering of a body from the environment (which can be referenced by the string that keyed in in the dict in `sat_simulator.py`), an arrow representing a vector quantity of a body, or an arrow representing a vector quantity of the environment like the magnetic field.
~~~~python
stage = Stage([ # construct the stage with which to render the simulation
	BodyActor('bod0', "ThinSatAsm->ThinSatAsm"),
	BodyActor('bod1', "HalfBarrel->Half-Barrel"),
	VectorActor('bod0', "velocity", "Resources/arrow->Arrow"),
	VectorFieldActor(environment.magnetic_field, "Resources/arrow->Arrow"),
], environment)
~~~~
Additional arguments such as time lapse speed and specific time interval to loop can also be passed. See the class definition for more details.

You can also plot things with `sat_plotter.py`, which just loads a simulation and extracts some useful quantities to plot.

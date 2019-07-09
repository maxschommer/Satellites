# Satellites

Scripts, optimizations and simulations for ThinSats Cosmic DC, Lt. Surge, and Fallout Boy

The simulation is sat_simulator.py. It creates an Environment object, which takes a dict of RigidBody objects, a dict of Sensor objects, a list of Constraint objects, a list of Impulsor objects, and a list of Event objects. Once the Environment exists, the solution is obtained by simply calling
~~~~python
environment.solve(0, 15)
~~~~
or however many seconds you want.

The sat_viewer script then displays that solution. Pass in a matching filename at the top, and then customize the Stage object to determine how things are rendered. Stage takes a list of Actors, which each represent one visual object on screen: a rendering of a body from the environment (which can be referenced by the string that keyed in in the dict in sat_simulator.py), an arrow representing a vector quantity of a body, or an arrow representing a vector quantity of the environment like the magnetic field.

You can also plot things with sat_plotter.py, which just loads a simulation and extracts some useful quantities to plot.

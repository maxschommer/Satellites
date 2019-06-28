import numpy as np
import pandas as pd


""" COORDINATE CONVENTION:
	The coordinate directions are the same as in GMAT.
	The satellite is the origin (quasi-inertial). z+ is north.
	Initiall, x+ is roughly up and y+ is roughly east.
	The sun, for simplicity's sake, comes from y+.
"""


class SunTable:
	""" An object to read sunwend tables and thus estimate the solar flux vector. """
	def __init__(self, filename, solar_constant=1.361e+3*np.array([0, -1, 0])):
		self.table = pd.read_csv(filename, sep='   ', names=['timestamp', 'time', 'type'], engine='python')
		self.solar_constant = solar_constant

	def get_value(self, t):
		for event in self.table.itertuples():
			if event.time > t: # get the next event
				if event.type == "Sunset": # if the sun is going down
					return self.solar_constant
				else:
					return np.zeros(3)


class VelocityTable:
	""" An object to read GMAT reports and thus estimate the relative air speed. """
	def __init__(self, filename):
		self.table = pd.read_csv(filename, sep=r'\s\s+', engine='python')
		self.table = self.table.rename(lambda s: s.split('.')[-1], axis='columns')

	def get_value(self, t):
		for event in self.table.itertuples():
			if event.ElapsedSecs > t: # find the next event
				i = event.Index - 1
				t0 = self.table.iloc[i].ElapsedSecs # and the last one
				t1 = self.table.iloc[i+1].ElapsedSecs
				v0 = np.array([self.table.iloc[i].VX,   self.table.iloc[i].VY,   self.table.iloc[i].VZ])
				v1 = np.array([self.table.iloc[i+1].VX, self.table.iloc[i+1].VY, self.table.iloc[i+1].VZ])
				return -1e3*((t - t0)/(t1 - t0)*(v1 - v0) + v0) # and commit linear interpolation (negate because we want air velocity, not satellite velocity)


class AtmosphericTable:
	""" An object to read GMAT reports and thus estimaet the air density. """
	def __init__(self, filename):
		self.table = pd.read_csv(filename, sep=r'\s\s+', engine='python')
		self.table = self.table.rename(lambda s: s.split('.')[-1], axis='columns')

	def get_value(self, t):
		for event in self.table.itertuples():
			if event.ElapsedSecs > t: # find the next event
				i = event.Index - 1
				t0 = self.table.iloc[i].ElapsedSecs # and the last one
				t1 = self.table.iloc[i+1].ElapsedSecs
				ρ0 = self.table.iloc[i].AtmosDensity
				ρ1 = self.table.iloc[i+1].AtmosDensity
				return 1e-9*((t - t0)/(t1 - t0)*(ρ1 - ρ0) + ρ0) # and commit linear interpolation (negate because we want air velocity, not satellite velocity)

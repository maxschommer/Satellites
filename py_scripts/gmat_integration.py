from collections import OrderedDict
import numpy as np
import pandas as pd


""" COORDINATE CONVENTION:
	The coordinate directions are the same as in GMAT.
	The satellite is the origin (quasi-inertial). z+ is north.
	Initiall, x+ is roughly up and y+ is roughly east.
	The sun, for simplicity's sake, comes from y+.
"""


class Table:
	""" An object that yields values at different times. """
	def __init__(self):
		self.mem = OrderedDict()

	def get_value(self, t):
		""" Return the value at this time. """
		if t in self.mem:
			return self.mem[t]
		else:
			self.mem[t] = self._value(t)
			if len(self.mem) > 6:
				self.mem.popitem(last=False)
			return self.mem[t]

	def _value(self, t):
		""" The part of get_value that the subclasses can override. """
		raise NotImplementedError("Subclasses should override.")


class SunTable(Table):
	""" An object to read sunwend tables and thus estimate the solar flux vector. """
	def __init__(self, filename, solar_constant=1.361e+3*np.array([0, -1, 0])):
		super().__init__()
		self.table = pd.read_csv(filename, sep='   ', names=['timestamp', 'ElapsedSecs', 'type'], engine='python')
		self.solar_constant = solar_constant

	def _value(self, t):
		next_wend = self.table.iloc[self.table.ElapsedSecs.searchsorted(t)] # get the first event after this time
		if next_wend.type == "Sunset": # if the sun is going down
			return self.solar_constant # it must be up now
		else: # otherwise
			return np.zeros(3) # it must be down


class VelocityTable(Table):
	""" An object to read GMAT reports and thus estimate the relative air speed. """
	def __init__(self, filename):
		super().__init__()
		self.table = pd.read_csv(filename, sep=r'\s\s+', engine='python')
		self.table = self.table.rename(lambda s: s.split('.')[-1], axis='columns')

	def _value(self, t):
		i = self.table.ElapsedSecs.searchsorted(t) - 1 # get the next event
		t0 = self.table.iloc[i].ElapsedSecs # and the last one
		t1 = self.table.iloc[i+1].ElapsedSecs
		v0 = np.array(self.table.iloc[i][['VX', 'VY', 'VZ']])
		v1 = np.array(self.table.iloc[i+1][['VX', 'VY', 'VZ']])
		return -((t - t0)/(t1 - t0)*(v1 - v0) + v0)*1e+3 # and commit linear interpolation (negate because we want air velocity, not satellite velocity)


class AtmosphericTable(Table):
	""" An object to read GMAT reports and thus estimaet the air density. """
	def __init__(self, filename):
		super().__init__()
		self.table = pd.read_csv(filename, sep=r'\s\s+', engine='python')
		self.table = self.table.rename(lambda s: s.split('.')[-1], axis='columns')

	def _value(self, t):
		i = self.table.ElapsedSecs.searchsorted(t) - 1 # find the next event
		t0 = self.table.iloc[i].ElapsedSecs # and the last one
		t1 = self.table.iloc[i+1].ElapsedSecs
		ρ0 = self.table.iloc[i].AtmosDensity
		ρ1 = self.table.iloc[i+1].AtmosDensity
		return 1e-9*((t - t0)/(t1 - t0)*(ρ1 - ρ0) + ρ0) # and commit linear interpolation (negate because we want air velocity, not satellite velocity)


class MagneticTable(Table):
	""" An object to read GMAT reports and thus estimaet the magnetic field of the Earth. """
	def __init__(self, filename, B_earth=3.12e-5, R_earth=6370e+3, axis=[0,0,-1]):
		super().__init__()
		self.table = pd.read_csv(filename, sep=r'\s\s+', engine='python')
		self.table = self.table.rename(lambda s: s.split('.')[-1], axis='columns')
		self.B_0 = B_earth*np.array(axis)
		self.R_E = R_earth

	def _value(self, t):
		i = self.table.ElapsedSecs.searchsorted(t) - 1 # find the next event
		t0 = self.table.iloc[i].ElapsedSecs # and the last one
		t1 = self.table.iloc[i+1].ElapsedSecs
		r0 = np.array(self.table.iloc[i][['X', 'Y', 'Z']])
		r1 = np.array(self.table.iloc[i+1][['X', 'Y', 'Z']])
		r = ((t - t0)/(t1 - t0)*(r1 - r0) + r0)*1e+3 # and commit linear interpolation
		return -(3*r*np.dot(self.B_0, r)/np.linalg.norm(r)**2 - self.B_0)*(self.R_E/np.linalg.norm(r))**3 # and use a dipole approximation

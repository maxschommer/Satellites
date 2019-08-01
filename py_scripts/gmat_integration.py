from collections import OrderedDict
import numpy as np
import pandas as pd


""" COORDINATE CONVENTION:
	The coordinate directions are the same as in GMAT.
	The satellite is the origin (quasi-inertial). z+ is north.
	Initiall, x+ is roughly up and y+ is roughly east.
	The sun, for simplicity's sake, comes from y+.
"""


SUNWEND_LENGTH = 7


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
		i = self.table.ElapsedSecs.searchsorted(t) # get the next event
		if i-1 >= 0 and t - self.table.iloc[i-1].ElapsedSecs < SUNWEND_LENGTH/2: # if there's a previous event and we're close to it
			nearest_wend = self.table.iloc[i-1] # look at that one
		else: # otherwise
			nearest_wend = self.table.iloc[i] # the next event will work fine
		t_wend = (t - nearest_wend.ElapsedSecs)/(SUNWEND_LENGTH/2)
		if nearest_wend.type == "Sunset": # if the sun is going down
			if t_wend < -1:
				return self.solar_constant
			else:
				return (np.arccos(t_wend) - t_wend*np.sqrt(1 - t_wend**2))/np.pi*self.solar_constant
		else: # otherwise
			if t_wend < -1:
				return np.zeros(self.solar_constant.shape)
			else:
				return (np.arccos(-t_wend) + t_wend*np.sqrt(1 - t_wend**2))/np.pi*self.solar_constant
			return np.zeros(3) # it must be down


class VelocityTable(Table):
	""" An object to read GMAT reports and thus estimate the relative air speed. """
	def __init__(self, filename):
		super().__init__()
		self.table = pd.read_csv(filename, sep=r'\s\s+', engine='python')
		self.table = self.table.rename(lambda s: s.split('.')[-1], axis='columns')

	def _value(self, t):
		i = self.table.ElapsedSecs.searchsorted(t) - 1 # get the next event
		y0, y1 = self.table.iloc[i], self.table.iloc[i+1] # and the last one
		t0, t1 = y0.ElapsedSecs, y1.ElapsedSecs
		v0, v1 = -np.array([y0.VX, y0.VY, y0.VZ]), -np.array([y1.VX, y1.VY, y1.VZ])
		return ((t - t0)/(t1 - t0)*(v1 - v0) + v0)*1e+3 # and commit linear interpolation (negate because we want air velocity, not satellite velocity)


class AtmosphericTable(Table):
	""" An object to read GMAT reports and thus estimaet the air density. """
	def __init__(self, filename):
		super().__init__()
		self.table = pd.read_csv(filename, sep=r'\s\s+', engine='python')
		self.table = self.table.rename(lambda s: s.split('.')[-1], axis='columns')

	def _value(self, t):
		i = self.table.ElapsedSecs.searchsorted(t) - 1 # find the next event
		y0, y1 = self.table.iloc[i], self.table.iloc[i+1] # and the last one
		t0, t1 = y0.ElapsedSecs, y1.ElapsedSecs
		ρ0, ρ1 = y0.AtmosDensity, y1.AtmosDensity
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
		y0, y1 = self.table.iloc[i], self.table.iloc[i+1] # and the last one
		t0, t1 = y0.ElapsedSecs, y1.ElapsedSecs
		r0, r1 = np.array([y0.X, y0.Y, y0.Z]), np.array([y1.X, y1.Y, y1.Z])
		r = ((t - t0)/(t1 - t0)*(r1 - r0) + r0)*1e+3 # and commit linear interpolation
		return -(3*r*np.dot(self.B_0, r)/np.linalg.norm(r)**2 - self.B_0)*(self.R_E/np.linalg.norm(r))**3 # and use a dipole approximation

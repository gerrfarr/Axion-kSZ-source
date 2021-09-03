from __future__ import division
import numpy as np


class Noise_Models:

	@staticmethod
	def noise_function(ell, delta, beam_FWHM, cut_off=np.inf):
		noise=delta**2*np.exp(ell*(ell+1)*beam_FWHM**2/(8*np.log(2)))
		noise[np.where(ell*(ell+1)*noise>cut_off)[0]]=None
		return noise
	@staticmethod
	def s4_noise(ell):
		return Noise_Models.noise_function(ell, 1.0/60/180*np.pi, 3.0/60/180*np.pi)
	@staticmethod
	def planck_noise(ell):
		return Noise_Models.noise_function(ell, 7.1/60/180*np.pi, 37/60/180*np.pi)
	@staticmethod
	def spt_noise(ell):
		return Noise_Models.noise_function(ell, 1.1/60/180*np.pi, 2.5/60/180*np.pi)
	@staticmethod
	def act_noise(ell):
		return Noise_Models.noise_function(ell, 1.4/60/180*np.pi, 8.9/60/180*np.pi)

def cosmic_variance(ell, Cl, noise_model=None):
	if noise_model is None:
		return np.sqrt(2/(2*ell+1))*np.fabs(Cl)
	else:
		return np.sqrt(2/(2*ell+1)*(Cl+noise_model(ell))**2)


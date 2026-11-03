# compute trap averaged bulk viscosity of unitary Fermi gas
# given \mu/T, trap \bar\omega/T and the drive frequency \omega/T
# (all quantities E/h in units of Hz or lengths in units of the thermal length lambda_T)
#
# (c) LW Enss 2024
#

# Colin Dale modifications 2024-03-24
# Split traped and uniform gases into two different classes that store
# computed results and parameters. Added another trap gas class that can find
# the chemical potential for given ToTF, barnu, and EF

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, tplquad
from scipy.optimize import root_scalar
import pickle
import os
library_folder = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis'
import sys
if library_folder not in sys.path:
	sys.path.append(library_folder)
from library import plt_settings, tint_shade_color, tintshade, pi, mK, hbar
from contact_correlations.contact_tabulated import ContactInterpolation
from contact_correlations.baryrat import BarycentricRational
from numba import jit 

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, 'theory')

# print results
print_results = False

#
# properties of homogeneous (bulk) gas: EOS, bulk viscosity, and bulk thermodynamics
#

eosfit = {'nodes': np.array([5.45981500e+01, 3.35462628e-04, 4.48168907e+00, 1.28402542e+00]), 
		  'values': np.array([2.66603452e+01, 3.35574145e-04, 5.63725236e+00, 1.91237718e+00]), 
		  'weights': np.array([ 0.52786226, -0.10489219, -0.69208542,  0.48101646])}
eosrat = BarycentricRational(eosfit['nodes'],eosfit['values'],eosfit['weights'])

def eos_ufg(betamu):
	"""EOS of unitary gas: phase space density f_n(beta*mu) for both spin components (Zwierlein data)"""
	z = np.exp(betamu)
	f_n = 2*np.where(betamu<-8,z,eosrat(z)) # approximant is for a single spin component, so multiply by 2
	return f_n

def Theta(betamu):
	return (4*pi)/((3*pi**2)* eos_ufg(betamu))**(2/3)

sumrulefit = {'nodes': np.array([1.22144641e+01, 8.33717634e-03, 3.05244000e+00, 3.48110474e-01]), 
			  'values': np.array([1.46259386e+00, 1.24595501e-04, 5.81003381e-01, 5.49151117e-02]), 
			  'weights': np.array([ 0.33160786, -0.30343046, -0.66528124,  0.59612671])}
sumrat = BarycentricRational(sumrulefit['nodes'],sumrulefit['values'],sumrulefit['weights'])

def zeta(betamu,betaomega):
	"""dimensionless bulk viscosity of unitary gas: zeta-tilde(beta*mu,beta*omega)"""
	z = np.exp(betamu)
	sumruleT = np.where(betamu<-4.8,0.36*z**(5/3),sumrat(z)) # area under viscosity peak in units of T
	gammaT = 1.739-0.0892*z+0.00156*z**2 # width of viscosity peak in units of T
	return sumruleT*gammaT/(betaomega**2+gammaT**2)

def sumrule(betamu):
	"""sumrule in temperature units"""
	z = np.exp(betamu)
	sumruleT = np.where(betamu<-4.8,0.36*z**(5/3),sumrat(z)) # area under viscosity peak in units of T
	return sumruleT

def sumruleint(betamu):
	"""sumrule in temperature units from integral of zeta over betaomega"""
	integral, int_err = quad(lambda betaomega: zeta(betamu,betaomega),0,np.inf,epsrel=1e-4)
	sumruleT = 2/pi*integral
	return sumruleT

def phaseshift_Drude(betamu,betaomega):
	""" arctan(omega zeta/sum_rule) """
	z = np.exp(betamu)
	gammaT = 1.739-0.0892*z+0.00156*z**2 # width of viscosity peak in units of T
	return betaomega * gammaT/(betaomega**2+gammaT**2)

def phaseshift_zeta(nu, zeta, sumrule):
	""" arctan(omega zeta/sum_rule) """
	return np.arctan(nu * zeta/sumrule)

def thermo_bulk(betamu, T): # Had to add T as an argument here - CD
	"""compute thermodynamics of homogeneous gas (energies E=h*nu=hbar*omega given as nu in Hz)"""
	f_n = eos_ufg(betamu) # phase space density
	theta = 4*np.pi/(3*np.pi**2*f_n)**(2/3)
	f_p,f_p_err = quad(lambda v: eos_ufg(betamu-v),0,np.inf,epsrel=1e-4) # pressure by integrating density over mu
	Ebulk = (3/2)*f_p*T # internal energy density of UFG is 3/2 times pressure, for two spin components (in units of lambda^-3)
	return f_n,theta,f_p,Ebulk

def heating_bulk(T,betamu,betaomega):
	"""compute viscous heating rate E-dot in homogeneous system"""
	Zbulk = eos_ufg(betamu)**(1/3)*zeta(betamu,betaomega)
	Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Zbulk
	return Edot

def heating_from_zeta(T,betamu,betaomega,zeta):
	"""compute viscous heating rate E-dot in homogeneous system"""
	Zbulk = eos_ufg(betamu)**(1/3)*zeta
	Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Zbulk
	return Edot

def C_bulk(betamu):
	"""compute Contact Density for bulk gas"""
	Cbulk = ContactInterpolation(Theta(betamu))
	return Cbulk

@jit(nopython=True)
def heating_C(T, betaomega, C):
	"""compute heating rate at high frequency from contact density"""
	"""C is just the integrated contact density without prefactors"""
	pifactors = (3*pi**2)**(1/3)/(36*pi*(2*pi)**(3/2))
	Edot_C = 9*pi*(T*betaomega)**2/(betaomega)**(3/2)*pifactors*C
	return Edot_C

def zeta_C(betamu, betaomega):
	"""compute heating rate at high frequency from contact density"""
	pifactors = 3*pi**2/(36*pi*(2*pi)**(3/2))
	zetaC = pifactors*eos_ufg(betamu) * ContactInterpolation(Theta(betamu)) / (betaomega)**(3/2)
	return zetaC

def sumruleintC(betamu, Theta):
	"""sumrule in temperature units from integral of zeta over betaomega, but using
	zeta from contact for omega > T, i.e. betaomega > 1, if Theta=1, or 
	omega > EF, i.e. betaomega > 1/Theta if Theta=Theta"""
	integrand = lambda x: np.piecewise(x, [x<(1/Theta), x>=(1/Theta)], [zeta(betamu, x), zeta_C(betamu, x)])
	integral, int_err = quad(integrand,0,np.inf,epsrel=1e-4)
	sumruleT = 2/pi*integral
	return sumruleT

@jit(nopython=True)
def sumrule_zetaint(nus, zetas):
	"""sumrule in Hz"""
	sumrule = 2/pi*np.trapezoid(zetas, x=nus)
	return sumrule

class BulkViscUniform:
	def __init__(self, T, mubulk, nus):
		#
		# initialize params
		# *all* energies E=h*nu=hbar*omega are given as nu in Hz (without the 2pi factor,
		# even if some of the variables are called omega, like the driving frequency or the trap frequency!)
		#
		self.T = T
		self.mubulk = mubulk
		self.lambda_T = np.sqrt(hbar/(mK*T)) # thermal wavelength (unit of length, in meters)
		a0 = self.lambda_T # put actual amplitude of scattering length drive, in meters
		self.A = self.lambda_T/a0 # dimensionless amplitude of drive
		self.nus = nus
# 		self.nus = np.linspace(0.,nu_max,num+1) # choose frequency grid for drive frequency nu (in Hz, without the 2pi)
		
		#
		# compute bulk properties
		#
		
		betamubulk = self.mubulk/self.T
		betaomegas = self.nus/self.T # all frequencies, temperatures and energies are without the 2pi
		f_n,self.Theta,f_p,self.Ebulk = thermo_bulk(betamubulk, self.T)
		self.EF = self.T/self.Theta
		self.Edotbulks = self.A**2*np.array([heating_bulk(self.T,betamubulk,
										betaomega) for betaomega in betaomegas])
		
		self.C = C_bulk(betamubulk)
		self.sumrule = sumrule(betamubulk)*self.T
		self.EdotbulksC = self.A**2*np.array([heating_C(self.T,betaomega,
				   self.C*eos_ufg(betamubulk)**(4/3)) for betaomega in betaomegas])
		
		self.zetas = np.array([zeta(betamubulk,betaomega) for betaomega in betaomegas])
		self.zetasC = np.array([zeta_C(betamubulk,betaomega) for betaomega in betaomegas])  # added for comparison - CD
		
		self.sumruleint = sumruleint(betamubulk)*self.T
		self.sumruleintC = sumruleintC(betamubulk, 1)*self.T # using nu=T as the change freq for zeta calc
															# could also use nu=EF by replaced 1 with self.Theta
		self.tau = 1/tau_inv(betamubulk, self.T) # inverse scattering rate
		self.phaseshifts = np.array([phaseshift_Drude(betamubulk,
									betaomega) for betaomega in betaomegas])
		self.phaseshiftsC = np.array([phaseshift_zeta(betaomega*T, zetaC, 
			self.sumruleintC) for betaomega, zetaC in zip(betaomegas, self.zetasC)])
		self.phaseshiftsQcrit = np.arctan(2*np.pi*self.nus * self.tau / (1 + (2*np.pi*self.nus*self.tau)**2))
		self.phiLR = np.arctan(2*np.pi*self.nus * self.tau)

		self.betamubulk = betamubulk
		
		#
		# labels for plots
		# 
		
		self.title = r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=$'+'{:.1f}'.format(self.T/1e3)+'kHz'
		self.label_uni = r'uniform with local $\varepsilon_F=$'+'{:.1f}'.format(self.T/1e3/self.Theta)+'kHz'
		
		#
		# print results
		#
		
		if print_results == True:
			print("drive parameters: amplitude 1/a0=%g 1/m, lambda_T=%g m, dimensionless A=%f" % (1/a0,self.lambda_T,self.A))
			print("homogeneous system: phase space density %f, local T/TF=%f, pressure %f, energy density %f" % (f_n,self.Theta,f_p,self.Ebulk))

#
# trapped gas
#

eps = 1e-4

@jit(nopython=True)
def mutrap_est(ToTF):
	a = -50e3
	b = 21e3
	return a*ToTF + b

@jit(nopython=True)
def weight_harmonic(v,betabaromega):
	"""area of equipotential surface of potential value V/T=v=0...inf"""
	return 2/(betabaromega**3)*np.sqrt(v/np.pi)

def number_per_spin(betamu,betabaromega,weight_func):
	"""compute number of particles per spin state for trapped unitary gas:
	   N_sigma = int_0^infty dv w(v) f_n_sigma*lambda^3(mu-v)"""
	N_sigma,Nerr = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)/2,0,np.inf,epsrel=eps)
	return N_sigma

def psd_trap(betamu,betabaromega,weight_func):
	"""compute density averaged over the trap"""
	psd,psd_traperr = quad(lambda v: weight_func(v,betabaromega)*(eos_ufg(betamu-v)/2)**2,0,np.inf,epsrel=1e-4)
	return psd

def Epot_trap(betamu,betabaromega,weight_func):
	"""compute trapping potential energy (in units of T):
	   E_trap = int_0^infty dv w(v) f_n*lambda^3(mu-v) v"""
	Epot,Eerr = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)*v,0,np.inf,epsrel=eps)
	return Epot

def thermo_trap(T,betamu,betabaromega,weight_func):
	"""compute thermodynamics of trapped gas"""
	Ns = number_per_spin(betamu,betabaromega,weight_func)
	EF = T*betabaromega*(6*Ns)**(1/3) # in Hz, without 2pi
	Theta = T/EF
	Epot = T*Epot_trap(betamu,betabaromega,weight_func) # in Hz, without 2pi
	return Ns,EF,Theta,Epot

def heating_trap(T,betamu,betaomega,betabaromega,weight_func):
	"""compute viscous heating rate E-dot averaged over the trap"""
	Ztrap,Ztraperr = quad(lambda v: weight_func(v,
			   betabaromega)*eos_ufg(betamu-v)**(1/3)*zeta(betamu-v,betaomega),0,np.inf,epsrel=1e-4)
	# Ztrap_norm,Ztraperr_norm = quad(lambda v: weight(v,betabaromega)*eos_ufg(betamu-v)**(1/3),0,np.inf,epsrel=1e-4)
	Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Ztrap
	return Edot #, Ztrap/Ztrap_norm # modified to return trap avged zeta

def tau_inv(betamu, T):
	'''This is in units of angular frequency, 2pi Hz'''
	z = np.exp(betamu)
	tauinv = ((1.739 - 0.0892*z + 0.00156*z**2) * T) * (2*pi)
	return tauinv

def tau_inv_noz(T):
	'''This is in units of angular frequency, 2pi Hz'''
	tauinv = ((1.739) * T) * (2*pi)
	return tauinv

# unused; this gives weird results
def heating_trap_sumrule(T,betamu,betaomega,betabaromega,weight_func):
	"""compute viscous heating rate E-dot averaged over the trap normalized by scale sus"""
	 
	  	# Strap,Straperr = quad(lambda v: weight_func(v,
	# 			   betabaromega)*eos_ufg(betamu-v)**(1/3)*sumrule(betamu-v),0,np.inf,epsrel=1e-4)
	tau = 1/tau_inv(betamu, T) # inverse scattering rate
	drude_form = tau / (1 + (2*pi*betaomega*T*tau)**2)
	Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*drude_form
	return Edot 

# def phaseshift_qcrit(T, betaomega, betamu):
# 	"""phi = arctan(omegatau/(1+(omegatau**2)) Eq.(30) in May note"""
# 	tau = 1/tau_inv(betamu, T)
# 	phiqcrit = np.arctan((2*pi*betaomega)*T * tau / (1 + (2*pi*betaomega*T*tau)**2))
# 	return phiqcrit

def phaseshift_arg_trap(betamu,betaomega,betabaromega,weight_func):
	"""compute viscous heating rate E-dot averaged over the trap"""
	argtrap,argtraperr = quad(lambda v: weight_func(v,
		   betabaromega)*eos_ufg(betamu-v)**(1/3)*phaseshift_Drude(betamu-v,betaomega),0,np.inf,epsrel=1e-4)
	argtrap_norm,argtraperr_norm = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)**(1/3),0,
										np.inf,epsrel=eps)
	return argtrap/argtrap_norm #, Ztrap/Ztrap_norm # modified to return trap avged zeta

# def phiLR(T, betaomega, betamu):
# 	"""phi = arctan(omegatau) based on LR theory"""
# 	tau = 1/tau_inv(betamu, T)
# 	phiLR = np.arctan(2*pi*betaomega*T * tau)
# 	return phiLR

def find_betamu(T, ToTF, betabaromega, weight_func, guess=None):
	"""solves for betamu that matches T, EF and betabaromega of trap"""
	sol = root_scalar(lambda x: T/ToTF - T*betabaromega*(6*number_per_spin(x, 
				 betabaromega, weight_func))**(1/3), 
				#  bracket=[20e3/T, -300e3/T], 
				bracket = [-50, 50], #changed this because with the above it wouldn't converge for some reason
				 x0=guess)
	return sol.root, sol.iterations

def sumrule_trap(betamu, betabaromega, weight_func):
	"""sumrule in temperature units"""
	sumruleT, sumruleTerr = quad(lambda v: weight_func(v, betabaromega)*sumrule(betamu-v),
								 0,np.inf,epsrel=eps)
	return sumruleT

def C_trap(betamu, betabaromega,weight_func):
	"""compute Contact Density averaged over the trap"""
	Ctrap,Ctraperr = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)**(4/3)*ContactInterpolation(Theta(betamu-v)),0,np.inf,epsrel=eps)
	return Ctrap

class BulkViscTrap:
	def __init__(self, ToTF, EF, barnu, nus, a0=None, mutrap_guess=None):
		#
		# initialize params
		# *all* energies E=h*nu=hbar*omega are given as nu in Hz (without the 2pi factor,
		# even if some of the variables are called omega, like the driving frequency or the trap frequency!)
		#
		self.T = ToTF*EF
		self.ToTF = ToTF
		self.barnu = barnu
		self.lambda_T = np.sqrt(hbar/(mK*self.T)) # thermal wavelength (unit of length, in meters)
		self.nus = nus
		
		betaomegas = self.nus/self.T # all frequencies, temperatures and energies are without the 2pi
		self.betabaromega = barnu/self.T # all frequencies, temperatures and energies are without the 2pi
		
		self.weight_func = weight_harmonic
		
		#
		# find betamutrap that produces correct ToTF given EF, ToTF and betabaromega
		#
		if mutrap_guess:
			betamutrap_guess = mutrap_guess/self.T
		else:
			betamutrap_guess = mutrap_est(ToTF)
			
		self.betamutrap, no_iter = find_betamu(self.T, ToTF, self.betabaromega, self.weight_func, guess=betamutrap_guess)
		if print_results == True:
			print("Found betamutrap={:.2f} in {} iterations".format(self.betamutrap, no_iter))
			print("From initial guess {:.2f}".format(betamutrap_guess))
		
		if a0 is None:
			a0 = self.lambda_T # put actual amplitude of scattering length drive, in meters
			
		self.A = self.lambda_T/a0 # dimensionless amplitude of drive
				
		#
		# compute trap properties
		#
		self.tau = 1/tau_inv(self.betamutrap, self.T)
		self.tau_noz = 1/tau_inv_noz(self.T)
		self.Ns,self.EF,self.Theta,Epot = thermo_trap(self.T,self.betamutrap,self.betabaromega,self.weight_func)
		self.kF = np.sqrt(4*pi*mK*self.EF/hbar) # global k_F, i.e. peak k_F
		self.Etotal = 2*Epot # virial theorem valid at unitarity, 
		# but we have to decide if we want to normalize the trap heating rate by the total or by the internal energy
		self.EdotDrude = self.A**2*np.array([heating_trap(self.T,self.betamutrap,
						betaomega,self.betabaromega,self.weight_func) for betaomega in betaomegas])
		
		self.ns = psd_trap(self.betamutrap,self.betabaromega,self.weight_func)/self.Ns # /self.lambda_T**3

		C_trap_unscaled = C_trap(self.betamutrap, self.betabaromega, self.weight_func)
		self.Ctrap =  C_trap_unscaled/(self.kF*self.lambda_T)*(3*pi**2)**(1/3)/self.Ns/2
		self.EdotC = self.A**2*heating_C(self.T, betaomegas, C_trap_unscaled)
		self.EdotNormC = self.A**2*heating_C(self.T, betaomegas, (self.kF*self.lambda_T)/(3*pi**2)**(1/3))

		# these were divided by A**4 for some reason when I first saw this code. Why?
		self.zetaDrude = self.EdotDrude/self.A**2 * (self.lambda_T**2*self.kF**2)/(9*pi*nus**2*2*self.Ns)
		self.zetaC = self.EdotC/self.A**2 * (self.lambda_T**2*self.kF**2)/(9*pi*nus**2*2*self.Ns)
		
		self.sumruletrap = sumrule_trap(self.betamutrap, self.betabaromega, self.weight_func) * self.T/self.EF
		
		self.EdotDrudeS = self.EdotDrude / self.sumruletrap
		self.EdotDrudeSalt = self.A**2*np.array([heating_trap_sumrule(self.T,self.betamutrap, 
							betaomega, self.betabaromega, self.weight_func) for betaomega in betaomegas])
		self.phaseshiftsQcrit = np.arctan(2*np.pi*self.nus * self.tau / (1 + (2*np.pi*self.nus*self.tau)**2))
	
		self.phiLR = np.arctan(2*np.pi*self.nus * self.tau)
		self.phiLR_noz = np.arctan(2*np.pi*self.nus * self.tau_noz)
		
		self.EdotSphi = np.tan(self.phaseshiftsQcrit)*betaomegas*self.T/self.EF * \
								 9*pi/self.kF**2/self.lambda_T**2
		
		# find frequency to change from Drude to contact scaling
		nu_small = 0
		for nu in nus:
			if nu < (2*self.T):	
				nu_small += 1
			
		# lists for plotting
		self.Edot = np.append(self.EdotDrude[:nu_small],
						  self.EdotC[nu_small:])/(2*self.Ns * self.EF**2)
		self.zeta = np.append(self.zetaDrude[:nu_small], 
						self.zetaC[nu_small:])
		
		#
		# labels for plots
		# 
		self.title = r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=$'+'{:.1f}'.format(self.T/1e3)+'kHz'
		self.label_trap = r'trap $\bar\nu=$'+'{:.0f}'.format(self.barnu)+'Hz, $E_F='+'{:.1f}'.format(self.EF/1e3)+'$kHz, $T/T_F=$'+'{:.2f}'.format(self.Theta)
		
		#
		# print results
		#
		if print_results == True:
			print("drive parameters: amplitude 1/a0=%g 1/m, lambda_T=%g m, dimensionless A=%f" % (1/a0,self.lambda_T,self.A))
			print("trapped system: N_sigma=%f, EF=%f, global T/TF=%f, total energy %f" % (self.Ns,self.EF,self.Theta,self.Etotal))
			
	def calc_Edot(self, nu):
		betaomega = nu/self.T
		if betaomega < 2:
			Edot = self.A**2 * heating_trap(self.T, self.betamutrap,
							betaomega, self.betabaromega, self.weight_func)
		else:
			Edot = self.A**2*heating_C(self.T,betaomega, 
				  C_trap(self.betamutrap, self.betabaromega, self.weight_func))
		return Edot/(2*self.Ns * self.EF**2)
	
	def calc_zeta(self, nu):
		# check this, it might not be right
		zeta = self.calc_Edot(nu)/self.A**2 * (self.lambda_T**2*self.kF**2)/(9*pi*nu**2) * self.EF**2
		return zeta
			
def calc_contact(ToTF, EF, barnu, mutrap_guess=None):
	""" Calculates the harminic trap-averaged contact using ToTF, EF, barnu
		(geometric mean trap freq) and an optional guess mu. Returns the contact."""
	T = ToTF * EF
	betabaromega = barnu/T
	lambda_T = np.sqrt(hbar/(mK*T))
	kF = np.sqrt(4*pi*mK*EF/hbar) # global k_F, i.e. peak k_F
	
	# find mu
	if mutrap_guess:
		betamutrap_guess = mutrap_guess/T
	else:
		betamutrap_guess = mutrap_est(ToTF)
	betamutrap, no_iter = find_betamu(T, ToTF, betabaromega, weight_harmonic, 
								   guess=betamutrap_guess)
	if print_results == True:
		print("Found betamutrap={:.2f} in {} iterations".format(betamutrap, no_iter))
		print("From initial guess {:.2f}".format(betamutrap_guess))
	
	# calculate thermodynamics
	Ns, EF, Theta, Epot = thermo_trap(T,betamutrap,betabaromega,weight_harmonic)
	
	# calculate C
	Ctrap =  C_trap(betamutrap, betabaromega, weight_harmonic)/(kF*lambda_T)* \
				(3*pi**2)**(1/3)/Ns/2
				
	return Ctrap, Ns, EF, Theta
			

def contact_from_Edot(Edot, freq, T, kF):
	pifactors = 8*pi*(2*pi)**(2/3)/(3*pi**2)**(1/3)
	factors = (freq/T)**(3/2)/np.sqrt(hbar/(mK*T))/freq**2/(2*pi)
	C = pifactors*factors/kF*Edot
	return C

def phaseshift_from_zeta(nus, EF, zetas):
	sumrule = np.trapezoid(zetas, x=nus)
	phi = np.arctan(nus/EF*zetas/sumrule)
	return phi

###
### Plotting
###

### Load plt settings
plt.rcParams.update(plt_settings)

### Uniform Density System 
if __name__ == "__main__":
	titlebulk = 'Uniform Density Gas'
	
	Thetas = [0.25, 0.37, 0.58, 1.00]
	Ts = [4.8e3,  5.18e3, 11e3, 19e3] # Hz
	barnus = [306, 306, 306, 306] # mean trap freq in Hz
	mubulks = [7520,  4000, 1500, -11680] # uniform trap chemical potential
	colors = ['teal', 'blue', 'r', 'orange']
	theta_indices = [0, 1, 2]
	params_bulk = list(zip(Ts, mubulks))
	
	# load LW L-W data from PRL
	LW_nus = []
	LW_zetas = []
	LWFiles = ["zetaomega_T0.25.txt", "zetaomega_T0.58.txt"]
	for file in LWFiles:
		xx, yy = np.loadtxt(os.path.join(data_path,file), unpack=True, delimiter=' ')
		yy = yy/12 	# funny arbitrary scaling
		LW_nus.append(xx)
		LW_zetas.append(yy)
	
	# set plotting parameters
	plt.rcParams.update({"figure.figsize": [12,8]})
	xlabel=r'Frequency $\omega/E_F$'
# 	xlabel = r'$\omega/T$'
# 	font = {'size'   : 12}
# 	plt.rc('font', **font)
# 	legend_font_size = 10 # makes legend smaller, so plot is visible
	fig, axs = plt.subplots(2,2)
	
	# make frequency array
	nu_max = 200000
	num = int(nu_max/1e3)
	nus = np.linspace(nu_max/num,nu_max,num)
	
	# set axes attributes
	ax_zeta = axs[0,0]
	ylabel = r'Contact Correlation $z$'
	ax_zeta.set(xlabel=xlabel, ylabel=ylabel, yscale='log', xscale='log')
	
	ax_Edot = axs[1,0]
	ylabel = r'Heating Rate $\dot{E}/E$ (Hz)'
	ax_Edot.set(xlabel=xlabel, ylabel=ylabel)
	
	ax_phase = axs[0,1]
	ylabel = r'Phase Shift  (rad)'
	ylims = [-0.1,0.65]
	xlims=[0.01,10]
	ax_phase.set(xlabel=xlabel, ylabel=ylabel, xscale='log', ylim=ylims,xlim=xlims)
	
	ax_table = axs[1,1]
# 	ax_table.set(title=titlebulk)
	ax_table.axis('off')
	ax_table.axis('tight')
	
	label_Drude = 'Drude'
	label_LW = 'L-W'
	label_C = r'$\tilde{\zeta} \propto \mathcal{C}/\omega^{3/2}$'
	label_Qcrit = 'Qcrit'
	labels_ToTF = ["$T/T_F=0.25$", "$T/T_F=0.58$"]

	# loop over ToTF
	BVUs = []
	table_rows = []
	for i in theta_indices:
# 		print(i)
		color = colors[i]
		BVU = BulkViscUniform(*params_bulk[i], nus)
		label = "$T/T_F=${:.2f}".format(Thetas[i])
		
		# find nus for nu/EF > 1 to use for high_freq
		nu_small = 0
		for nu in nus:
			if nu < (BVU.T):
				nu_small += 1
# 		nu_small = 0
		
		ax_Edot.plot(BVU.nus/BVU.EF, BVU.Edotbulks/BVU.Ebulk, ':', label=label, 
		  color=color)
		ax_Edot.plot(BVU.nus[nu_small:]/BVU.EF, 
			   BVU.EdotbulksC[nu_small:]/BVU.Ebulk, '--', color=color)
		
		ax_zeta.plot(BVU.nus/BVU.EF, BVU.zetas, ':', 
			   label=label_Drude, color=color)
		ax_zeta.plot(BVU.nus[nu_small:]/BVU.EF, BVU.zetasC[nu_small:], '--', 
			   label=label_C, color=color)
		
		ax_phase.plot(BVU.nus/BVU.T, BVU.phaseshifts, ':', 
				label=label_Drude, color=color)
		ax_phase.plot(BVU.nus[nu_small:]/BVU.T, BVU.phaseshiftsC[nu_small:], '--', 
				label=label_C, color=color)
		ax_phase.plot(BVU.nus/BVU.T, BVU.phaseshiftsQcrit, '-.', label=label_Qcrit,color=color)
		
		#these plot versions were for omega/T on the x-axis
# 		ax_phase.plot(BVU.nus/BVU.T, BVU.phaseshifts, ':', 
# 				label=label_Drude, color=color)
# 		ax_phase.plot(BVU.nus[nu_small:]/BVU.T, BVU.phaseshiftsC[nu_small:], '--', 
# 				label=label_C, color=color)
# 		ax_phase.plot(BVU.nus/BVU.T, BVU.phaseshiftsQcrit, '-.', label=label_Qcrit,color=color)
# 		if (i == 0) or (i == 2):
# 			
# 			LW_i = 0 if i==0 else 1
# 			LW_Edotbulks = BVU.A**2*np.array([heating_from_zeta(BVU.T, BVU.betamubulk, 
# 				betaomega, zeta) for betaomega, zeta in zip(LW_nus[LW_i]/BVU.Theta, LW_zetas[LW_i])])
# 			LW_sumrule = sumrule_zetaint(LW_nus[LW_i], LW_zetas[LW_i])
# 			LW_phaseshifts = np.array([phaseshift_zeta(LW_nu, LW_zeta, LW_sumrule) \
# 							  for LW_nu, LW_zeta in zip(LW_nus[LW_i], LW_zetas[LW_i])])
# 			ax_Edot.plot(LW_nus[LW_i], LW_Edotbulks/BVU.Ebulk, '-', color=color)
# 			ax_zeta.plot(LW_nus[LW_i], LW_zetas[LW_i], '-', label=label_LW, color=color)
# 			ax_phase.plot(LW_nus[LW_i]*BVU.EF/BVU.T, LW_phaseshifts, '-', label=label_LW, color=color)
# 		
# 		
# 		if i == 1:
# 			if show_data == True:
# 				EF = 13
# 				freqs = np.array([5,8,10])/EF # 8 kHz is more like 0.31 ToTF
# 				phases = np.array([0.33, 0.37, 0.57])
# 				phases_err = np.array([0.17, 0.12, 0.14])
# 				label = r"Measurements @ $T/T_F={:.2f}$".format(Thetas[i])
# 				light_color = tint_shade_color(color, amount=1+tintshade)
# 				dark_color = tint_shade_color(color, amount=1-tintshade)
# 				ax_phase.errorbar(freqs, phases, yerr=phases_err, capsize=0, 
# 					  fmt="^", color=dark_color, markerfacecolor=light_color, 
# 					    markeredgecolor=dark_color, markeredgewidth=2, label=label)
# 		
# 		if i == 2:
# 			if show_data == True:
# 				EF = 19
# 				freqs = np.array([2, 5])/EF
# # 				freqs = np.array([2000,5000])/BVU.T # for plotting against omega/T
# 				phases = np.array([-0.06, 0.30])
# 				phases_err = np.array([0.09, 0.09])
# 				label = r"Measurements @ $T/T_F={:.2f}$".format(Thetas[i])
# 				light_color = tint_shade_color(color, amount=1+tintshade)
# 				dark_color = tint_shade_color(color, amount=1-tintshade)
# 				ax_phase.errorbar(freqs, phases, yerr=phases_err, capsize=0, 
# 					  fmt="^", color=dark_color, markerfacecolor=light_color, 
# 					    markeredgecolor=dark_color, markeredgewidth=2, label=label)
# 				
# 				EF = 21
# 				freqs = np.array([10])/EF
# # 				freqs = np.array([2000,5000])/BVU.T # for plotting against omega/T
# # 				phases = np.array([0.20])
# # 				phases_err = np.array([0.20])
# 				phases = np.array([0.25])
# 				phases_err = np.array([0.13])
# 				label = r"Measurements @ $T/T_F={:.2f}$".format(Thetas[i])
# 				light_color = tint_shade_color(color, amount=1+tintshade)
# 				dark_color = tint_shade_color(color, amount=1-tintshade)
# 				ax_phase.errorbar(freqs, phases, yerr=phases_err, capsize=0, 
# 					  fmt="^", color=dark_color, markerfacecolor=light_color, 
# 					    markeredgecolor=dark_color, markeredgewidth=2, label=label)
					
		ax_zeta.legend()
# 		ax_phase.legend()
		
		
		table_row = ["{:.2f}".format(Thetas[i]),
		  "{:.1f} kHz".format(BVU.EF/1e3), 
		  "{:.1f} Hz".format(barnus[i]),
		  "{:.3f} ".format(BVU.sumrule/BVU.EF),
		  "{:.3f} ".format(BVU.sumruleintC/BVU.EF),
		  "{:.2f} ".format(BVU.C)]
		
		BVUs.append(BVU)
		table_rows.append(table_row)
		
	# add temperature legend
	ax_Edot.legend()
	
	# generate table
	quantities = ["Theta", "$E_F$", "$\\bar\omega$", "Drude $s$", 
			   "Drude-$\mathcal{C}$ $s$", "$\mathcal{C}$"]
	table = list(zip(quantities, *table_rows))
	
	the_table = ax_table.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(10)
	the_table.scale(1,1.5)
	fig.savefig(os.path.join(proj_path,"figures/uniform_density_plots.pdf"))
	plt.tight_layout()
	plt.show()

### Harmonic Trap
	title = 'Harmonically Trapped Gas'
	
	trapfreqs = np.array([173, 406, 434])
	trapfreqs1p5 = np.array([173, 406, 434])*np.sqrt(1.5)
	barnu = np.prod(trapfreqs)**(1/3)
	barnu1p5 = np.prod(trapfreqs1p5)**(1/3)
	
	Thetas = [0.36, 0.442, 0.447, 0.529] # 0.58 -> 11e3
	EFs = [15e3, 17e3, 15e3, 20.9e3]
	barnus = [barnu, barnu, barnu1p5, barnu1p5] # mean trap freq in Hz
	mutraps = [9825, 3000, -3800, -3800] # harmonic trap chemical potential
	colors = ['teal', 'b', 'r', 'k']
	theta_indices = [0,1,2,3]
	params_trap = list(zip(Thetas, EFs, barnus))
	
	# set plotting parameters
	plt.rcParams.update({"figure.figsize": [12,8]})
	xlabel=r'Frequency $\omega/E_F$'
	font = {'size'   : 12}
	plt.rc('font', **font)
	legend_font_size = 10 # makes legend smaller, so plot is visible
	fig, axs = plt.subplots(3,2)
	
	# make frequency array
	nu_max = 120000
	num = int(nu_max/1e3)
	nus = np.linspace(nu_max/num,nu_max,num)
	
	# set axes attributes
	ax_zeta = axs[0,0]
	ylabel = r'Contact Correlation $z$'
	ax_zeta.set(xlabel=xlabel, ylabel=ylabel, yscale='log', xscale='log')
	
	ax_Edot = axs[1,0]
	ylabel = r'Heating Rate $\dot{E}/E$ (Hz)'
	ax_Edot.set(xlabel=xlabel, ylabel=ylabel)
	
	ax_phase = axs[0,1]
	ylabel = r'Phase Shift  (rad)'
	ylims = [-0.1,0.6]
	xlims=[0.001, 10]
	ax_phase.set(xlabel=xlabel, ylabel=ylabel, xscale='log', ylim=ylims, xlim=xlims)
	
	ax_table = axs[1,1]
	ax_table.set(title=title)
	ax_table.axis('off')
	ax_table.axis('tight')
	
	ax_EdotS = axs[2,0]
	ylabel = r'Heating Rate $\dot{E}/E/S$'
# 	ylims = [-0.01,0.03]
	xlims=[0,10]
	ax_EdotS.set(xlabel=xlabel, ylabel=ylabel,xlim=xlims)
	
	ax_EdotC = axs[2,1]
	ylabel = r'Heating Rate $\dot{E}/E/C$'
	ax_EdotC.set(xlabel=xlabel, ylabel=ylabel)
	
	label_Drude = 'Drude'
	label_LW = 'L-W'
	label_C = 'Contact'
	label_Qcrit = 'Qcrit'
	labels_ToTF = ["$T/T_F=0.25$", "$T/T_F=0.58$"]

	# loop over ToTF
	BVTs = []
	table_rows = []
	for i in theta_indices:
		color = colors[i]
		BVT = BulkViscTrap(*params_trap[i], nus, a0=-2.70e-6)
		label = "$T/T_F=${:.2f}".format(Thetas[i])
		
		# find nus for nu/EF > 1 to use for high_freq
		nu_small = 0
		for nu in nus:
			if nu < (BVT.T):
				nu_small += 1
# 		nu_small = 0
		
		ax_Edot.plot(BVT.nus/BVT.EF, BVT.EdotDrude/BVT.Etotal, ':', label=label, 
		  color=color)
		ax_Edot.plot(BVT.nus[nu_small:]/BVT.EF, BVT.EdotC[nu_small:]/BVT.Etotal, '--', color=color)
		
		# plot "normalized" heating rates
		ax_EdotS.plot(BVT.nus/BVT.EF, BVT.EdotDrudeS/BVT.Etotal, ':', color=color,label=label)
				
		BVT.zetas = np.concatenate([BVT.zetaDrude[:nu_small], BVT.zetaC[nu_small:]])
		sumrulezetaint = sumrule_zetaint(BVT.nus, BVT.zetas)
		ax_EdotS.plot(BVT.nus/BVT.EF, BVT.EdotDrude/sumrulezetaint/BVT.Etotal, '--',color=color, label='EdotDrude/sumrulezetaint')
		# ax_EdotS.plot(BVT.nus/BVT.EF, BVT.EdotDrudeS/BVT.Etotal, '.',color=color,label='EdotDrudeS')
		# ax_EdotS.plot(BVT.nus/BVT.EF, BVT.EdotDrudeSalt/BVT.Etotal, '.-',color=color,label='EdotDrudeSalt')
		# ax_EdotS.plot(BVT.nus/BVT.EF, BVT.EdotSphi/BVT.Etotal, '^',color=color,label='EdotSphi')
		ax_EdotC.plot(BVT.nus[nu_small:]/BVT.EF, BVT.EdotNormC[nu_small:]/BVT.Etotal, '--',color=color,label=label)
		
		ax_zeta.plot(BVT.nus/BVT.EF, BVT.zetaDrude, ':', label=label_Drude, color=color)
		ax_zeta.plot(BVT.nus[nu_small:]/BVT.EF, BVT.zetaC[nu_small:], '--', 
			   label=label_C, color=color)
		
		if i == 0:
			ax_zeta.legend(prop={'size':legend_font_size})
			ax_EdotS.legend()
		
		table_row = ["{:.2f}".format(Thetas[i]),
		  "{:.1f} kHz".format(BVT.EF/1e3), 
		  "{:.1f} Hz".format(barnus[i]),
		  "{:.3f} ".format(BVT.sumruletrap/BVT.EF),
		  "{:.2f} ".format(BVT.Ctrap)]
		  
		
		BVTs.append(BVT)
		table_rows.append(table_row)
	
	ax_Edot.legend(prop={'size':legend_font_size})
	ax_EdotS.legend(prop={'size':legend_font_size})
	ax_EdotC.legend(prop={'size':legend_font_size})
	
	# generate table
	quantities = ["Theta", "$E_F$", "$\\bar\omega$", "Drude $s$", 
			   "$\mathcal{C}$"]
	table = list(zip(quantities, *table_rows))
	
	the_table = ax_table.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(legend_font_size)
	the_table.scale(1,1.5)
	
	plt.tight_layout()
	plt.show()

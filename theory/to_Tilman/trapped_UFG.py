# compute trap averaged bulk viscosity of unitary Fermi gas
# given \mu/T, trap \bar\omega/T and the drive frequency \omega/T
# (all quantities E/h in units of Hz or lengths in units of the thermal length lambda_T)
#
# (c) LW Enss 2024
#

# Modifications made 2025-04-07 by Colin J. Dale
# Split trpaped and uniform gases into two different classes that store
# computed results and parameters. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.constants import pi, hbar
from baryrat import BarycentricRational

# constants
uatom = 1.660538921E-27
a0 = 5.2917721092E-11
uB = 9.27400915E-24
gS = 2.0023193043622
gJ = gS
mK = 39.96399848 * uatom
gI = 0.000176490 # total nuclear g-factor

# load tabulated contact density calculations
ToTFs, _, _, _, _, Cdensity = np.loadtxt('luttward-thermodyn.txt', 
										 skiprows=5, unpack=True)

# convert kF^3 to peak trap density of Harmonic trap
# trapped contact density c/(k_F n) = C/k_F^4 * (3 pi^2)
Cs = Cdensity * 3*np.pi**2

# interpolate data
ContactInterpolation = lambda x: np.interp(x, ToTFs, Cs)

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

def C_bulk(betamu):
    """compute Contact Density for bulk gas"""
    Cbulk = ContactInterpolation(Theta(betamu))
    return Cbulk


class UFG_Uniform:
	""" Uniformly trapped unitary fermi gas Class. On initialization, computes
		heating rate for each nu in the nus input list.
		inputs:
			T - float (Hz)
			mubulk - float (Hz)
			nus - list (Hz)
	"""
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
		
		self.betamubulk = betamubulk
		
		#
		# labels for plots
		# 
		
		self.title = r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=$'+'{:.1f}'.format(self.T/1e3)+'kHz'
		self.label_uni = r'uniform with local $\varepsilon_F=$'+'{:.1f}'.format(self.T/1e3/self.Theta)+'kHz'

#
# trapped gas
#

eps = 1e-4

def mutrap_est(ToTF):
	"""a hacky guess function for mutrap given ToTF for our typical trap params"""
	a = -50e3
	b = 21e3
	return a*ToTF + b

def weight_harmonic(v,betabaromega):
    """area of equipotential surface of potential value V/T=v=0...inf"""
    return 2/(betabaromega**3)*np.sqrt(v/np.pi)

def number_per_spin(betamu,betabaromega):
    """compute number of particles per spin state for trapped unitary gas:
       N_sigma = int_0^infty dv w(v) f_n_sigma*lambda^3(mu-v)"""
    N_sigma,Nerr = quad(lambda v: weight_harmonic(v,betabaromega)* \
						eos_ufg(betamu-v)/2,0,np.inf,epsrel=eps)
    return N_sigma

def Epot_trap(betamu,betabaromega):
    """compute trapping potential energy (in units of T):
       E_trap = int_0^infty dv w(v) f_n*lambda^3(mu-v) v"""
    Epot,Eerr = quad(lambda v: weight_harmonic(v,betabaromega)* \
					 eos_ufg(betamu-v)*v,0,np.inf,epsrel=eps)
    return Epot

def thermo_trap(T,betamu,betabaromega):
    """compute thermodynamics of trapped gas"""
    Ns = number_per_spin(betamu,betabaromega)
    EF = T*betabaromega*(6*Ns)**(1/3) # in Hz, without 2pi
    Theta = T/EF
    Epot = T*Epot_trap(betamu,betabaromega) # in Hz, without 2pi
    return Ns,EF,Theta,Epot

def heating_trap(T,betamu,betaomega,betabaromega):
	"""compute viscous heating rate E-dot averaged over the trap"""
	Ztrap,Ztraperr = quad(lambda v: weight_harmonic(v,
			   betabaromega)*eos_ufg(betamu-v)**(1/3)*zeta(betamu-v,betaomega),
					   0,np.inf,epsrel=1e-4)
	Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Ztrap
	return Edot 

def find_betamu(T, ToTF, betabaromega, guess=None):
	"""solves for betamu that matches T, EF and betabaromega of trap"""
	sol = root_scalar(lambda x: T/ToTF - T*betabaromega*(6*number_per_spin(x, 
				 betabaromega))**(1/3), bracket=[20e3/T, -300e3/T], x0=guess)
	return sol.root, sol.iterations

def C_trap(betamu, betabaromega):
    """compute Contact Density averaged over the trap"""
    Ctrap,Ctraperr = quad(lambda v: weight_harmonic(v,betabaromega)* \
			  eos_ufg(betamu-v)**(4/3)*ContactInterpolation(Theta(betamu-v)),
				  0,np.inf,epsrel=eps)
    return Ctrap


class UFG_Trap:
	""" Class for harmonically trapped unitary fermi gas. On initialization, 
		computes heating rate and contact for each nu in the nus input list.
		inputs:
			ToTF - float (dimless)
			EF - flaot (Hz)
			barnu - float (Hz)
			nus - list (Hz)
		optional inputs:
			a0 - scattering length oscillation amplitude
			mutrap_guess - starting value for chemical potential solver
	"""
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
		
		#
		# find betamutrap that produces correct ToTF given EF, ToTF and betabaromega
		#
		if mutrap_guess:
			betamutrap_guess = mutrap_guess/self.T
		else:
			betamutrap_guess = mutrap_est(ToTF)
			
		self.betamutrap, no_iter = find_betamu(self.T, ToTF, self.betabaromega,
											   guess=betamutrap_guess)
		
		if a0 is None:
			a0 = self.lambda_T # put actual amplitude of scattering length drive, in meters
			
		self.A = self.lambda_T/a0 # dimensionless amplitude of drive
				
		#
		# compute trap properties
		#
		self.Ns,self.EF,self.Theta,Epot = thermo_trap(self.T,self.betamutrap,self.betabaromega)
		self.kF = np.sqrt(4*pi*mK*self.EF/hbar) # global k_F, i.e. peak k_F
		self.Etotal = 2*Epot # virial theorem valid at unitarity, 
		
		# but we have to decide if we want to normalize the trap heating rate by the total or by the internal energy
		self.EdotDrude = self.A**2*np.array([heating_trap(self.T,self.betamutrap,
						betaomega,self.betabaromega) for betaomega in betaomegas])
	
		self.Ctrap =  C_trap(self.betamutrap, self.betabaromega)/(self.kF*self.lambda_T)*(3*pi**2)**(1/3)/self.Ns/2
		
		#
		# labels for plots
		# 
		self.title = r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=$'+'{:.1f}'.format(self.T/1e3)+'kHz'
		self.label_trap = r'trap $\bar\nu=$'+'{:.0f}'.format(self.barnu)+'Hz, $E_F='+'{:.1f}'.format(self.EF/1e3)+'$kHz, $T/T_F=$'+'{:.2f}'.format(self.Theta)

			
def calc_contact(ToTF, EF, barnu, mutrap_guess=None):
	""" calculates the harmonic trap-averaged contact using ToTF, EF, barnu
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
	betamutrap, no_iter = find_betamu(T, ToTF, betabaromega, 
								   guess=betamutrap_guess)
	
	# calculate thermodynamics
	Ns, EF, Theta, Epot = thermo_trap(T,betamutrap,betabaromega)
	
	# calculate C
	Ctrap =  C_trap(betamutrap, betabaromega)/(kF*lambda_T)* \
				(3*pi**2)**(1/3)/Ns/2
				
	return Ctrap, Ns, EF, Theta
			
def contact_from_Edot(Edot, freq, T, kF):
	""" calculates contact from heating rate """
	pifactors = 8*pi*(2*pi)**(2/3)/(3*pi**2)**(1/3)
	factors = (freq/T)**(3/2)/np.sqrt(hbar/(mK*T))/freq**2/(2*pi)
	C = pifactors*factors/kF*Edot
	return C

###
### Plotting
###

# plotting settings
frame_size = 1.5
plt_settings = {"axes.linewidth": frame_size,
				"axes.edgecolor":'black',
				"scatter.edgecolors":'black',
				"lines.linewidth":2,
				"font.size": 12,
				"legend.fontsize": 10,
				"legend.framealpha": 1.0,
				"xtick.major.width": frame_size,
				"xtick.minor.width": frame_size*0.75,
				"xtick.direction":'in',
				"xtick.major.size": 3.5*frame_size,
				"xtick.minor.size": 2.0*frame_size,
				"ytick.major.width": frame_size,
				"ytick.minor.width": frame_size*0.75,
				"ytick.major.size": 3.5*frame_size,
				"ytick.minor.size": 2.0*frame_size,
				"ytick.direction":'in',
				"lines.linestyle":'',
				"lines.markeredgewidth": 2,
				"figure.dpi": 300}

plt.rcParams.update(plt_settings)


# typical chip lab fermi gas parameters
EF = 16e3  # Hz
barnu = 307  # Hz
thetas = np.linspace(0.2, 1.0, 20)

xlabel = r"Temperature, $T$ [$T_F$]"

# compute contacts in DC, i.e. nu = 0
plt.figure(figsize=(6,6))
plt.ylabel(r"Contact per atom, $C/N$ [$k_F$]")
plt.xlabel(xlabel)

contacts = []
for theta in thetas:
	contacts.append(UFG_Trap(theta, EF, barnu, [0]).Ctrap)
	
contacts = np.array(contacts)

plt.plot(thetas, contacts, linestyle='-', marker='')
plt.show()


# plot interpolation of contact density calculations
plt.figure(figsize=(6,6))
plt.xlabel(xlabel)
plt.ylabel(r"Contact density, $\mathcal{C}/(n k_F)$")
plt.xlim(0, 1.0)
plt.ylim(2.3, 3.05)

plt.plot(ToTFs, Cs, 'o')
plt.plot(ToTFs, ContactInterpolation(ToTFs), '--')
plt.show()
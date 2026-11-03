# Computes trap-averaged bulk viscosity of unitary Fermi gas
# given \mu/T, trap \bar\omega/T and the drive frequency \omega/T
# (all quantities E/h in units of Hz or lengths in units of the thermal length lambda_T)
#
# (c) LW Enss 2024
#
# Modified by the chip lab.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from datetime import datetime
from constants import pi, mK, hbar
import sys
analysis_folder = 'F:\\Analysis Scripts\\Fast-Modulation-Contact-Correlation-Project\\analysis'
if analysis_folder not in sys.path:
	sys.path.append(analysis_folder)
from baryrat import BarycentricRational
from luttinger_ward_calculations import \
    contact_density, scale_susceptibility


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
sumrat = BarycentricRational(sumrulefit['nodes'], sumrulefit['values'], sumrulefit['weights'])


def betagamma(z):
    """Width of viscosity peak in units of T."""
    return 1.739-0.0892*z+0.00156*z**2


def zeta(betamu, betaomega):
    """dimensionless bulk viscosity of unitary gas: zeta-tilde(beta*mu,beta*omega)"""
    z = np.exp(betamu)
    betasumrule = np.where(betamu<-4.8,0.36*z**(5/3), sumrat(z))  # Area under viscosity peak in units of T.
    bgamma = betagamma(z)
    return betasumrule*bgamma/(betaomega**2 + bgamma**2)


def betasumrule(betamu):
    """Sumrule in dimensionless units (via temperature)."""
    z = np.exp(betamu)
    sumrule = np.where(betamu<-4.8,0.36*z**(5/3), sumrat(z))  # Area under viscosity peak in units of T.
    return sumrule


def sumruleint(betamu):
    """Sumrule in temperature units from integral of zeta over betaomega."""
    integral, int_err = quad(lambda betaomega: zeta(betamu,betaomega), 0, np.inf, epsrel=1e-4)
    sumruleT = 2/pi*integral
    return sumruleT


def phaseshift_Drude(betamu, betaomega):
    """ arctan(omega zeta/sum_rule) """
    z = np.exp(betamu)
    gammaT = 1.739 - 0.0892*z + 0.00156*z**2  # Width of viscosity peak in units of T.
    return betaomega * gammaT/(betaomega**2 + gammaT**2)


def phaseshift_zeta(nu, zeta, sumrule):
    """ arctan(omega zeta/sum_rule) """
    return np.arctan(nu * zeta/sumrule)


def thermo_bulk(betamu, T): # Had to add T as an argument here - CD
    """Compute thermodynamics of homogeneous gas (energies E=h*nu=hbar*omega given as nu in Hz)."""
    f_n = eos_ufg(betamu) # phase space density
    theta = 4*np.pi/(3*np.pi**2*f_n)**(2/3)
    f_p, f_p_err = quad(lambda v: eos_ufg(betamu-v), 0, np.inf, epsrel=1e-4) # pressure by integrating density over mu
    Ebulk = (3/2)*f_p*T # internal energy density of UFG is 3/2 times pressure, for two spin components (in units of lambda^-3)
    return f_n,theta,f_p,Ebulk


def heating_bulk(T, betamu, betaomega):
    """Compute viscous heating rate E-dot in homogeneous system."""
    Zbulk = eos_ufg(betamu)**(1/3)*zeta(betamu,betaomega)
    Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Zbulk
    return Edot


def heating_from_zeta(T, betamu, betaomega, zeta):
    """Compute viscous heating rate E-dot in homogeneous system."""
    Zbulk = eos_ufg(betamu)**(1/3)*zeta
    Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Zbulk
    return Edot


def heating_C(T, betaomega, C):
    """compute heating rate at high frequency from contact density.
    Contact is just the integrated contact density without prefactors."""
    pifactors = (3*pi**2)**(1/3)/(36*pi*(2*pi)**(3/2))
    Edot_C = 9*pi*(T*betaomega)**2/(betaomega)**(3/2) * pifactors * C
    return Edot_C


def zeta_C(betamu, betaomega):
    """Compute heating rate at high frequency from contact density."""
    pifactors = 3*pi**2/(36*pi*(2*pi)**(3/2))
    zetaC = pifactors * eos_ufg(betamu) * contact_density(Theta(betamu)) / (betaomega)**(3/2)
    return zetaC


def sumruleintC(betamu, Theta):
    """Sumrule in temperature units from integral of zeta over betaomega, but using
    zeta from contact for omega > T, i.e. betaomega > 1, if Theta=1, or 
    omega > EF, i.e. betaomega > 1/Theta if Theta=Theta."""
    integrand = lambda x: np.piecewise(x, [x<(1/Theta), x>=(1/Theta)], [zeta(betamu, x), zeta_C(betamu, x)])
    integral, int_err = quad(integrand,0,np.inf,epsrel=1e-4)
    sumruleT = 2/pi*integral
    return sumruleT


def sumrule_zetaint(nus, zetas):
    """Sumrule in Hz."""
    sumrule = 2/pi*np.trapezoid(zetas, x=nus)
    return sumrule


def gamma(betamu, T):
    '''This is in units of frequency, Hz.'''
    z = np.exp(betamu)
    g = ((1.739 - 0.0892*z + 0.00156*z**2) * T)
    return g


class BulkViscUniform:
    """Object to compute quantities in a uniform density unitary Fermi gas. 
    All energies, temperatures and frequencies are in Hz (no 2pi)."""
    
    def __init__(self, T, mu, nus):
        self.T = T
        self.mu = mu
        self.lambda_T = np.sqrt(hbar/(mK*T))  # Thermal wavelength (unit of length, in meters)
        a0 = self.lambda_T  # Put actual amplitude of scattering length drive, in meters
        self.A = self.lambda_T/a0  # Dimensionless amplitude of drive
        self.nus = nus
        self.birthday = datetime.now()
        
        betamu = self.mu/self.T
        self.betaomegas = self.nus/self.T
        
        #
        # Compute bulk properties
        #
        
        f_n, self.Theta, f_p, self.E = thermo_bulk(betamu, self.T)
        self.EF = self.T/self.Theta
        self.Edot = self.A**2*np.array([heating_bulk(self.T,betamu,
                                        betaomega) for betaomega in self.betaomegas])
        
        self.C = contact_density(Theta(betamu))
        self.tau = 1/gamma(betamu, self.T)  # Scattering rate.

        self.sumrule = betasumrule(betamu) * self.T
        self.EdotbulksC = self.A**2 * np.array([heating_C(self.T, betaomega,
                    self.C*eos_ufg(betamu)**(4/3)) for betaomega in self.betaomegas])

        self.zetas = np.array([zeta(betamu,betaomega) for betaomega in self.betaomegas])
        self.zetasC = np.array([zeta_C(betamu,betaomega) for betaomega in self.betaomegas])  # added for comparison - CD

        self.sumruleint = sumruleint(betamu)*self.T
        self.sumruleintC = sumruleintC(betamu, 1)*self.T # using nu=T as the change freq for zeta calc
                                                            # could also use nu=EF by replaced 1 with self.Theta

        self.phaseshifts = np.array([phaseshift_Drude(betamu,
                                    betaomega) for betaomega in self.betaomegas])
        
        self.phaseshiftsC = np.array([phaseshift_zeta(betaomega*T, zetaC, 
            self.sumruleintC) for betaomega, zetaC in zip(self.betaomegas, self.zetasC)])
        
        self.phaseshiftsQcrit = np.arctan(self.nus * self.tau / (1 + (self.nus*self.tau)**2))

        self.phiLR = np.arctan(self.nus * self.tau)

        self.betamu = betamu

    def __repr__(self):
        return f"TUG({self.ToTF}, {self.EF}, {self.barnu}) created at {self.birthday.strftime('%Y-%m-%d %H:%M:%S')}"
    


#
# trapped gas
#

eps = 1e-4  # Small quantity.


def weight_harmonic(v, betabaromega):
    """Area of equipotential surface of potential value V/T=v=0...inf."""
    return 2/(betabaromega**3)*np.sqrt(v/np.pi)


def number_per_spin(betamu, betabaromega, weight_func=weight_harmonic, v_max=np.inf):
    """Compute number of particles per spin state for trapped unitary gas:
       N_sigma = int_0^infty dv w(v) f_n_sigma*lambda^3(mu-v)."""
    N_sigma, Nerr = quad(lambda v: weight_func(v,betabaromega) * eos_ufg(betamu-v)/2, 0, v_max, epsrel=eps)
    return N_sigma


def psd_trap(betamu, betabaromega, weight_func=weight_harmonic):
    """Compute density averaged over the trap."""
    psd, psd_traperr = quad(lambda v: weight_func(v,betabaromega)*\
                        (eos_ufg(betamu-v)/2)**2, 0, np.inf, epsrel=1e-4)
    return psd


def Epot_trap(betamu, betabaromega, weight_func=weight_harmonic):
    """Compute trapping potential energy (in units of T):
       E_trap = int_0^infty dv w(v) f_n*lambda^3(mu-v) v."""
    Epot, Eerr = quad(lambda v: weight_func(v,betabaromega) * eos_ufg(betamu-v) * v, 0, np.inf, epsrel=eps)
    return Epot


def EF_trap(T, betabaromega, Ns):
    """Computes peak EF for a harmonic trap, for Ns = number of one spin."""
    return T * betabaromega * (6*Ns)**(1/3)


def thermo_trap(T, betamu, betabaromega, weight_func=weight_harmonic):
    """Compute thermodynamics of trapped gas."""
    Ns = number_per_spin(betamu, betabaromega, weight_func)
    EF = EF_trap(T, betabaromega, Ns)
    Theta = T/EF
    Epot = T * Epot_trap(betamu, betabaromega, weight_func)
    return Ns, EF, Theta, Epot


def heating_trap(T, betamu, betaomega, betabaromega, weight_func=weight_harmonic):
    """Compute viscous heating rate E-dot averaged over the trap."""
    Ztrap, Ztraperr = quad(lambda v: weight_func(v,
               betabaromega) * eos_ufg(betamu-v)**(1/3) * zeta(betamu-v, betaomega), 0, np.inf, epsrel=1e-4)
    Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Ztrap
    return Edot 


# unused; this gives weird results
def heating_trap_sumrule(T, betamu, betaomega, betabaromega, weight_func=weight_harmonic):
    """compute viscous heating rate E-dot averaged over the trap normalized by scale sus"""
     
          # Strap,Straperr = quad(lambda v: weight_func(v,
    #                betabaromega)*eos_ufg(betamu-v)**(1/3)*sumrule(betamu-v),0,np.inf,epsrel=1e-4)
    tau = 1/gamma(betamu, T) # inverse scattering rate
    drude_form = tau / (1 + (2*pi*betaomega*T*tau)**2)
    Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*drude_form
    return Edot 

# def phaseshift_qcrit(T, betaomega, betamu):
#     """phi = arctan(omegatau/(1+(omegatau**2)) Eq.(30) in May note"""
#     tau = 1/gamma(betamu, T)
#     phiqcrit = np.arctan((2*pi*betaomega)*T * tau / (1 + (2*pi*betaomega*T*tau)**2))
#     return phiqcrit


## ???
def phaseshift_arg_trap(betamu, betaomega, betabaromega, weight_func=weight_harmonic):
    """Compute viscous heating rate E-dot averaged over the trap"""
    argtrap, argtraperr = quad(lambda v: weight_func(v,
           betabaromega)*eos_ufg(betamu-v)**(1/3)*phaseshift_Drude(betamu-v,betaomega),0,np.inf,epsrel=1e-4)
    argtrap_norm,argtraperr_norm = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)**(1/3),0,
                                        np.inf,epsrel=eps)
    return argtrap/argtrap_norm #, Ztrap/Ztrap_norm # modified to return trap avged zeta


# def phiLR(T, betaomega, betamu):
#     """phi = arctan(omegatau) based on LR theory"""
#     tau = 1/gamma(betamu, T)
#     phiLR = np.arctan(2*pi*betaomega*T * tau)
#     return phiLR

def mutrap_est(ToTF):
    """Dumb guess function for mutrap given ToTF."""
    a = -50e3
    b = 21e3
    return a*ToTF + b


def find_betamu(T, ToTF, betabaromega, weight_func=weight_harmonic, mu_guess=None):
    """Solves for betamu that matches T, EF and betabaromega of trap"""
    if mu_guess == None:
        mu_guess = mutrap_est(ToTF)

    # Find solution to equation EF - EF(mu) = 0.
    sol = root_scalar(lambda x: T/ToTF - T*betabaromega*(6*number_per_spin(x, 
                 betabaromega, weight_func))**(1/3), bracket=[20e3/T, -300e3/T], x0=mu_guess)
    return sol.root, sol.iterations


def sumrule_trap(betamu, betabaromega, weight_func=weight_harmonic):
    """Sumrule in temperature units."""
    sumruleT, sumruleTerr = quad(lambda v: weight_func(v, betabaromega)*betasumrule(betamu-v), 0, np.inf, epsrel=eps)
    return sumruleT


def C_trap_int(betamu, betabaromega, weight_func=weight_harmonic):
    """Contact Density integral, missing some factors."""
    Ctrap, Ctraperr = quad(lambda v: weight_func(v, betabaromega) \
                            * eos_ufg(betamu-v)**(4/3) \
                            * contact_density(Theta(betamu-v)), 0, np.inf, epsrel=eps)
    return Ctrap


def dC_dkFa_inv_trap(betamu, betabaromega, weight_func=weight_harmonic):
    """Compute d \tilde C/d((kF a)^-1) averaged over the trap."""
    integral, err = quad(lambda v: weight_func(v, betabaromega) \
                        * eos_ufg(betamu-v) \
                        * scale_susceptibility(Theta(betamu-v)), 0, np.inf, epsrel=eps)
    Ns = number_per_spin(betamu, betabaromega, weight_func)
    return 18*pi * integral / (2*Ns)


class TrappedUnitaryGas:
    """Object to compute quantities in a trapped unitary Fermi gas. 
    All energies, temperatures and frequencies are in Hz (no 2pi)."""

    def __init__(self, ToTF, EF, barnu, verbose=False):
        self.T = ToTF*EF
        self.ToTF = ToTF
        self.barnu = barnu
        self.lambda_T = np.sqrt(hbar/(mK*self.T))  # Thermal wavelength (unit of length, in meters).
        self.betabaromega = barnu/self.T
        self.birthday = datetime.now()

        # Find betamutrap that produces correct ToTF given EF, ToTF and betabaromega.
        self.betamu, _ = find_betamu(self.T, ToTF, self.betabaromega)

        # Compute trap properties
        self.tau = 1/gamma(self.betamu, self.T) / (2 * pi)
        self.Ns, self.EF, self.Theta, self.Epot = thermo_trap(self.T, 
                                                    self.betamu, self.betabaromega)

        # Self-consistency checks of EF and Theta:
        rtol = 1e-2  # Within 1%
        if not np.allclose(self.EF, EF, rtol=rtol):
                raise ValueError(f"Computed EF={self.EF} not close to given EF={EF}.")
        if not np.allclose(self.ToTF, self.Theta, rtol=rtol):
                raise ValueError(f"Computed T/TF={self.Theta} not close to given T/TF={self.ToTF}.")
        
        self.kF = np.sqrt(2*mK*(2*pi)*self.EF/hbar)  # Global k_F, i.e. peak k_F
        self.Etotal = 2*self.Epot  # Virial theorem valid at unitarity

        self.calc_contact()
        self.calc_dC_dkFa_inv()

        if verbose:
            print(f"A trapped unitary gas with EF={EF:.0f}Hz, T/TF={ToTF:.2f} and barnu={barnu:.0f}Hz")
            print(f"has T={self.T:.0f}Hz, kF={self.kF:.2e} m^-1, lambda_T={self.lambda_T:.2e}m, mu={self.betamu*self.T:.0f}Hz")
            print(f"tau={self.tau:.2e}s, Ns={self.Ns:.0f}, Ntotal={2*self.Ns:.0f}, Epot={self.Epot:.0f}Hz.")
            print(f"C={self.C:.2f}kF")

    def __repr__(self):
        return f"TUG({self.ToTF}, {self.EF}, {self.barnu}) created at {self.birthday.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def evaluate(self, betaomega, value='phiLR'):
        """
        Evaluator for TUG attributes, intended for after calling modulate_field() so self.betaomegas exists
        """
        # make sure attribute exists
        if not hasattr(self, value):
            raise AttributeError(f"TUG object has no attribute '{value}'")
        elif not hasattr(self, "betaomegas"):
            raise AttributeError("TUG object has no attribute 'betaomegas'. Did you call modulate_field()?")
            
        data_y = getattr(self, value)

        # interpolate 
        f = interp1d(self.betaomegas, data_y, kind='linear', fill_value="extrapolate")
        return float(f(betaomega))

    def calc_contact(self, weight_func=weight_harmonic):
        """Calculates the harmonic trap-averaged contact using ToTF, EF, barnu
        (geometric mean trap freq) and an optional guess mu. Returns the contact."""
        integral, _ = quad(lambda v: weight_func(v, self.betabaromega) \
                           *eos_ufg(self.betamu-v)**(4/3) \
                            *contact_density(Theta(self.betamu-v)), 0, np.inf, epsrel=eps)
        self.Ctrap = integral/(self.kF*self.lambda_T)*(3*pi**2)**(4/3)/(2*self.Ns)
        return self.Ctrap
    

    def calc_dC_dkFa_inv(self, weight_func=weight_harmonic):
        """Compute d \tilde C/d((kF a)^-1) averaged over the trap."""
        integral, _ = quad(lambda v: weight_func(v, self.betabaromega) \
                            * eos_ufg(self.betamu-v) \
                            * scale_susceptibility(Theta(self.betamu-v)), 0, np.inf, epsrel=eps)
        self.dCdkFa_inv = 18*pi * integral / (2*self.Ns)
        return self.dCdkFa_inv
    

    def calc_distributions(self):
        """Calculates various distributions of gas parameters in the trap."""
        return ...


    def modulate_field(self, nus, a0=None):
        if a0 is None:  # Then set it s.t. the dimless amp, A, is just 1.
            a0 = self.lambda_T  

        self.A = self.lambda_T/a0  # Dimensionless amplitude of drive.
        self.nus = nus
        self.betaomegas = self.nus/self.T
              
        # but we have to decide if we want to normalize the trap heating rate by the total or by the internal energy
        self.EdotDrude = self.A**2*np.array([heating_trap(self.T,self.betamu,
                        betaomega,self.betabaromega) for betaomega in self.betaomegas])
        
        self.ns = psd_trap(self.betamu,self.betabaromega)/self.Ns # /self.lambda_T**3
    
        self.EdotC = self.A**2*np.array([heating_C(self.T,betaomega, self.Ctrap) for betaomega in self.betaomegas])
        self.EdotNormC = self.A**2*np.array([heating_C(self.T,betaomega, (self.kF*self.lambda_T)/(3*pi**2)**(1/3)) for betaomega in self.betaomegas])
        
        # these were divided by A**4 for some reason when I first saw this code. Why?
        self.zetaDrude = self.EdotDrude/self.A**2 * (self.lambda_T**2*self.kF**2)/(9*pi*nus**2*2*self.Ns)
        self.zetaC = self.EdotC/self.A**2 * (self.lambda_T**2*self.kF**2)/(9*pi*nus**2*2*self.Ns)
        
        self.sumruletrap = sumrule_trap(self.betamu, self.betabaromega) * self.T/self.EF
        
        self.EdotDrudeS = self.EdotDrude / self.sumruletrap
        self.EdotDrudeSalt = self.A**2*np.array([heating_trap_sumrule(self.T,self.betamu, 
                            betaomega, self.betabaromega) for betaomega in self.betaomegas])
        self.phaseshiftsQcrit = np.arctan(self.nus * self.tau / (1 + (self.nus*self.tau)**2))
    
        self.phiLR = np.arctan(2 * np.pi * self.nus * self.tau) # recall tau is 1/2pi
        
        self.EdotSphi = np.array([np.tan(phi)*betaomega*self.T/self.EF * \
                                 9*pi/self.kF**2/self.lambda_T**2 for phi, 
                                 betaomega in zip(self.phaseshiftsQcrit, self.betaomegas)])
        
        # Find frequency to change from Drude to contact scaling.
        # Arbitrarily chose this as 2 * T.
        nu_small = np.searchsorted(nus, 2*self.T, side='right')
    
        # Lists for plotting
        self.Edot = np.append(self.EdotDrude[:nu_small],
                          self.EdotC[nu_small:])/(2*self.Ns * self.EF**2)
        self.zeta = np.append(self.zetaDrude[:nu_small], 
                        self.zetaC[nu_small:])

    def calc_Edot(self, nu):
        betaomega = nu/self.T
        if betaomega < 2:
            Edot = self.A**2 * heating_trap(self.T, self.betamu,
                            betaomega, self.betabaromega)
        else:
            Edot = self.A**2*heating_C(self.T,betaomega, 
                  C_trap_int(self.betamu, self.betabaromega))
        return Edot/(2*self.Ns * self.EF**2)
    
    def calc_zeta(self, nu):
        # Check this, it might not be right.
        zeta = self.calc_Edot(nu)/self.A**2 * (self.lambda_T**2*self.kF**2)/(9*pi*nu**2) * self.EF**2
        return zeta
    
            
def trap_averaged_contact(ToTF, EF, barnu):
    """Calculates the harmonic trap-averaged contact using ToTF, EF, barnu
       (geometric mean trap freq) and an optional guess mu. Returns the contact."""
    T = ToTF * EF
    betabaromega = barnu/T
    lambda_T = np.sqrt(hbar/(mK*T))
    kF = np.sqrt(2*mK*(2*pi)*EF/hbar)  # Global k_F, here peak k_F of harmonic trap
    
    betamu, no_iter = find_betamu(T, ToTF, betabaromega)
    
    # Calculate number of atoms in one spin.
    Ns = (EF/barnu)**3/6
    
    # Calculate C
    Ctrap =  C_trap_int(betamu, betabaromega)/(kF*lambda_T)* \
                (3*pi**2)**(4/3)/Ns/2
                
    return Ctrap


def trap_averaged_contact_slope(ToTF, EF, barnu):
    """Calculates the harmonic trap-averaged contact using ToTF, EF, barnu
       (geometric mean trap freq) and an optional guess mu. Returns the contact."""
    T = ToTF * EF
    betabaromega = barnu/T
    kF = np.sqrt(2*mK*(2*pi)*EF/hbar)  # Global k_F, here peak k_F of harmonic trap
    betamutrap, no_iter = find_betamu(T, ToTF, betabaromega)
    
    # Calculate C slope
    C_slope =  dC_dkFa_inv_trap(betamutrap, betabaromega)
    return C_slope


def contact_from_Edot(Edot, freq, T, kF):
    pifactors = 8*pi*(2*pi)**(2/3)/(3*pi**2)**(1/3)
    factors = (freq/T)**(3/2)/np.sqrt(hbar/(mK*T))/freq**2/(2*pi)
    C = pifactors*factors/kF*Edot
    return C

def phaseshift_from_zeta(nus, EF, zetas):
    sumrule = np.trapezoid(zetas, x=nus)
    phi = np.arctan(nus/EF*zetas/sumrule)
    return phi
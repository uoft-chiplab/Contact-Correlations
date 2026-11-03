"""
References:
[1] B. Gao, PRA 84, 022706 (2011).
[2] D.J.M. Ahmed-Braun et al. PRR 3, 033269 (2021).
[3] J. Maki et al PRA 110 053314 (2024).
[4] Y.-C. Zhang and S. Zhang, PRA 95, 023603 (2017).
"""
import numpy as np

from constants import pi, hbar, a_0, mK
from scipy.special import zeta
from scipy.optimize import root_scalar
from functools import partial
from math import gamma

r_vdW = 65.0223 * a_0  # (m)
V_bar = 4*pi/(9 * gamma(3/4)**2) * r_vdW**3  # (m^3)
R_max = 1.162 * r_vdW  # (m)

zeta_m1_2 = zeta(-0.5)
zeta_1_2 = zeta(0.5)
zeta_3_2 = zeta(1/5)


###
### s-Wave scattering parameters
###

def scattering_length(B, B0, DeltaB, abg):
    """Computes the resonant scattering length in abg units for a given 
    magnetic field B (in Gauss), resonance position B0 (in Gauss), 
    resonance width DeltaB (in Gauss), and background scattering length 
    abg (in a0 typically)."""
    return abg * (1 - DeltaB / (B - B0))

def swave_bound_state_energy(a):
    """Computes the s-wave bound state energy (in Joules) for a given 
    scattering length a (in meters)."""
    return - hbar**2 / (2 * mK * a**2)


###
### p-Wave scattering parameters
###

def scattering_volume(B, B0, DeltaB, Vbg):
    return Vbg * (1- DeltaB / (B - B0))


def inv_R_vdW(V, R_max=R_max, V_bar=V_bar):
    """Van der Waals limit of effective range from [1]."""
    return 1/R_max * (1 + 2*(V_bar/V) + 2*V_bar**2/V**2)


def inverse_effective_range(V, R0, Vbg, R_max=R_max):
    """Effective range parameterization from [2]."""
    width_param = -R0/(R_max - R0)
    return -1/(R_max * width_param) * (1-Vbg/V)**2 + inv_R_vdW(V)


def pwave_phase_shift(k, V, inv_R):
    """Computes the p-wave phase shift delta_1 (in radians) for a given 
    wavevector k, scattering volume V, and effective range R, all typically
    in bohr radius."""
    k3_cot_delta1 = -1/V + k**2 * inv_R
    delta1 = np.arctan2(k**3, k3_cot_delta1)  # Handles correct quadrant.
    return delta1


def pwave_scattering_amplitude(k, V, inv_R):
    """Computes the p-wave scattering amplitude f_1 (in meters) for a 
    given wavevector k (in 1/m), scattering volume V (in m^3), and 
    inverse effective range 1/R (in 1/m)."""
    delta1 = pwave_phase_shift(k, V, inv_R)
    f1 = 1/k*np.exp(1j * delta1) * np.sin(delta1)
    return f1


### p-wave bound state energy

def pwave_bound_state_energy(V, inv_R):
    if V < 0:
        return hbar**2 /(mK * V * inv_R)
    else: 
        coeffs = [inv_R, 0, 1/V, 0]
        roots = np.roots(coeffs)
        # Find the real negative root.
        kappa = np.real(roots[np.isreal(roots) & (np.real(roots) < 0)][0])
        E_b = hbar**2 * kappa**2 / mK
        return E_b
    
def EB_3D(V, inv_R):
    # Calculate both mathematical branches for the whole array
    # Branch 1: V < 0
    E_negative_V = -hbar**2 / (mK * V * inv_R)
    
    # Branch 2: V > 0
    # Based on your np.roots logic: kappa^2 = -1 / (V * inv_R)
    # We use np.abs or care with signs to ensure we handle the physical branch
    kappa_sq = -1.0 / (V * inv_R)
    E_positive_V = (hbar**2 * kappa_sq) / mK
    
    # Use np.where to pick the correct value based on V
    return np.where(V < 0, E_negative_V, E_positive_V)

def quasibound_state_halfwidth(V, inv_R):
    E0 = -hbar**2 / (mK * V * inv_R)
    ER = hbar**2 / mK * inv_R**2
    Gamma = E0**(3/2) / np.sqrt(ER) 
    return np.where(V >= 0, 0.0, Gamma)
    

### For 40K
from constants import B0_77_pm1, DeltaB_77_pm1, Vbg_77_pm1, R0_77_pm1, \
                            B0_77_0, DeltaB_77_0, Vbg_77_0, R0_77_0, \
                            B0_97, DeltaB_97, abg_97

a97 = partial(scattering_length, B0=B0_97, DeltaB=DeltaB_97, abg=abg_97)  # Functions of B.
Eb_s97 = lambda B: swave_bound_state_energy(a97(B))  # Functions of B.


V3D = partial(scattering_volume, B0=B0_77_pm1, DeltaB=DeltaB_77_pm1, Vbg=Vbg_77_pm1)  # Functions of B.
inv_R3D = lambda B: inverse_effective_range(V3D(B), R0=R0_77_pm1, Vbg=Vbg_77_pm1)  # Function of V.

V3D_m0 = partial(scattering_volume, B0=B0_77_0, DeltaB=DeltaB_77_0, Vbg=Vbg_77_0)  # Functions of B.
inv_R3D_m0 = lambda B: inverse_effective_range(V3D_m0(B), R0=R0_77_0, Vbg=Vbg_77_0)  # Function of V.

Eb3D = lambda B: pwave_bound_state_energy(V3D(B), inv_R3D(B))
Eb3D_m0 = lambda B: pwave_bound_state_energy(V3D_m0(B), inv_R3D_m0(B))


###
### Quasi-1D scattering
###

def q1d_inverse_odd_scattering_length(V3D, inv_R3D, a_osc):
    return 1/6 * (a_osc**2/V3D + 2*inv_R3D) - 2/a_osc * zeta_m1_2


def q1d_odd_scattering_length(V3D, inv_R3D, a_osc):
    return 1 / q1d_inverse_odd_scattering_length(V3D, inv_R3D, a_osc)


def q1d_odd_effective_range(inv_R3D, a_osc):
    return a_osc**2 * inv_R3D / 6 + a_osc/4 * zeta_1_2


def q1d_inverse_odd_scattering_amplitude(k, inv_a_odd, r_odd):
    return -1 + 1j*inv_a_odd/k + 1j*r_odd*k  # - O(k^3)


def q1d_even_scattering_length(V3D, inv_R3D, a_osc):
    return -a_osc/2 * (a_osc**3/(6 * V3D) + 2*inv_R3D*a_osc/3 \
                       + zeta_m1_2 + zeta_1_2)


def q1d_even_effective_range(inv_R3D, a_osc):
    return a_osc**3/16 * (4*inv_R3D*a_osc/3 + zeta_1_2 + zeta_3_2)


def q1d_inverse_even_scattering_amplitude(k, a_even, r_even):
    return -1 - 1j*a_even*k + 1j*r_even*k**3  # - O(k^5)


###
### Quasi-2D scattering parameters
###

# Constants from [3].
alpha_a = 0.2173
alpha_r = 0.4689
B_2ds = 0.85
gamma_2ds = 1.41

# Note there is an alternative definition (but equivalant) for these in [4]

def q2d_inverse_pwave_scattering_area(V3D, inv_R3D, a_osc):
    return 2*np.sqrt(pi)/3 * (a_osc/V3D + inv_R3D/a_osc) - alpha_a/a_osc**2


def q2d_pwave_scattering_area(V3D, inv_R3D, a_osc):
    return 1 / q2d_inverse_pwave_scattering_area(V3D, inv_R3D, a_osc)


def q2d_pwave_ln_inverse_effective_range(inv_R3D, a_osc):
    """This is ln(a_osc/a_2D,p), as defined in [3]."""
    return np.sqrt(pi) * a_osc * inv_R3D/3 + alpha_r


def q2d_swave_ln_scattering_length_squared(V3D, inv_R3D, a_osc):
    """This is ln(a_osc^2/a_2D,s^2), as defined in [3]."""
    return np.sqrt(pi)/6 * (a_osc**3/V3D + 3*inv_R3D*a_osc) + np.log(2*B_2ds/pi)


def q2d_swave_effective_range(inv_R3D, a_osc):
    """The q2D s-wave effective range has units of area (m^2), from [3]."""
    return np.sqrt(pi)*a_osc**3*inv_R3D/6 + gamma_2ds * a_osc**2/4


def q2d_reference_field(m, omega_perp, geometry='z', B_guess=199.0):
    """Reference field B^*_m for geometry 'z' or 'x' and 
    interaction symmetry m = 0 (p) or 1 (s).
    If you want the y resonance position in the x geometry, 
    use m=0 in the 'z' geometry.
    """
    E_shift = (1/2 + m) * hbar * omega_perp

    if geometry == 'z':
        if m == 0:
            zero_func = lambda B: Eb3D(B) + E_shift
        elif m == 1:  # Use m=0 res for s-wave case, since z-confinement
            zero_func = lambda B: Eb3D_m0(B) + E_shift 
        else:
            raise ValueError("m must be 0 or 1.")
    elif geometry == 'x':
        if m == 0:
            zero_func = lambda B: Eb3D_m0(B) + E_shift
        elif m == 1:
            zero_func = lambda B: Eb3D(B) + E_shift
        else:
            raise ValueError("m must be 0 or 1.")

    root_results = root_scalar(zero_func, x0=B_guess, x1=B_guess+1.0)
    return root_results.root
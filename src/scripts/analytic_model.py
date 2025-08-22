"""
An analytic model for the effect of radial migration on the stellar metallicity
distribution function (MDF).
"""

import numpy as np
from utils import LinearExponential, NormalDistribution

DISK_SCALE_LENGTH = 2.5 # kpc
SOLAR_RADIUS = 8. # kpc
GRAD_OH = -0.08

def main():
    print(birth_distribution(8))

def birth_distribution(
        Rfinal, 
        sigmaRM8=3.6, 
        alpha=0.5, 
        sfr=None,
        birth_range=(0, 20), 
        dR=0.1, 
        age_range=(0, 13.2), 
        dt=0.1
    ):
    """
    The distribution of birth radii for stars at the given final radius.
    """
    # Set up a coordinate grid
    Rbirth = np.arange(birth_range[0], birth_range[1]+dR, dR)
    ages = np.arange(age_range[0], age_range[1]+dt, dt)[:,np.newaxis]

    # An MxN array of ages (M, rows) and birth radii (N, columns)
    # RR = Rbirth * np.ones(ages.shape)
    # TT = ages * np.ones(Rbirth.shape)

    radial_profile = LinearExponential(
        scale=-DISK_SCALE_LENGTH, 
        coeff=DISK_SCALE_LENGTH**-2
    )
    # metallicity_gradient = lambda r: GRAD_OH * (r - SOLAR_RADIUS)

    # Assume constant SFR if none specified
    if sfr is None:
        sfr = lambda t: 1/age_range[1] * np.ones(t.shape)
    age_distribution = sfr(age_range[1] - ages)

    sigmaRM = lambda t: sigmaRM8 * (t / 8) ** alpha
    migration_probability = NormalDistribution(
        mean=Rfinal, 
        width=sigmaRM(ages)
    )

    migration_distribution = migration_probability(Rbirth)
    # First timestep should be a delta function if time-dependent
    if alpha and age_range[0] == 0:
        migration_distribution[0] = np.zeros(migration_distribution[0].shape)
        migration_distribution[0,int(Rfinal/dR)] = 1.

    dN_dRbirth = radial_profile(Rbirth) * np.sum(
        migration_distribution * age_distribution * dt,
        axis=0
    )
    # Normalize
    dN_dRbirth /= np.sum(dN_dRbirth * dR)
    return dN_dRbirth, Rbirth



if __name__ == '__main__':
    main()

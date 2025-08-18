"""
Global variables for the multizone simulations.
"""

# Seed for random number generation (e.g., Gaussian migration)
RANDOM_SEED = 20250818

# Total simulation time in Gyr
END_TIME = 13.2

# Width of each annulus in kpc
ZONE_WIDTH = 0.1

# Radius in kpc beyond which the SFR = 0
MAX_SF_RADIUS = 15.5

# Stellar mass of Milky Way (Licquia & Newman 2015, ApJ, 806, 96)
M_STAR_MW = 5.17e10

# Simulation timestep size in Gyr
DT = 0.01

# List of elements to simulate
# Order is important! "Reference elements" (e.g., Fe) should come first
ELEMENTS = ["fe", "o"]

# Number of stellar populations per zone per timestep
NSTARS = 8

# The minimum SN Ia delay time in Gyr
MIN_RIA_DELAY = 0.04

# Disk component parameters (Bland-Hawthorn & Gerhard 2016)
THIN_DISK_SCALE_RADIUS = 2.5 # kpc
THICK_DISK_SCALE_RADIUS = 2.0 # kpc
# Thick-to-thin disk surface density ratio
CENTRAL_DISK_RATIO = 0.27 # at r = 0
LOCAL_DISK_RATIO = 0.12 # at r = Rsun
GAS_DISK_SCALE_RADIUS = 3.6 # kpc

# Exponential outflow scale radius
ETA_SCALE_RADIUS = 5.0

import astropy.constants as const
import astropy.units as u
import numpy as np

# Parameters
H0 = (70 * u.km / u.s / u.Mpc).decompose()  # Hubble constant with h=0.7
radius = 10 * u.km  # radius of observable universe/proper distance to particle horizon
Omega_r0 = 1e-4  # Radiation density parameter
Omega_m0 = 0.3  # matter density parameter
T0 = 2.725 * u.K  # CMB temperature today


### SUBPROBLEM 3c
print("\nProblem 3c:\n----------------")
# Redshift
z = np.sqrt(const.c / (H0 * np.sqrt(Omega_r0) * radius)) - 1
print(f"Redshift z at radius {radius:.3e} : {z:.3e}")

# Mass density
rho_c0 = 3 * H0 * H0 / (8 * np.pi * const.G)
rho_m = rho_c0 * Omega_m0 * (1 + z) ** 3
rho_neutron = (
    1.5 * const.M_sun / (4 / 3 * np.pi * radius**3)
).decompose()  # density of typical neutron star
print(
    f"Mass density of universe at above redshift : {rho_m:.3e}   =   {rho_m / rho_neutron:.3e} ρ_neutron_star"
)
print(f"Where the typical neutron star density ρ_neutron_star = {rho_neutron:.3e}")

# Radiation density
rho_r = rho_c0 * Omega_r0 * (1 + z) ** 4
print(
    f"Radiation density of universe at above redshift : {rho_r:.3e}   =   {rho_r / rho_neutron:.3e} ρ_neutron_star"
)

### SUBPROBLEM 3d
print("\nProblem 3d\n----------------")

# CMB/photon temperature at redshift
T = T0 * (1+z)
print(f"CMB temperature T at above redshift : {T:.3e}")


### SUBPROBLEM e)
print("\nProblem 3e\n----------------")
H = H0 * np.sqrt(Omega_r0) * (1+z)**2
t = 1/(2*H)
print(f"Age of the universe at the above redshift (radiation-dominated) : {t:.3e}")


print()

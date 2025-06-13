import cupy as cp
import numpy as np
from cupyx.scipy.fft import fft, ifft
import scipy.constants as const
import time

harmonic_potential = False

start = time.time()
#properties of the grid
N_points = 2**11 #number of gridpoints
x = cp.arange(- N_points // 2, N_points // 2, 1) # spatial grid
k_array = 2*cp.pi*cp.fft.fftfreq(N_points, 1)  # momentum grid
dk = cp.abs(k_array[1] - k_array[0]) # spacing in momentum space

# define physical properties
N_par = 100  # number of particles
m = 1.44*10**(-25) # atom mass
l = 220e-6 # length of the system in meters
n = N_par / N_points # particle density
omega_parallel = 2* cp.pi * 4  # Trapping frequency in longitudinal direction in Hz
omega_perp =  2 * cp.pi * 400 #  Trapping frequency in transverse direction in Hz (isotropic in transverse direction)
a_B = const.physical_constants ["Bohr radius"] [0] # Bohr radius in meter
a0 = 101.8*a_B #Scattering length 0 channel
a2 = 100.4*a_B #Scattering length 2 channel
dx = l  / (N_points) # distance of two points in meters 
sigma = N_points / (4 * cp.sqrt(2)) # std of initial gaussian guess in unitless dimensions

# calculating other parameters
Tscale = m * dx**2 / const.hbar
a_HO = cp.sqrt(const.hbar/(m*omega_perp)) # harmonic oscillator length of transverse trap
c = 4*cp.pi*const.hbar**2/(3*m)*(a0+2*a2)/(2*cp.pi*a_HO**2) # density-density interactions

# transforming in unitless dimensions
c *= m *dx/ (const.hbar**2) # make unitless
omega_parallel *= Tscale  #making frequency unitless

# defining magnetic potential p and interaction strenght c
c = 0.017
mu_spec = c * (1/N_points)
p = cp.linspace(0, 1.5*mu_spec, 100)
lenght_p = len(p)
p = p.reshape(lenght_p, 1)

# defining dimensionless interaction strenght and linear zeeman shift and chemical potential
mu = 0.8 * mu_spec * cp.ones([lenght_p, 1])

# define potential
if harmonic_potential:
    def V(x):
        return (omega_parallel * x)**2
else:
    def V(x):
        return 0 * x

Vx = V(x) * cp.ones([lenght_p, 1]) 
Lap = - k_array**2

# define initial wavefunction
if (not harmonic_potential):
    # define initial wavefunction
    def Psi_Initial_11(x):
        return cp.sqrt(N_par / (4 * N_points)) * cp.ones(N_points) * (1.1 + 0.0 * cp.random.random(N_points))

    def Psi_Initial_12(x):
        return cp.sqrt(N_par / (4 * N_points)) * cp.ones(N_points) * (1.0 + 0.0 * cp.random.random(N_points))

    def Psi_Initial_21(x):
        return cp.sqrt(N_par / (4 * N_points)) * cp.ones(N_points) * (1.0 + 0.0 * cp.random.random(N_points))

    def Psi_Initial_22(x):
        return cp.sqrt(N_par / (4 * N_points)) * cp.ones(N_points) * (1.1 + 0.0 * cp.random.random(N_points))
else:
    # define initial wavefunction
    def Psi_Initial_11(x):
        return cp.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * cp.random.random(N_points))

    def Psi_Initial_12(x):
        return cp.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * cp.random.random(N_points))

    def Psi_Initial_21(x):
        return cp.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * cp.random.random(N_points)) 

    def Psi_Initial_22(x):
        return cp.exp(- x**2 / (2 * sigma**2)) * (1 + 0.05 * cp.random.random(N_points))  

# normalize initial wavefunctions
Psi_i_11 = Psi_Initial_11(x); Psi_i_12 = Psi_Initial_12(x); Psi_i_21 = Psi_Initial_21(x); Psi_i_22 = Psi_Initial_22(x)

norm = cp.sqrt( N_par / (cp.sum(cp.abs(Psi_i_11)**2 + cp.abs(Psi_i_12)**2 + cp.abs(Psi_i_21)**2 + cp.abs(Psi_i_22)**2)))
Psi_i_11 *= norm; Psi_i_12 *= norm; Psi_i_21 *= norm; Psi_i_22 *= norm

# initial iterate
Psi_11 = Psi_i_11 * cp.ones([lenght_p, N_points]); Psi_12 = Psi_i_12 * cp.ones([lenght_p, N_points]); Psi_21 = Psi_i_21 * cp.ones([lenght_p, N_points]); Psi_22 = Psi_i_22 * cp.ones([lenght_p, N_points])
FT_Psi0_11 = fft(Psi_11, axis=1); FT_Psi0_12 = fft(Psi_12, axis=1); FT_Psi0_21 = fft(Psi_21, axis=1); FT_Psi0_22 = fft(Psi_22, axis=1)
FT_Psi1_11 = FT_Psi0_11; FT_Psi1_12 = FT_Psi0_12; FT_Psi1_21 = FT_Psi0_21; FT_Psi1_22 = FT_Psi0_22

abs_Psi_11_squared = cp.abs(Psi_11)**2
abs_Psi_12_squared = cp.abs(Psi_12)**2
abs_Psi_21_squared = cp.abs(Psi_21)**2
abs_Psi_22_squared = cp.abs(Psi_22)**2

rho = abs_Psi_11_squared + abs_Psi_12_squared + abs_Psi_21_squared + abs_Psi_22_squared

PsiX_11 = (mu + p - Vx) * Psi_11 - 2 * c * ((abs_Psi_11_squared + abs_Psi_21_squared + abs_Psi_12_squared) * Psi_11 + Psi_12 * cp.conj(Psi_22) * Psi_21)
PsiX_12 = (mu - Vx) * Psi_12 - 2 * c * ((abs_Psi_11_squared + abs_Psi_22_squared + abs_Psi_12_squared) * Psi_12 + Psi_11 * cp.conj(Psi_21) * Psi_22)
PsiX_21 = (mu - Vx) * Psi_21 - 2 * c * ((abs_Psi_11_squared + abs_Psi_21_squared + abs_Psi_22_squared) * Psi_21 + Psi_22 * cp.conj(Psi_12) * Psi_11)
PsiX_22 = (mu - p - Vx) * Psi_22 - 2 * c * ((abs_Psi_21_squared + abs_Psi_12_squared + abs_Psi_22_squared) * Psi_22 + Psi_21 * cp.conj(Psi_11) * Psi_12)

FT_PsiX_11 = fft(PsiX_11, axis=1); FT_PsiX_12 = fft(PsiX_12, axis=1); FT_PsiX_21 = fft(PsiX_21, axis=1); FT_PsiX_22 = fft(PsiX_22, axis=1) 

#iteration parameters
dt = 0.55; c_pre = 4;               #stepsize and parameter for preconditioner
Restart = 2000                     #for the condition of restarting
restarts = cp.zeros([lenght_p, 1])                    #for counting the number of restarts
ITER = 30000                     #number of maximal iterations
tol=10**(-12)                   #tolerance
jj = 0; ii = cp.zeros([lenght_p, 1]); i = 0 
e_total = cp.ones(4)

P_inv =  (1/(c_pre - Lap))

# beginning the while loop 
while cp.max(e_total)>tol and i < ITER:
    i += 1; ii += cp.ones([lenght_p, 1]); jj += 1

    #iteration
    FT_Psi2_11 = (2 - 3/ii) * FT_Psi1_11 + dt**2 * P_inv * (Lap * FT_Psi1_11 + FT_PsiX_11) - (1 - 3/ii)*FT_Psi0_11
    Psi2_11 = ifft(FT_Psi2_11, axis=1)

    FT_Psi2_12 = (2 - 3/ii) * FT_Psi1_12 + dt**2 * P_inv * (Lap * FT_Psi1_12 + FT_PsiX_12) - (1 - 3/ii)*FT_Psi0_12
    Psi2_12 = ifft(FT_Psi2_12, axis=1)

    FT_Psi2_21 = (2 - 3/ii) * FT_Psi1_21 + dt**2 * P_inv * (Lap * FT_Psi1_21 + FT_PsiX_21) - (1 - 3/ii)*FT_Psi0_21
    Psi2_21 = ifft(FT_Psi2_21, axis=1)

    FT_Psi2_22 = (2 - 3/ii) * FT_Psi1_22 + dt**2 * P_inv * (Lap * FT_Psi1_22 + FT_PsiX_22) - (1 - 3/ii)*FT_Psi0_22
    Psi2_22 = ifft(FT_Psi2_22, axis=1)

    # calculating the new squares
    abs_Psi2_11_squared = cp.abs(Psi2_11)**2
    abs_Psi2_12_squared = cp.abs(Psi2_12)**2
    abs_Psi2_21_squared = cp.abs(Psi2_21)**2
    abs_Psi2_22_squared = cp.abs(Psi2_22)**2

    #gradient restart
    sum1 = cp.sum((cp.conj(Lap * FT_Psi1_11 + FT_PsiX_11)) * (FT_Psi2_11 - FT_Psi1_11), axis=1).reshape(lenght_p, 1)
    sum2 = cp.sum((cp.conj(Lap * FT_Psi1_12 + FT_PsiX_12)) * (FT_Psi2_12 - FT_Psi1_12), axis=1).reshape(lenght_p, 1)
    sum3 = cp.sum((cp.conj(Lap * FT_Psi1_21 + FT_PsiX_21)) * (FT_Psi2_21 - FT_Psi1_21), axis=1).reshape(lenght_p, 1)
    sum4 = cp.sum((cp.conj(Lap * FT_Psi1_22 + FT_PsiX_22)) * (FT_Psi2_22 - FT_Psi1_22), axis=1).reshape(lenght_p, 1)

    cond1 = sum1 + sum2 + sum3 + sum4
    mask = (cond1[:, 0] > 0) & (ii[:, 0] > Restart)
    ii[mask, 0] = 1
    restarts += mask.astype(int).reshape(lenght_p, 1)

    rho = abs_Psi2_11_squared + abs_Psi2_12_squared + abs_Psi2_21_squared + abs_Psi2_22_squared

    # Updating the PsiX terms
    PsiX_11 = (mu + p - Vx) * Psi2_11 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_21_squared + abs_Psi2_12_squared) * Psi2_11 + Psi2_12 * cp.conj(Psi2_22) * Psi2_21)
    PsiX_12 = (mu - Vx) * Psi2_12 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_22_squared + abs_Psi2_12_squared) * Psi2_12 + Psi2_11 * cp.conj(Psi2_21) * Psi2_22)
    PsiX_21 = (mu - Vx) * Psi2_21 - 2 * c * ((abs_Psi2_11_squared + abs_Psi2_21_squared + abs_Psi2_22_squared) * Psi2_21 + Psi2_22 * cp.conj(Psi2_12) * Psi2_11)
    PsiX_22 = (mu - p - Vx) * Psi2_22 - 2 * c * ((abs_Psi2_21_squared + abs_Psi2_12_squared + abs_Psi2_22_squared) * Psi2_22 + Psi2_21 * cp.conj(Psi2_11) * Psi2_12)

    FT_PsiX_11 = fft(PsiX_11, axis = 1); FT_PsiX_12 = fft(PsiX_12, axis = 1); FT_PsiX_21 = fft(PsiX_21, axis = 1); FT_PsiX_22 = fft(PsiX_22, axis = 1)

    # calculating the error
    e_11 = cp.sqrt((1/N_points) * cp.sum(cp.abs(Lap * FT_Psi2_11 + FT_PsiX_11)**2, axis=1))
    e_12 = cp.sqrt((1/N_points) * cp.sum(cp.abs(Lap * FT_Psi2_12 + FT_PsiX_12)**2, axis=1))
    e_21 = cp.sqrt((1/N_points) * cp.sum(cp.abs(Lap * FT_Psi2_21 + FT_PsiX_21)**2, axis=1))
    e_22 = cp.sqrt((1/N_points) * cp.sum(cp.abs(Lap * FT_Psi2_22 + FT_PsiX_22)**2, axis=1))
    
    e_total = cp.array([e_11, e_12, e_21, e_22])

    # updating wavefunctions
    FT_Psi0_11 = FT_Psi1_11; FT_Psi1_11 = FT_Psi2_11
    FT_Psi0_12 = FT_Psi1_12; FT_Psi1_12 = FT_Psi2_12
    FT_Psi0_21 = FT_Psi1_21; FT_Psi1_21 = FT_Psi2_21
    FT_Psi0_22 = FT_Psi1_22; FT_Psi1_22 = FT_Psi2_22

Iterations = i
x_Iter = cp.arange(0, Iterations, 1)

# calculating number of particles
Nparticles = cp.sum(rho, axis=1)[:, cp.newaxis]

# defining the physical components
Psi_pos = Psi2_11; Psi_neg = Psi2_22; Psi_0 = (1 / cp.sqrt(2)) * (Psi2_12 + Psi2_21); eta_0 = (1 / cp.sqrt(2)) * (Psi2_12 - Psi2_21) 

#calculating spin components like on the poster
F_x = (1 / cp.sqrt(2)) * (cp.conj(Psi_pos) * (Psi_0 + eta_0) + cp.conj(Psi_0) * (Psi_pos + Psi_neg) + cp.conj(Psi_neg) * (Psi_0 + eta_0) + cp.conj(eta_0) * (Psi_neg + Psi_neg))
F_y = (1j / cp.sqrt(2)) *(cp.conj(Psi_pos) * (Psi_0 + eta_0) + cp.conj(Psi_0) * (- Psi_pos + Psi_neg) + cp.conj(Psi_neg) * (- Psi_0 + eta_0) - cp.conj(eta_0) * (Psi_neg + Psi_neg))
#F_z = cp.abs(Psi_pos)**2 - cp.conj(Psi_0)*eta_0 - cp.conj(eta_0) * Psi_0 - cp.abs(Psi_neg)**2
F_z = cp.abs(Psi_pos)**2 - cp.abs(Psi_neg)**2

F_perp = cp.real(F_x + 1j * F_y)
F_z = cp.real(F_z)

Magnetisation_per_N = cp.sum(F_z, axis = 1) / Nparticles.T # Magnetisation from the F_z like in the poster

# showing interesting results
#print('Number of Iterations =', Iterations)
#print("Biggest error =", cp.max([e_11, e_12, e_21, e_22].get()))

ende = time.time()
print("Dauer =",- start + ende)

# saving data
print("F_z = ", F_z.get())
print(cp.sum(F_z, axis = 1).get())
print("Nparticles = ", Nparticles.get())
print("Magnetisation per N = ", Magnetisation_per_N)

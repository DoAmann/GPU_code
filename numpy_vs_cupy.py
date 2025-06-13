import numpy as np
import cupy as cp
import time
from scipy.fft import fftn, ifftn
cp.fft.config.enable_nd_planning = True
cp.fft.config.use_multi_gpus = False  # Disable if using 1 GPU
cp.fft.config.use_threading = True

# properties of the grid
N_points = 2**11 # number of gridpoints
x = cp.arange(- N_points // 2, N_points // 2, 1)

def Psi_Initial_0(x):
    return cp.exp(- x**2 / (2)) 

Psi_0 = Psi_Initial_0(x)

start_cp = time.time()
FT_Psi_0 = cp.fft.fftn(Psi_0 + Psi_0 * 300)
end_cp = time.time()

x_np = np.arange(- N_points // 2, N_points // 2, 1)

def Psi_Initial_0(x):
    return np.exp(- x**2 / (2))

Psi_0 = Psi_Initial_0(x_np) 

start_np2 = time.time()
PsiX = Psi_0 + Psi_0 * 300
FT_Psi_0 = fftn(PsiX)
end_np2 = time.time()

start_np = time.time()
FT_Psi_0 = fftn(Psi_0 + Psi_0 * 300)
end_np = time.time()

print("time for gpu = ", end_cp - start_cp)
print("time for scipy = ", end_np - start_np)
print("time for scipy = ", end_np2 - start_np2)
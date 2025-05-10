#!/usr/bin/env python

'''
Input a XC functional which was not implemented in pyscf.
[Rest of comments omitted for brevity]
'''

from pyscf import gto
from pyscf import dft, scf # Import scf for DIIS example
import pylibxc
import numpy as np
import os

# --- Molecule Setup ---
mol = gto.M(
    atom = '''
    O  0.   0.       0.
    H  0.   -0.757   0.587 ''',
    basis = 'ccpvdz',
    spin = 1
)

# --- Parameters ---
hybrid_coeff = 0.54
param_file = "par_pbe0pbe0_old_d3bj.npy"
if not os.path.exists(param_file):
    raise FileNotFoundError(f"Parameter file '{param_file}' not found.")
params = np.load(param_file)[-1][1:]

# --- Custom XC Function ---
def eval_xc_uks(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
    rhoa, rhob = rho

    npl = len(rhoa[0])

    rho = np.zeros(npl * 2)
    sigma = np.zeros(npl * 3)
    lapl = np.zeros(npl * 2)
    tau = np.zeros(npl * 2)

    for i in range(npl):
        rho[i * 2 + 0] = rhoa[0][i]
        rho[i * 2 + 1] = rhob[0][i]
        sigma[i * 3 + 0] = rhoa[1][i] * rhoa[1][i] + rhoa[2][i] * rhoa[2][i] + rhoa[3][i] * rhoa[3][i]
        sigma[i * 3 + 1] = rhoa[1][i] * rhob[1][i] + rhoa[2][i] * rhob[2][i] + rhoa[3][i] * rhob[3][i]
        sigma[i * 3 + 2] = rhob[1][i] * rhob[1][i] + rhob[2][i] * rhob[2][i] + rhob[3][i] * rhob[3][i]
        lapl[i * 2 + 0] = rhoa[4][i]
        lapl[i * 2 + 1] = rhob[4][i]
        tau[i * 2 + 0] = rhoa[5][i]
        tau[i * 2 + 1] = rhob[5][i]


    params = np.load("par_pbe0pbe0_old_d3bj.npy")[-1][1:]
    funcc = pylibxc.LibXCFunctional("MGGA_C_M06_2X", "polarized")
    funcx = pylibxc.LibXCFunctional("HYB_MGGA_X_M06_2X", "polarized")

    ext_c = np.zeros(27)
    for i in range(22):
        ext_c[i + 4] = params[i]
    for i in range(4):
        ext_c[i] = params[i + 22]
    ext_c[26] = 1e-10

    ext_x = np.zeros(14)
    ext_x[:13] = params[26:26+13]
    ext_x[13] = 0.54 #nothing changed
    ext_x[12] = 1 #change
    # ext_x[0] = 0.46 #changed

    funcc.set_ext_params(np.array(ext_c))
    funcx.set_ext_params(np.array(ext_x))

    inp = {"rho": rho, "sigma": sigma, "tau": tau}
    retx = funcx.compute(inp)
    retc = funcc.compute(inp)

    exc = retx['zk'] + retc['zk']

    vxc, fxc, kxc = None, None, None

    vrho = retx["vrho"] + retc["vrho"]       # Shape (Ngrid, 2)
    vsigma = retx["vsigma"] + retc["vsigma"] # Shape (Ngrid, 3)
    vtau = retx["vtau"] + retc["vtau"]       # Shape (Ngrid, 2)

    vlapl = None
    vxc = (vrho, vsigma, vlapl, vtau)

    fxc = None
    kxc = None
    return exc, vxc, fxc, kxc

# --- Run PySCF calculation ---
mf = dft.UKS(mol)
mf = mf.define_xc_(eval_xc_uks, 'MGGA', hyb=hybrid_coeff)
mf.verbose = 4
# Using DIIS is generally recommended for stability
mf.diis = scf.DIIS()
mf.kernel()

# --- Print Results ---
print("Calculation Finished")
print(f"Total Energy = {mf.e_tot}")

mf = dft.UKS(mol)
mf.xc = "M06-2X"
mf.verbose = 4
# Using DIIS is generally recommended for stability
mf.diis = scf.DIIS()
mf.kernel()

# --- Print Results ---
print("Calculation Finished")
print(f"Total Energy = {mf.e_tot}")
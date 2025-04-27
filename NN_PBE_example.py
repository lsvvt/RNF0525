from functionals import NN_FUNCTIONAL

from pyscf import gto
from pyscf import dft

# Create the molecule of interest and select the basis set.
mol = gto.Mole()
mol.atom = 'Ne 0.0 0.0 0.0'
mol.basis = 'cc-pVDZ'
mol.build()

# Create a DFT solver and insert the DM21 functional into the solver.
mf = dft.RKS(mol)
model = NN_FUNCTIONAL("NN_PBE_18")
mf.define_xc_(model.eval_xc, 'MGGA')

# Run the DFT calculation.
mf.kernel()
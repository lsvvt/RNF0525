# first line: 61
@memory.cache
def compute_energy_pyscf(geometry_data, basis_set, xc):
    """
    Собирает молекулу (Mole) и считает энергию в PySCF.
    method: "HF" или "DFT" (при желании расширить).
    Возвращает энергию в Hartree (float).
    """
    mol = build_pyscf_mol(geometry_data, basis_set)
    mf = dft.RKS(mol, xc=xc).density_fit().to_gpu()

    # mf.grids.level = 5
    # mf.conv_tol = 1E-9
    # mf.conv_tol_grad = 1E-6
    # mf.with_df.auxbasis = "def2-universal-jfit"
    energy = mf.kernel()
    # print(geometry_data["id"], energy)
    return energy

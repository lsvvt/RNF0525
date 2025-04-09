# first line: 149
@memory.cache
def compute_energy_pyscf(geometry_data, basis_set, xc):
    """
    Собирает молекулу (Mole) и считает энергию в PySCF.
    method: "HF" или "DFT" (при желании расширить).
    Возвращает энергию в Hartree (float).
    """
    mol = build_pyscf_mol(geometry_data, basis_set)
    mf = dft.RKS(mol).density_fit()#.to_gpu()

    if xc == "DM21":
        mf.xc = 'B3LYP'
        mf.run()
        dm0 = mf.make_rdm1()

        mf._numint = dm21.NeuralNumInt(dm21.Functional.DM21)

        mf.conv_tol = 1E-6
        mf.conv_tol_grad = 1E-3

        energy = mf.kernel(dm0=dm0)
    else:
        if xc == "PBE-2X":
            mf.xc = "PBE*0.46+HF*0.54,PBE"
        elif xc == "piM06-2X-DL":
            mf = mf.define_xc_(eval_gga_xc_pidl, 'MGGA', hyb=0.54)
        elif xc == "piM06-2X":
            mf = mf.define_xc_(eval_gga_xc_pi, 'MGGA', hyb=0.54)
        else:
            mf.xc = xc
        energy = mf.kernel()

    return energy

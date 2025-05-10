import yaml
import numpy as np
from pyscf import gto, scf, dft
from progress_table import ProgressTable
import pandas as pd
# from joblib import Memory
import os
try:
    import density_functional_approximation_dm21 as dm21
except:
    pass
import jsonpickle
# import pylibxc
import sys
try:
    from functionals import NN_FUNCTIONAL
except:
    pass

from mokit.lib.fch2py import fch2py
from mokit.lib.gaussian import load_mol_from_fch, mo_fch2py




cachedir = os.path.join(os.getcwd(), 'mycache')
# memory = Memory(location=cachedir, verbose=0)

import warnings
warnings.filterwarnings("ignore", message="using cupy as the tensor contraction engine.")

datav2 = {"r":[], "xc": [], "basis": [], "e": []}

dm_l = []

# Константа перевода из Hartree в kcal/mol
HARTREE_TO_KjmolL = 2625.5

def load_reactions_database(yaml_file):
    """
    Считывает YAML-файл с базой (geometries + reactions).
    Возвращает словарь: {"geometries": [...], "reactions": [...]}.
    """
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def build_pyscf_mol(geometry_data, basis_set):
    """
    Создаёт объект pyscf.gto.Mole из записи о геометрии:
      - 'atomic_coordinates' (список строк "Atom x y z")
      - 'charge'
      - 'multiplicity'
      - 'basis_set'
    Возвращает объект Mole, готовый к расчёту.
    """
    # Соберём список кортежей (atom_label, (x,y,z))
    # или можно напрямую склеить в одну строку формата PySCF
    atoms_for_pyscf = []
    for coord_line in geometry_data["atomic_coordinates"]:
        parts = coord_line.split()
        atom_label = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms_for_pyscf.append((atom_label, (x, y, z)))

    # Если charge=0, multiplicity=1 — то spin = 0
    # Если multiplicity=2 => spin=1 и т.п.
    spin = geometry_data["multiplicity"] - 1

    mol = gto.Mole()

    mol = gto.Mole()
    mol.atom      = atoms_for_pyscf
    mol.unit      = "Å"
    mol.charge    = geometry_data["charge"]
    mol.spin      = spin                # 2S = 0  →  singlet
    mol.basis     = basis_set
    # mol.symmetry  = True              # turn symmetry support on
    # mol.groupname = "D4h"             # request full D4h (needs libmsym)
    # mol.verbose=4
    mol.build()
    # mol.build(
    #     atom=atoms_for_pyscf,
    #     charge=geometry_data["charge"],
    #     spin=spin,
    #     basis=basis_set,
    #     verbose=0  # Меньше логов
    # )
    return mol

def eval_gga_xc_pidl(xc_code, rho, spin=0, relativity=0, deriv=2, omega=None, verbose=None):
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


def eval_gga_xc_pi(xc_code, rho, spin=0, relativity=0, deriv=2, omega=None, verbose=None):
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


    params = np.load("par_pbe0pbe0_old.npy")[-5][1:]
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


def get_mo_coeff(id, mf):
    
    for f in os.listdir("orca_HOCl_fch/orca_HOCl"):
        if id in f:
            hf_fch = "orca_HOCl_fch/orca_HOCl/" + f + "/input.fch"

    nbf = mf.mo_coeff[0].shape[0]
    nif = mf.mo_coeff[0].shape[1]
    alpha_coeff = fch2py(hf_fch, nbf, nif, 'a')
    beta_coeff = fch2py(hf_fch, nbf, nif, 'b')
    return (alpha_coeff, beta_coeff)


def get_mol(id):
    
    for f in os.listdir("orca_HOCl_fch/orca_HOCl"):
        if id in f:
            hf_fch = "orca_HOCl_fch/orca_HOCl/" + f + "/input.fch"

    return load_mol_from_fch(hf_fch)


# @memory.cache
def compute_energy_pyscf(geometry_data, basis_set, xc, fyaml):
    """
    Собирает молекулу (Mole) и считает энергию в PySCF.
    method: "HF" или "DFT" (при желании расширить).
    Возвращает энергию в Hartree (float).
    """
    mol = build_pyscf_mol(geometry_data, basis_set)
    mol = get_mol(geometry_data["id"])
    mol.verbose=3
    mol.build()

    if "gpu" in sys.argv:
        mf = dft.UKS(mol).density_fit().to_gpu()
    else:
        mf = dft.UKS(mol).density_fit()

    if fyaml == "h4_database.yaml":
        mf.DIIS = scf.ADIIS
    # mf.DIIS = scf.ADIIS

    if xc == "DM21":
        mf.xc = 'B3LYP'
        mf.run()
        dm0 = mf.make_rdm1()

        mf._numint = dm21.NeuralNumInt(dm21.Functional.DM21)

        mf.conv_tol = 1E-6
        mf.conv_tol_grad = 1E-3

        if fyaml == "h4_database.yaml":
            mf.DIIS = scf.DIIS
            energy = mf.kernel()
        else:
            energy = mf.kernel(dm0=dm0)

        print("E", energy)
    else:
        if xc == "PBE-2X":
            mf.xc = "PBE*0.46+HF*0.54,PBE"
        elif xc == "piM06-2X-DL":
            mf = mf.define_xc_(eval_gga_xc_pidl, 'MGGA', hyb=0.54)
        elif xc == "piM06-2X":
            mf = mf.define_xc_(eval_gga_xc_pi, 'MGGA', hyb=0.54)
        elif xc == "NN-PBE":
            model = NN_FUNCTIONAL("NN_PBE_18")
            mf.define_xc_(model.eval_xc, 'MGGA')
        else:
            mf.xc = xc
        
        mf.kernel()
        mf.mo_coeff = get_mo_coeff(geometry_data["id"], mf)
        # if len(dm_l) > 0:
        #     energy = mf.kernel(dm0=dm_l[-1])
        # else:
        #     energy = mf.kernel()
        energy = mf.kernel()
        
        

    dm = mf.make_rdm1()
    # mf.stable_opt_internal()
    return energy, dm

def main(basis_set, xc, fyaml):
    # Пусть наш сгенерированный YAML-файл называется reactions_database.yaml
    data = load_reactions_database(fyaml)

    geometries = data.get("geometries", [])
    reactions = data.get("reactions", [])

    # Считаем энергии для каждой геометрии
    computed_energies = {}
    for geom in table(geometries):
        geom_id = geom["id"]
        table["geom_id"] = geom_id
        # print(geom_id)
        # Меняем метод при необходимости (HF, DFT, CCSD, ...)
        level = geom_id + "_" + xc + "_" + basis_set
        if os.path.exists(f"bak/{level}"):
            with open(f"bak/{level}", "r") as f:
                e_hartree = jsonpickle.decode(f.read())
            # with open(f"bak/{level}", "w") as f:
            #     f.write(jsonpickle.encode(float(e_hartree)))
        else:                
            e_hartree, dm = compute_energy_pyscf(geom, basis_set, xc, fyaml)

            if "HOCl" in geom["id"]:
                dm_l.append(dm)

            with open(f"bak/{level}", "w") as f:
                f.write(jsonpickle.encode(float(e_hartree)))

        
        computed_energies[geom_id] = e_hartree


    # Теперь считаем ошибки для реакций
    abs_errors = []  # будем хранить |E_calc - E_ref| для каждой реакции

    for rxn in reactions:
        # Допустим, в rxn["reactants"] и rxn["products"] у нас по одному компоненту:
        # reactants = [ { geometry_id: "...", stoichiometry: 1 }, ... ]
        # products  = [ { geometry_id: "...", stoichiometry: 1 }, ... ]
        # И есть rxn["details"]["reference_energy_kcal_mol"]
        # По условию (1 минус 2), примем E_calc = E(reactants) - E(products).
        # Если у вас сложнее стехиометрия, суммируйте соответствующие энергии.

        # Соберём полную энергию «реактантов» (с учётом стехиометрии)
        E_reactants_hartree = 0.0
        for r in rxn.get("reactants", []):
            geom_id_r = r["geometry_id"]
            stoich_r = r.get("stoichiometry", 1)
            E_reactants_hartree += stoich_r * computed_energies[geom_id_r]

        # Соберём полную энергию «продуктов»
        E_products_hartree = 0.0
        for p in rxn.get("products", []):
            geom_id_p = p["geometry_id"]
            stoich_p = p.get("stoichiometry", 1)
            E_products_hartree += stoich_p * computed_energies[geom_id_p]

        # Расчётная разность (в Hartree)
        E_calc_diff_hartree = E_reactants_hartree - E_products_hartree
        # Переводим в kcal/mol
        E_calc_diff_kcalmol = E_calc_diff_hartree * HARTREE_TO_KjmolL

        # Сравниваем с reference_energy_kcal_mol
        ref_energy = rxn["details"].get("reference_energy_kcal_mol", None)

        if fyaml == "h4_database.yaml":
            error = E_calc_diff_kcalmol
        elif fyaml == "hocl_dissociation_database.yaml":
            error = E_calc_diff_hartree * 627.509 - ref_energy
            # print(basis_set, error, xc, rxn["id"].split("_d")[1])
            datav2["e"].append(error)
            datav2["r"].append(rxn["id"].split("_d")[1])
            datav2["basis"].append(basis_set)
            datav2["xc"].append(xc)
        else:
            error = E_calc_diff_kcalmol - ref_energy
        
        abs_errors.append(abs(error))

        # print(f"Реакция: {rxn['id']} / Расчёт: {E_calc_diff_kcalmol:.2f} кдж/моль / "
        #       f"Референс: {ref_energy:.2f} кдж/моль / Ошибка: {error:.2f}")

    mae = (sum(abs_errors) / len(abs_errors))
    # print(f"\nMAE по базе: {mae:.3f} кдж/моль")

    return mae, max(abs_errors)

if __name__ == "__main__":
    table = ProgressTable(num_decimal_places=3,
    pbar_style="square",  # specify specific style
    pbar_embedded=False,  # force all progress bars to be non-embedded)
    pbar_show_progress=True,
    pbar_show_percents=True,
    pbar_show_eta=True,
    )
    table.add_column("xc")

    fyaml = sys.argv[1]


    data = {
        "basis_set": [],
        "xc": [],
        "mae": [],
        "maxe": [],
        # "is_hybrid": [],
    }

    # is_hybrid = ["non hybrid"] * 6 + ["hybrid"] * 9
    for i, xc in enumerate(table(["NN-PBE", "LDA", "M06-L", "PBE", "TPSS", "r2SCAN", "SCAN", "SCAN0", "PBE-2X", "PBE0", "M06-2X", "M05-2X", "B3LYP", "DM21", "piM06-2X-DL", "piM06-2X"])):

        table["xc"] = xc

        if fyaml == "ctb22_reactions_database.yaml":
            basis_set = "cc-pVQZ"
        elif fyaml == "hocl_dissociation_database.yaml":
            basis_set = "aug-cc-pVTZ"
        elif fyaml == "h4_database.yaml":
            basis_set = "6-31G"

        table["basis_set"] = basis_set
        dm_l = []

        mae, maxe = main(basis_set, xc, sys.argv[1])

        # try:
        #     mae, maxe = main(basis_set, xc, sys.argv[1])
        # except:
        #     mae, maxe = 0, 0

        data["basis_set"].append(basis_set)

        xc_tmp = xc
        if xc == "PBE*0.46+HF*0.54,PBE":
            xc_tmp = "PBE-2X"
        data["xc"].append(xc_tmp)
        
        data["mae"].append(mae)
        data["maxe"].append(maxe)
        # data["is_hybrid"].append(is_hybrid[i])

        table["mae"] = mae
        table["maxe"] = maxe

        table.next_row()

    table.close()

    df = pd.DataFrame(data)
    df.to_csv(sys.argv[1] + '_data.csv', index=False)  


    if fyaml == "hocl_dissociation_database.yaml":
        df = pd.DataFrame.from_dict(datav2)
        df.to_csv('hocl_new_data.csv', index=False)  
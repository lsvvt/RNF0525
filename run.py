import yaml
import math
from pyscf import gto, scf, dft
from progress_table import ProgressTable
import pandas as pd
from joblib import Memory
import os

cachedir = os.path.join(os.getcwd(), 'mycache')
memory = Memory(location=cachedir, verbose=0)

import warnings
warnings.filterwarnings("ignore", message="using cupy as the tensor contraction engine.")



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
    mol.build(
        atom=atoms_for_pyscf,
        charge=geometry_data["charge"],
        spin=spin,
        basis=basis_set,
        verbose=0  # Меньше логов
    )
    return mol

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

def main(basis_set, xc):
    # Пусть наш сгенерированный YAML-файл называется reactions_database.yaml
    data = load_reactions_database("reactions_database.yaml")

    geometries = data.get("geometries", [])
    reactions = data.get("reactions", [])

    # Считаем энергии для каждой геометрии
    computed_energies = {}
    for geom in table(geometries):
        geom_id = geom["id"]
        table["geom_id"] = geom_id
        # print(geom_id)
        # Меняем метод при необходимости (HF, DFT, CCSD, ...)
        e_hartree = compute_energy_pyscf(geom, basis_set, xc)
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



    data = {
        "basis_set": [],
        "xc": [],
        "mae": [],
        "maxe": [],
        "is_hybrid": [],
    }
    is_hybrid = ["non hybrid"] * 6 + ["hybrid"] * 6
    for i, xc in enumerate(table(["LDA", "M06-L", "PBE", "TPSS", "r2SCAN", "SCAN", "SCAN0", "PBE*0.46+HF*0.54,PBE", "PBE0", "M06-2X", "M05-2X", "B3LYP"])):

        table["xc"] = xc

        for basis_set in table(["cc-pVQZ", "cc-pVTZ", ]):#"cc-pVDZ", "def2-QZVP", "def2-TZVP", "def2-SVP"]):
            table["basis_set"] = basis_set

            mae, maxe = main(basis_set, xc)

            data["basis_set"].append(basis_set)

            xc_tmp = xc
            if xc == "PBE*0.46+HF*0.54,PBE":
                xc_tmp = "PBE-2X"
            data["xc"].append(xc_tmp)
            
            data["mae"].append(mae)
            data["maxe"].append(maxe)
            data["is_hybrid"].append(is_hybrid[i])

            table["mae"] = mae
            table["maxe"] = maxe

            table.next_row()

    table.close()

    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)  

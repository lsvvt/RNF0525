import yaml
import numpy as np
import os

# Conversion factor
HARTREE_TO_KCAL_MOL = 627.509

# Atomic numbers dictionary (reused for electron count)
ATOMIC_NUMBERS = {
    'H': 1,  'He': 2, 'Li': 3, 'Be': 4, 'B': 5,  'C': 6,  'N': 7,  'O': 8,
    'F': 9,  'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15,  'S':16,
    'Cl':17, 'Ar':18, 'K':19,  'Ca':20, 'Sc':21, 'Ti':22, 'V':23,  'Cr':24,
    'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32,
    'As':33, 'Se':34, 'Br':35, 'Kr':36, 'Rb':37, 'Sr':38, 'Y':39,  'Zr':40
    # Add more if needed
}

def calculate_electrons(formula_dict):
    """Calculates total electrons based on atomic numbers."""
    total_electrons = 0
    for atom_label, count in formula_dict.items():
        Z = ATOMIC_NUMBERS.get(atom_label)
        if Z is None:
            print(f"Warning: Atom {atom_label} not found in ATOMIC_NUMBERS!")
            Z = 0 # Assign 0 if unknown, though this shouldn't happen for HOCl
        total_electrons += Z * count
    return total_electrons

def parse_reference_energies(filepath):
    """Parses the CCSD(T) energy file."""
    energies = {}
    e_ho = None
    e_cl_minus = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"Skipping invalid line in {filepath}: {line}")
                continue
            label, energy_str = parts
            try:
                energy = float(energy_str)
                if label.upper() == 'OH':
                    e_ho = energy
                elif label.upper() == 'CL':
                    e_cl_minus = energy
                else:
                    # Assume it's a distance
                    distance = float(label)
                    energies[distance] = energy
            except ValueError:
                print(f"Skipping line with non-numeric value in {filepath}: {line}")
                continue

    if e_ho is None:
        raise ValueError("Energy for HO not found in reference file.")
    if e_cl_minus is None:
        raise ValueError("Energy for Cl- not found in reference file.")

    return e_ho, e_cl_minus, energies

def generate_geometry_dict(geom_id, formula_dict, coords, charge, multiplicity, basis_set="CCSD(T)"):
    """Creates a dictionary representing a geometry."""
    electrons = calculate_electrons(formula_dict)
    # Adjust for charge
    electrons -= charge

    formula_parts = []
    for a_label in sorted(formula_dict.keys()):
        count = formula_dict[a_label]
        if count == 1:
            formula_parts.append(a_label)
        else:
            formula_parts.append(f"{a_label}{count}")
    formula = "".join(formula_parts)

    return {
        "id": geom_id,
        "formula": formula,
        "atomic_coordinates": [f"{atom} {x:.8f} {y:.8f} {z:.8f}" for (atom, x, y, z) in coords],
        "charge": charge,
        "multiplicity": multiplicity,
        "electrons": electrons,
        "basis_set": basis_set
    }

def main():
    ref_energy_file = "HOCl_CCSD(T).txt"
    output_yaml = "hocl_dissociation_database.yaml"
    oh_bond_length = 1.0 # Angstrom

    # 1. Parse Reference Energies
    try:
        e_ho_ref, e_cl_minus_ref, e_hocl_dissoc_curve = parse_reference_energies(ref_energy_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error processing reference energy file: {e}")
        return

    geometries = []
    reactions = []

    # 2. Define HO Geometry
    ho_formula = {'H': 1, 'O': 1}
    ho_coords = [('O', 0.0, 0.0, 0.0), ('H', oh_bond_length, 0.0, 0.0)]
    ho_geom = generate_geometry_dict("HO", ho_formula, ho_coords, charge=0, multiplicity=2)
    geometries.append(ho_geom)

    # 3. Define Cl- Geometry
    cl_minus_formula = {'Cl': 1}
    cl_minus_coords = [('Cl', 0.0, 0.0, 0.0)]
    cl_minus_geom = generate_geometry_dict("Cl", cl_minus_formula, cl_minus_coords, charge=-1, multiplicity=1)
    geometries.append(cl_minus_geom)

    # 4. Define HO...Cl- Geometries and Reactions for each distance
    hocl_formula = {'H': 1, 'O': 1, 'Cl': 1}
    distances = sorted(e_hocl_dissoc_curve.keys())

    for dist in distances:
        # Define HOCl geometry for this distance (linear H-O...Cl)
        # Place O at origin, H along -x, Cl along +x
        hocl_coords = [
            ('O', -oh_bond_length, 0.0, 0.0),
            ('H', 0.0, 0.0, 0.0),
            ('Cl', dist, 0.0, 0.0)
        ]
        hocl_id = f"HOCl_d{dist:.2f}" # Use 2 decimal places for ID
        hocl_geom = generate_geometry_dict(hocl_id, hocl_formula, hocl_coords, charge=-1, multiplicity=2)
        geometries.append(hocl_geom)

        # Calculate Dissociation Energy
        e_hocl_ref = e_hocl_dissoc_curve[dist]
        # Dissociation Energy = E(products) - E(reactant) = (E_HO + E_Cl-) - E_HOCl
        dissoc_energy_hartree = (e_ho_ref + e_cl_minus_ref) - e_hocl_ref
        dissoc_energy_kcal_mol = dissoc_energy_hartree * HARTREE_TO_KCAL_MOL

        # Create Reaction Entry
        reaction_id = f"dissoc_{hocl_id}"
        reaction_dict = {
            "id": reaction_id,
            "description": f"Dissociation of HO...Cl- complex at O-Cl distance {dist:.2f} A",
            "reactants": [
                {"geometry_id": hocl_geom["id"], "stoichiometry": 1}
            ],
            "products": [
                {"geometry_id": ho_geom["id"], "stoichiometry": 1},
                {"geometry_id": cl_minus_geom["id"], "stoichiometry": 1}
            ],
            "details": {
                "reference_energy_kcal_mol": dissoc_energy_kcal_mol
            }
        }
        reactions.append(reaction_dict)

    # 5. Prepare final data structure and write YAML
    data_for_yaml = {
        "geometries": geometries,
        "reactions": reactions
    }

    try:
        with open(output_yaml, 'w', encoding='utf-8') as out:
            # Use default_flow_style=None for better readability (block style)
            # Use sort_keys=False to maintain insertion order
            yaml.dump(data_for_yaml, out, sort_keys=False, allow_unicode=True, default_flow_style=None, indent=2)
        print(f"File {output_yaml} successfully generated.")
    except IOError as e:
        print(f"Error writing YAML file: {e}")

if __name__ == "__main__":
    main()
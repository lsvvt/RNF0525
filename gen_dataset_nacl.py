import os
import yaml
import numpy as np # Using numpy for easier array handling

# Dictionary of approximate atomic numbers (for calculating electron count)
# Extend as needed for larger systems.
ATOMIC_NUMBERS = {
    'H': 1,  'He': 2, 'Li': 3, 'Be': 4, 'B': 5,  'C': 6,  'N': 7,  'O': 8,
    'F': 9,  'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15,  'S':16,
    'Cl':17, 'Ar':18, 'K':19,  'Ca':20, 'Sc':21, 'Ti':22, 'V':23,  'Cr':24,
    'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32,
    'As':33, 'Se':34, 'Br':35, 'Kr':36, 'Rb':37, 'Sr':38, 'Y':39,  'Zr':40
    # etc.
}

def parse_nacl_energies(file_path):
    """
    Parses the NaCl_CCSD(T).txt file.

    Returns:
        tuple: (energy_na, energy_cl, nacl_energies_dict)
               energy_na (float): Energy of isolated Na atom (Hartree).
               energy_cl (float): Energy of isolated Cl atom (Hartree).
               nacl_energies_dict (dict): Dictionary {distance (float): energy (float)}
                                          for NaCl molecule (Hartree).
    """
    energy_na = None
    energy_cl = None
    nacl_energies = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue

            label = parts[0]
            try:
                energy = float(parts[1])
            except ValueError:
                print(f"Warning: Skipping line with non-numeric energy: {line}")
                continue

            if label == 'Na':
                energy_na = energy
            elif label == 'Cl':
                energy_cl = energy
            else:
                # Assume it's a distance if it's a number
                try:
                    distance = float(label)
                    # Check if distance is within the expected range
                    if 2.0 <= distance <= 8.0:
                         nacl_energies[distance] = energy
                    else:
                         print(f"Warning: Skipping distance outside expected range [2.0, 8.0]: {line}")
                except ValueError:
                    print(f"Warning: Skipping line with unrecognized label: {line}")

    if energy_na is None:
        raise ValueError(f"Energy for 'Na' not found in {file_path}")
    if energy_cl is None:
        raise ValueError(f"Energy for 'Cl' not found in {file_path}")
    if not nacl_energies:
        raise ValueError(f"No NaCl energies found in the expected format in {file_path}")

    # Sort by distance for predictable order
    sorted_nacl_energies = dict(sorted(nacl_energies.items()))

    return energy_na, energy_cl, sorted_nacl_energies


def create_geometry(geom_id, formula, coordinates, charge, multiplicity, basis):
    """Helper function to create a geometry dictionary."""
    electrons = 0
    # Simple formula parsing for electron count (assumes single letters or letter+number)
    import re
    atom_counts = {}
    for part in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        atom, count_str = part
        count = int(count_str) if count_str else 1
        atom_counts[atom] = atom_counts.get(atom, 0) + count

    for atom, count in atom_counts.items():
         Z = ATOMIC_NUMBERS.get(atom)
         if Z is None:
              print(f"Warning: Atom {atom} not found in ATOMIC_NUMBERS dictionary!")
              Z = 0 # Assign 0 electrons if unknown
         electrons += Z * count

    return {
        "id": geom_id,
        "formula": formula,
        "atomic_coordinates": coordinates,
        "charge": charge,
        "multiplicity": multiplicity,
        "electrons": electrons,
        "basis_set": basis
    }

def main():
    input_energy_file = "NaCl_CCSD(T).txt"
    output_yaml = "nacl_dissociation_database.yaml"
    basis_description = "Reference CCSD(T)" # Or adjust as needed

    print(f"Parsing energies from: {input_energy_file}")
    try:
        e_na, e_cl, nacl_energies = parse_nacl_energies(input_energy_file)
        print(f"  Found E(Na) = {e_na:.6f} Hartree")
        print(f"  Found E(Cl) = {e_cl:.6f} Hartree")
        print(f"  Found {len(nacl_energies)} NaCl energy points.")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error parsing input file: {e}")
        return

    geometries = []
    reactions = []

    # 1. Define Atomic Geometries
    na_geom = create_geometry(
        geom_id="Na_atom",
        formula="Na",
        coordinates=["Na 0.0 0.0 0.0"],
        charge=0,
        multiplicity=1, # As per plan confirmation
        basis=basis_description
    )
    cl_geom = create_geometry(
        geom_id="Cl_atom",
        formula="Cl",
        coordinates=["Cl 0.0 0.0 0.0"],
        charge=0,
        multiplicity=1, # As per plan confirmation
        basis=basis_description
    )
    geometries.extend([na_geom, cl_geom])
    print("Defined Na and Cl atom geometries.")

    # 2. Define NaCl Geometries and Dissociation Reactions
    print("Generating NaCl geometries and dissociation reactions...")
    for distance, e_nacl in nacl_energies.items():
        # Format distance to avoid floating point issues in IDs/filenames
        dist_str = f"{distance:.1f}"

        # Create NaCl geometry for this distance
        nacl_geom_id = f"NaCl_R_{dist_str}"
        nacl_coords = [
            "Na 0.0 0.0 0.0",
            f"Cl {dist_str} 0.0 0.0" # Place Cl along X-axis
        ]
        nacl_geom = create_geometry(
            geom_id=nacl_geom_id,
            formula="NaCl",
            coordinates=nacl_coords,
            charge=0,
            multiplicity=1, # Singlet ground state for NaCl molecule
            basis=basis_description
        )
        geometries.append(nacl_geom)

        # Calculate dissociation energy: E(Na) + E(Cl) - E(NaCl)
        dissociation_energy = e_na + e_cl - e_nacl

        # Create reaction entry
        reaction_id = f"NaCl_dissociation_R_{dist_str}"
        reaction_dict = {
            "id": reaction_id,
            "description": f"Dissociation energy of NaCl at R={dist_str} Angstrom",
            "reactants": [
                {"geometry_id": nacl_geom_id, "stoichiometry": 1}
            ],
            "products": [
                {"geometry_id": na_geom["id"], "stoichiometry": 1},
                {"geometry_id": cl_geom["id"], "stoichiometry": 1}
            ],
            "details": {
                # Store energy in Hartree as per plan
                "reference_energy_hartree": dissociation_energy
            }
        }
        reactions.append(reaction_dict)

    print(f"  Generated {len(nacl_energies)} NaCl geometries and reactions.")

    # 3. Prepare data for YAML output
    data_for_yaml = {
        "geometries": geometries,
        "reactions": reactions
    }

    # 4. Write to YAML file
    print(f"Writing database to: {output_yaml}")
    try:
        with open(output_yaml, 'w', encoding='utf-8') as out:
            # Use SafeDumper to avoid aliases and represent data clearly
            # sort_keys=False preserves insertion order (important for reactions list)
            yaml.dump(data_for_yaml, out, Dumper=yaml.SafeDumper, sort_keys=False, allow_unicode=True, indent=2)
        print(f"File {output_yaml} successfully generated.")
    except Exception as e:
        print(f"Error writing YAML file: {e}")


if __name__ == "__main__":
    main()
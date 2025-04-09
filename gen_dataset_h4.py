import yaml
import io

# Simplified ATOMIC_NUMBERS for H
ATOMIC_NUMBERS = {'H': 1}

# XYZ template for generating geometries
xyz_template = """H 0.0 %.3f %.3f
H 0.0 -%.3f %.3f
H 0.0 %.3f -%.3f
H 0.0 -%.3f -%.3f
"""

def parse_geometry_string(xyz_string):
    """
    Parses a multi-line XYZ string and returns:
      1) list of coordinates [(atom, x, y, z), ...]
      2) formula string (e.g., "H4")
      3) total electron count
    """
    lines = [line.strip() for line in xyz_string.strip().split('\n') if line.strip()]
    atomic_coordinates = []
    atom_counts = {}
    total_electrons = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            # Skip lines that don't look like coordinates (e.g., atom count/comment)
            continue
        atom_label = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            raise ValueError(f"Invalid coordinate format in line: {line}")

        atomic_coordinates.append((atom_label, x, y, z))
        atom_counts[atom_label] = atom_counts.get(atom_label, 0) + 1

        Z = ATOMIC_NUMBERS.get(atom_label, 0)
        if Z == 0:
            print(f"Warning: Atom {atom_label} not found in ATOMIC_NUMBERS!")
        total_electrons += Z

    # Generate formula string
    formula_parts = []
    for a_label in sorted(atom_counts.keys()):
        count = atom_counts[a_label]
        if count == 1:
            formula_parts.append(a_label)
        else:
            formula_parts.append(f"{a_label}{count}")
    formula = "".join(formula_parts)

    return atomic_coordinates, formula, total_electrons


def main():
    # --- Initialization ---
    energies = [0.0] * 11  # Placeholder absolute energies in Hartree
    basis_set = "6-31G"
    output_yaml = "h4_database.yaml"

    geometries_list = []
    reactions_list = []

    # --- Main Logic ---
    for i in range(11):  # Generate 11 geometries (i from 0 to 10)
        coord_val = 0.45 + i / 20.0
        # Format the template. Need 8 values for the 8 format specifiers.
        xyz_string_for_geom = xyz_template % tuple([coord_val] * 8)

        # Parse the generated geometry string
        coords, formula, electrons = parse_geometry_string(xyz_string_for_geom)

        # Create geometry ID and dictionary
        geom_id = f"H4_geom_{i}"
        geom_dict = {
            "id": geom_id,
            "formula": formula,
            "atomic_coordinates": [
                f"{atom} {x:.8f} {y:.8f} {z:.8f}" for (atom, x, y, z) in coords
            ],
            "charge": 0,
            "multiplicity": 1,  # Assuming singlet state for H4
            "electrons": electrons,
            "basis_set": basis_set
        }
        geometries_list.append(geom_dict)

        # Create reaction ID and dictionary (representing absolute energy)
        reaction_id = f"{geom_id}_abs_E"
        energy_hartree = energies[i]  # Get the placeholder energy

        reaction_dict = {
            "id": reaction_id,
            "description": f"Absolute energy of {geom_id}",
            "reactants": [],  # Empty for absolute energy representation
            "products": [{"geometry_id": geom_id, "stoichiometry": 1}],
            "details": {"absolute_energy_hartree": energy_hartree}
        }
        reactions_list.append(reaction_dict)

    # Combine lists for YAML output
    data_for_yaml = {
        "geometries": geometries_list,
        "reactions": reactions_list
    }

    # --- Output ---
    try:
        with open(output_yaml, 'w', encoding='utf-8') as out:
            yaml.dump(data_for_yaml, out, sort_keys=False, allow_unicode=True, default_flow_style=None, indent=2)
        print(f"File {output_yaml} successfully generated.")
    except Exception as e:
        print(f"Error writing YAML file: {e}")


if __name__ == "__main__":
    main()
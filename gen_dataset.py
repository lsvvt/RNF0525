import os
import yaml

# Словарь приблизительных порядковых номеров атомов (для вычисления числа электронов)
# Для расширенных систем дополняйте по мере необходимости.
ATOMIC_NUMBERS = {
    'H': 1,  'He': 2, 'Li': 3, 'Be': 4, 'B': 5,  'C': 6,  'N': 7,  'O': 8,
    'F': 9,  'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15,  'S':16,
    'Cl':17, 'Ar':18, 'K':19,  'Ca':20, 'Sc':21, 'Ti':22, 'V':23,  'Cr':24,
    'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32,
    'As':33, 'Se':34, 'Br':35, 'Kr':36, 'Rb':37, 'Sr':38, 'Y':39,  'Zr':40
    # И т.д.
}


def parse_xyz(xyz_path):
    """
    Считывает .xyz-файл и возвращает:
      1) список координат вида [(atom, x, y, z), ...]
      2) текстовую «простую» формулу (по подсчёту атомов, напр. O1F1 -> OF)
      3) число электронов (сумма Z атомов, т.к. заряд=0)
    """
    with open(xyz_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Первая строка обычно число атомов, вторая — комментарий, дальше координаты
    # Но часто встречаются файлы без явного комментария на второй строке.
    # Будем гибкими и просто возьмём все строки после первой для координат.
    try:
        n_atoms = int(lines[0])
    except ValueError:
        raise ValueError(f"Формат .xyz файла {xyz_path} неверный (нет числа атомов в первой строке).")

    coord_lines = lines[2 : 2 + n_atoms]
    atomic_coordinates = []
    atom_counts = {}

    for cl in coord_lines:
        parts = cl.split()
        if len(parts) < 4:
            raise ValueError(f"Неверная строка координат в {xyz_path}: {cl}")
        atom_label = parts[0]
        x, y, z = map(float, parts[1:4])
        atomic_coordinates.append((atom_label, x, y, z))

        # Подсчёт для формулы
        atom_counts[atom_label] = atom_counts.get(atom_label, 0) + 1

    # Простая строка-форсула, например из {'O':1, 'F':1} -> "OF"
    # или из {'S':1, 'Cl':2} -> "SCl2"
    # Для совсем корректной формулы нужно учитывать порядок, но тут упрощённо
    formula_parts = []
    for a_label in sorted(atom_counts.keys()):
        count = atom_counts[a_label]
        if count == 1:
            formula_parts.append(a_label)
        else:
            formula_parts.append(f"{a_label}{count}")
    formula = "".join(formula_parts)

    # Число электронов = сумма Z, т.к. заряд=0
    total_electrons = 0
    for a_label, count in atom_counts.items():
        Z = ATOMIC_NUMBERS.get(a_label, 0)
        if Z == 0:
            print(f"Предупреждение: атом {a_label} не известен в словаре ATOMIC_NUMBERS!")
        total_electrons += Z * count

    return atomic_coordinates, formula, total_electrons


def main():
    res_path = "CTB22/.res"      # исходный файл с данными
    base_folder = "CTB22"        # корневая папка для xyz
    output_yaml = "reactions_database.yaml"

    # Словарь для хранения всех уникальных геометрий:
    # ключ — «путь_без_базы» (например, "OF/c1"), значение — объект-словарь с данными
    geometries_dict = {}

    # Список реакций
    reactions = []

    with open(res_path, 'r') as f:
        lines = f.readlines()

    for line in lines[-12:]:
        line = line.strip()
        # Пропускаем пустые строки и возможные комментарии
        if not line or line.startswith('#'):
            continue

        # Строка формата: "OF/c1 OF/0 4.75"
        parts = line.split()
        if len(parts) != 3:
            print(f"Строка пропущена (не 3 части): {line}")
            continue
        geom1_rel, geom2_rel, energy_str = parts
        energy_kcal_mol = float(energy_str)

        # Проверяем, есть ли уже данные по geom1 в словаре
        if geom1_rel not in geometries_dict:
            xyz_path_1 = os.path.join(base_folder, geom1_rel, "geom.xyz")
            coords_1, formula_1, electrons_1 = parse_xyz(xyz_path_1)
            # Генерируем id. Можно просто заменить «/» на «_»:
            geom1_id = geom1_rel.replace("/", "_")
            geometries_dict[geom1_rel] = {
                "id": geom1_id,
                "formula": formula_1,
                "atomic_coordinates": [
                    f"{atom} {x} {y} {z}" for (atom, x, y, z) in coords_1
                ],
                "charge": 0,
                "multiplicity": 1,
                "electrons": electrons_1,
                "basis_set": "Unknown"  # Если нужно, можно заменить или убрать
            }

        # Аналогично для geom2
        if geom2_rel not in geometries_dict:
            xyz_path_2 = os.path.join(base_folder, geom2_rel, "geom.xyz")
            coords_2, formula_2, electrons_2 = parse_xyz(xyz_path_2)
            geom2_id = geom2_rel.replace("/", "_")
            geometries_dict[geom2_rel] = {
                "id": geom2_id,
                "formula": formula_2,
                "atomic_coordinates": [
                    f"{atom} {x} {y} {z}" for (atom, x, y, z) in coords_2
                ],
                "charge": 0,
                "multiplicity": 1,
                "electrons": electrons_2,
                "basis_set": "Unknown"
            }

        # Создаём запись о «реакции»
        # По условию: «Реакция это 1 минус 2»
        # То есть энергия = E(geom1) - E(geom2). Сохраним это как "reactants=geom1, products=geom2".
        # Или как вам удобнее. Ниже — простой вариант с reactants и products.
        reaction_id = f"{geom1_rel.replace('/', '_')}_minus_{geom2_rel.replace('/', '_')}"
        reaction_dict = {
            "id": reaction_id,
            "description": f"Energy difference: {geom1_rel} minus {geom2_rel}",
            "reactants": [
                {
                    "geometry_id": geometries_dict[geom1_rel]["id"],
                    "stoichiometry": 1
                }
            ],
            "products": [
                {
                    "geometry_id": geometries_dict[geom2_rel]["id"],
                    "stoichiometry": 1
                }
            ],
            "details": {
                "reference_energy_kcal_mol": energy_kcal_mol
            }
        }
        reactions.append(reaction_dict)

    # Формируем структуру для итогового YAML
    # geometries — список уникальных, reactions — список всех «реакций»
    data_for_yaml = {
        "geometries": list(geometries_dict.values()),
        "reactions": reactions
    }

    # Записываем в файл
    with open(output_yaml, 'w', encoding='utf-8') as out:
        yaml.dump(data_for_yaml, out, sort_keys=False, allow_unicode=True)

    print(f"Файл {output_yaml} успешно сгенерирован.")


if __name__ == "__main__":
    main()

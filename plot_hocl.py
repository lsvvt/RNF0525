import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ────────────────────────────────────────────────────────────────
# 1) Мастер‑параметр: масштаб шрифтов, линий, размера фигуры
# ────────────────────────────────────────────────────────────────
SCALE         = 1.2          # 1.0 ‑ экран, 1.5‑2.0 ‑ печать, 0.8 ‑ слайды
BASE_FONTSIZE = 10 * SCALE   # автоматически задаёт почти все шрифты

# ────────────────────────────────────────────────────────────────
# 2) Данные (копируйте или читайте из файла)
# ────────────────────────────────────────────────────────────────
df = pd.read_csv('hocl_new_data.csv')  # путь к csv-файлу с данными о кривых растяжения H4

# ────────────────────────────────────────────────────────────────
# 3) Классификация функционалов (та же, что и в предыдущем графике)
# ────────────────────────────────────────────────────────────────
category_map = {
    'LDA': 'LDA',
    'PBE': 'GGA',
    'PBE-2X': 'Hybrid GGA',
    'PBE0': 'Hybrid GGA',
    'B3LYP': 'Hybrid GGA',
    'TPSS': 'meta-GGA',
    'M06-L': 'meta-GGA',
    'SCAN': 'meta-GGA',
    'r2SCAN': 'meta-GGA',
    'SCAN0': 'Hybrid meta-GGA',
    'M05-2X': 'Hybrid meta-GGA',
    'M06-2X': 'Hybrid meta-GGA',
    'piM06-2X': 'Hybrid meta-GGA',
    'piM06-2X-DL': 'Hybrid meta-GGA',
    'DM21': 'Hybrid meta-GGA',
    'NN-PBE': 'Machine Learning',
}
df["category"] = df["xc"].map(category_map)

# Цвета — те же, что и раньше
color_map = {
    'LDA':              '#1b9e77',
    'GGA':              '#d95f02',
    'meta-GGA':         '#7570b3',
    'Hybrid GGA':       '#e7298a',
    'Hybrid meta-GGA':  '#66a61e',
    'Machine Learning': '#e6ab02',
}

# ────────────────────────────────────────────────────────────────
# 4) Общий стиль matplotlib «под масштаб»
# ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE * 1.4,
    "axes.labelsize": BASE_FONTSIZE * 1.2,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
    "lines.linewidth": 1.5 * SCALE,
})

# ────────────────────────────────────────────────────────────────
# 5) Построение кривых E(r) для каждого функционала
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10 * SCALE, 6 * SCALE))

for xc, g in df.groupby("xc"):
    if xc in list(category_map.keys()):
    # if xc in ["NN-PBE", "PBE0", "M06-2X", "PBE", "PBE-2X"]:
        g = g.sort_values("r")
        cat = g["category"].iloc[0]
        ax.plot(g["r"], g["e"],
                label=xc,
                color=color_map[cat],
                marker='o',
                markersize=3 * SCALE,
                alpha=0.9)

# оси, подписи
ax.set_xlabel("r  (Å)")
ax.set_ylabel("Energy  /  E$_\mathrm{el}$ (Hartree)")
ax.set_title("H$_4$ stretch: E(r) for different XC functionals")

# легенду выводим сбоку, чтобы не закрывать кривые
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)

plt.tight_layout()
plt.show()

plt.savefig("figure3.pdf")
import pandas as pd
import matplotlib.pyplot as plt

category_map = {
    'SVWN5': 'LDA',
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
    'DM21': 'Machine Learning',
    'NN-PBE': 'Machine Learning',
}

color_map = {
    'LDA':              '#1b9e77',
    'GGA':              '#d95f02',
    'meta-GGA':         '#7570b3',
    'Hybrid GGA':       '#e7298a',
    'Hybrid meta-GGA':  '#66a61e',
    'Machine Learning': '#e6ab02',
}

# 1) Параметры
SCALE         = 1.3
BASE_FONTSIZE = 10 * SCALE
plt.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE * 1.2,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE * 0.9,
    "ytick.labelsize": BASE_FONTSIZE * 0.9,
    "lines.linewidth": 2 * SCALE,
    "figure.figsize": (14, 10),
})

# 2) Данные
df = pd.read_csv('h4_new_data.csv')
ref = (df.query("xc=='FCI / SUMR-CCSD'")
         .sort_values("r")["e"]
         .to_numpy()[:-1])

# 3) Категории и цветаcats
line_styles  = ['-', '--', '-.', ':']  # разные стили
markers      = ['o', 's', 'D', '^']    # разные маркеры
df["category"] = df["xc"].map(category_map)

# 4) Сетап subplots
cats = df["category"].unique()
n = len(cats)
cols = 2
rows = (n) // cols
fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
axes = axes.flatten()

# 5) Рисуем по категориям
for ax, cat in zip(axes, cats):
    sub = df[df["category"]==cat]
    for i,(xc, g) in enumerate(sub.groupby("xc")):
        if xc != "SVWN5":
            g = g.sort_values("r")
            ax.plot(g["r"],
                    g["e"].to_numpy() - ref,
                    label=xc,
                    color=color_map[cat],
                    linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)],
                    markersize=4 * SCALE,
                    alpha=0.9)
    ax.set_title(cat)
    ax.legend(frameon=False, ncol=2, fontsize=BASE_FONTSIZE*0.8)
    ax.grid(True)

# Общие подписи
fig.text(0.5, 0.04, "r (Å)", ha='center', va='center', fontsize=BASE_FONTSIZE)
fig.text(0.06, 0.5, "ΔE (Hartree)", ha='center', va='center',
         rotation='vertical', fontsize=BASE_FONTSIZE)
# fig.suptitle("H₄ stretch: ΔE(r) по категориям функционалов", y=0.95)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()

plt.savefig("figure2v2.pdf")

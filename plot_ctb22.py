import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ------------- USER‑TUNABLE PARAMETERS ------------------------------------
SCALE = 1.0      # master scale factor: 1.0 for screen, 1.5–2.0 for print, <1 for slides
BASE_FONTSIZE = 15 * SCALE   # rescaled base font size (ticks, labels, etc.)

# ------------- DATA -------------------------------------------------------
data_csv = """basis_set,xc,mae,maxe
cc-pVQZ,NN-PBE,6.315652015596204,9.73456122793019
cc-pVQZ,SVWN5,4.030642255373261,7.076848691286305
cc-pVQZ,M06-L,3.3220856139280808,9.002006989215861
cc-pVQZ,PBE,5.9975994485063495,8.944927123897433
cc-pVQZ,TPSS,3.9609019449877767,5.842977621868304
cc-pVQZ,r2SCAN,5.17148860390228,11.560851858848455
cc-pVQZ,SCAN,5.589795845096521,11.914026343514397
cc-pVQZ,SCAN0,3.430429292201737,8.465531609005776
cc-pVQZ,PBE-2X,2.027060287974758,4.553137039081886
cc-pVQZ,PBE0,2.5646735752679928,5.227303707620305
cc-pVQZ,M06-2X,2.6534209656054633,4.608177813898347
cc-pVQZ,M05-2X,2.059242277837153,4.322903741076626
cc-pVQZ,B3LYP,4.945917022779766,9.538281747513029
cc-pVQZ,DM21,2.8810386050240506,8.168671559570758
cc-pVQZ,piM06-2X-DL,2.240382760786605,4.680221187497974
cc-pVQZ,piM06-2X,2.344615109225035,4.815085377060576
"""

df = pd.read_csv(StringIO(data_csv))

# ------------- CLASSIFICATION --------------------------------------------
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

df["category"] = df["xc"].map(category_map)

category_order = [
    'LDA',
    'GGA',
    'meta-GGA',
    'Hybrid GGA',
    'Hybrid meta-GGA',
    'Machine Learning',
]

# enforce order
df["category"] = pd.Categorical(df["category"], categories=category_order, ordered=True)
df_sorted = df.sort_values(["category", "mae"], kind="mergesort").reset_index(drop=True)

# ------------- COLORS -----------------------------------------------------
color_map = {
    'LDA':              '#1b9e77',
    'GGA':              '#d95f02',
    'meta-GGA':         '#7570b3',
    'Hybrid GGA':       '#e7298a',
    'Hybrid meta-GGA':  '#66a61e',
    'Machine Learning': '#e6ab02',
}

# ------------- MATPLOTLIB GLOBAL STYLE ------------------------------------
plt.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE * 1.4,
    "axes.labelsize": BASE_FONTSIZE * 1.2,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
})

# ------------- PLOT -------------------------------------------------------
fig_width = 14 * SCALE  # inches
fig_height = 6 * SCALE
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

bar_positions = range(len(df_sorted))
bar_colors = [color_map[c] for c in df_sorted["category"]]

ax.bar(
    bar_positions,
    df_sorted["mae"],
    width=0.7 * SCALE,
    color=bar_colors,
    edgecolor="black"
)

ax.scatter(
    bar_positions,
    df_sorted["maxe"],
    marker="v",
    s=60 * SCALE,
    facecolors="none",
    edgecolors=bar_colors,
    linewidths=1.5 * SCALE
)

ax.set_ylabel('$\Delta E$, ккал·моль$^{-1}$', fontsize=BASE_FONTSIZE*1.5)
ax.set_xticks(bar_positions)
ax.set_xticklabels(df_sorted["xc"], rotation=45, ha="right")
ax.set_ylim(bottom=0)

# top axis with class labels
group_centres = {}
for cat in category_order:
    idx = df_sorted.index[df_sorted["category"] == cat]
    if len(idx):
        group_centres[cat] = (idx.min() + idx.max()) / 2

secax = ax.secondary_xaxis('top')
secax.set_xticks(list(group_centres.values()))
secax.set_xticklabels(group_centres.keys(), fontweight='bold')

for tick in secax.get_xticklabels():
    tick.set_color(color_map[tick.get_text()])

secax.spines['top'].set_visible(False)

# vertical separators
prev_cat = df_sorted.loc[0, "category"]
for pos, cat in zip(bar_positions, df_sorted["category"]):
    if cat != prev_cat:
        ax.axvline(pos - 0.5, color="grey", linestyle="--", linewidth=0.8)
    prev_cat = cat

# legend
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker='s', markersize=10*SCALE, markerfacecolor='white',
           markeredgecolor='black', linestyle='None', label='MAE'),
    Line2D([], [], marker='v', markersize=10*SCALE, markerfacecolor='white',
           markeredgecolor='black', linestyle='None', label='MAXE'),
]
ax.legend(handles=handles, frameon=False, loc='upper right')

#ax.set_title("cc‑pVQZ: Mean (bars) vs Maximum (triangles) Absolute Errors by Functional Class")

plt.tight_layout()
plt.show()

plt.savefig("figure1.pdf")
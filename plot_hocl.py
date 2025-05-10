import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import seaborn as sns


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

# 1) Загрузка данных
df = pd.read_csv('hocl_new_data.csv')  # путь к csv-файлу с данными о кривых растяжения H4

# df['r'] *= 2
# Опорная кривая (reference energies)
ref = (df.query("xc=='CCSD(T)'")
         .sort_values('r')['e']
         .to_numpy()[:])

# 2) Настройка палитры и стилей
funcs = sorted(df['xc'].unique())
palette = sns.color_palette('tab20', len(funcs))
color_func = dict(zip(funcs, palette))

dash_patterns = [(None, None), (5,2), (1,1), (3,1,1,1), (7,1,1,1,1,1)]
markers      = ['o', 's', 'D', '^']    # разные маркеры

fm = {}
for i, xc in enumerate(funcs):
    fm[xc] = markers[i % len(markers)]

SCALE = 1.3
BASE = 15
plt.rcParams.update({
    'font.size': BASE,
    'axes.titlesize': BASE * 1.2,
    'lines.linewidth': 2.5 * SCALE,
})

# 3) Категории и раскладка 3x2

df['category'] = df['xc'].map(category_map)
plot_cats = ['Hybrid GGA', 'meta-GGA', 'MAE, MAXE', 'Hybrid meta-GGA', 'GGA', 'Machine Learning']
fig, axes = plt.subplots(3, 2, figsize=(14,10))
axes = axes.flatten()

# 4) Вычислим общие пределы для всех кривых (кроме Minima)
y_vals = []
x_vals = []
for cat in plot_cats:
    if cat == 'MAE, MAXE':
        continue
    sub = df[df['category']==cat]
    for xc, g in sub.groupby('xc'):
        if xc in ('CCSD(T)', 'SVWN5'):
            continue
        g = g.sort_values('r')
        y = g['e'].to_numpy() - ref[:len(g)]
        x = g['r'].to_numpy()
        y_vals.extend(y)
        x_vals.extend(x)
# Задаём глобальные границы
xmin, xmax = min(x_vals), max(x_vals)
ymin, ymax = min(y_vals), max(y_vals)
ymin = -10

# 5) Построение
for ax, cat in zip(axes, plot_cats):
    if cat != 'MAE, MAXE':
        sub = df[df['category']==cat]
        for i, (xc, g) in enumerate(sub.groupby('xc')):
            if xc in ('CCSD(T)', 'SVWN5'):
                continue
            g = g.sort_values('r')
            ax.plot(
                g['r'], g['e']-ref[:len(g)],
                label=xc,
                color=color_func[xc],
                dashes=dash_patterns[i % len(dash_patterns)],
                marker=fm[xc],
                markersize=6*SCALE,
                markerfacecolor=color_func[xc],
                markeredgecolor='black',
                markeredgewidth=0.8,
            )
        ax.set_title(cat)
        ax.legend(frameon=False, ncol=2, fontsize=BASE)
        ax.grid(True)
        # Единая шкала для этих графиков
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    # else:
    #     ...

plt.tight_layout()
plt.show()

plt.savefig("figure3v2.pdf")
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
df = pd.read_csv('h4_new_data.csv')

df['r'] *= 2
# Опорная кривая (reference energies)
ref = (df.query("xc=='FCI'")
         .sort_values('r')['e']
         .to_numpy()[:-1])

# 2) Настройка палитры и стилей
funcs = sorted(df['xc'].unique())
palette = sns.color_palette('tab20', len(funcs))
color_func = dict(zip(funcs, palette))

print(color_func)

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
plot_cats = ['Hybrid GGA', 'meta-GGA', 'Minima', 'Hybrid meta-GGA', 'GGA', 'Machine Learning']
fig, axes = plt.subplots(3, 2, figsize=(14,10))
axes = axes.flatten()

# 4) Вычислим общие пределы для всех кривых (кроме Minima)
y_vals = []
x_vals = []
for cat in plot_cats:
    if cat == 'Minima':
        continue
    sub = df[df['category']==cat]
    for xc, g in sub.groupby('xc'):
        if xc in ('FCI', 'SVWN5'):
            continue
        g = g.sort_values('r')
        y = g['e'].to_numpy() - ref
        x = g['r'].to_numpy()
        y_vals.extend(y)
        x_vals.extend(x)
# Задаём глобальные границы
xmin, xmax = min(x_vals), max(x_vals)
ymin, ymax = min(y_vals), max(y_vals)

# 5) Построение
for ax, cat in zip(axes, plot_cats):
    if cat != 'Minima':
        sub = df[df['category']==cat]
        for i, (xc, g) in enumerate(sub.groupby('xc')):
            if xc in ('FCI', 'SVWN5'):
                continue
            g = g.sort_values('r')
            ax.plot(
                g['r'], g['e']-ref,
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
    else:
        # Интерполированные минимумы
        methods = []
        r_min = []
        e_min = []
        for xc, g in df.groupby('xc'):
            if xc in ('SVWN5', "PBE_ref", "B3LYP_ref", "O3P86"):
                continue
            grp = g.sort_values('r')
            r = grp['r'].values
            e = grp['e'].values
            if xc == "DM21":
                r = r[:-1]; e = e[:-1]
            spline = UnivariateSpline(r, e, k=3, s=0)
            opt = minimize_scalar(spline, bounds=(r.min(), r.max()), method='bounded')
            methods.append(xc)
            r_min.append(opt.x)
            e_min.append(opt.fun)

        # now plot each minimum with the same marker+color as in the curves
        for xc, rv, ev in zip(methods, r_min, e_min):
            idx = funcs.index(xc)                   # find its index in your sorted list
            mk = fm[xc]   # pick the same marker
            col = color_func[xc]                    # pick the same color
            ax.scatter(
                rv, ev,
                marker=mk,
                s=10**2,              # size comparable to curve markers
                facecolor=col,
                edgecolor='black',
                linewidth=0.8,
                label=xc
            )

            offsets = {
                'FCI' : (5, 5),
                'DM21':     ( 8, -5),
                'M06-2X':   ( 8, -10),
                'CCSD':   (-25, -22),
                'piM06-2X-DL':( 5, 10),
                'TPSS': ( 8, -5),
                # 'B3LYP': ( 8, -5),
                'NN-PBE': (-40, -20),
                'r2SCAN': ( 8, -5),
            }
            if xc in offsets.keys():
                ax.annotate(
                    xc,
                    (rv, ev),
                    xytext=offsets.get(xc, (3,3)),
                    textcoords='offset points',
                    fontsize=BASE * 1
                )

        ax.set_title('Interpolated Minima')
        ax.grid(True)
        # ax.legend(frameon=False, ncol=2, fontsize=BASE*0.8)


# Общие подписи
# fig.text(0.5, 0.04, 'r (Å)', ha='center', fontsize=BASE)
# fig.text(0.06, 0.5, 'ΔE (Hartree)', va='center', rotation='vertical', fontsize=BASE)
plt.tight_layout()
plt.show()

plt.savefig("figure2.pdf")
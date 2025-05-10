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
color_func = {'B3LYP': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 'B3LYP_ref': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274), 'CCSD': (1.0, 0.4980392156862745, 0.054901960784313725), 'DM21': (1.0, 0.7333333333333333, 0.47058823529411764), 'FCI': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 'M05-2X': (0.596078431372549, 0.8745098039215686, 0.5411764705882353), 'M06-2X': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), 'M06-L': (1.0, 0.596078431372549, 0.5882352941176471), 'NN-PBE': (0.5803921568627451, 0.403921568627451, 0.7411764705882353), 'O3P86': (0.7725490196078432, 0.6901960784313725, 0.8352941176470589), 'PBE': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), 'PBE-2X': (0.7686274509803922, 0.611764705882353, 0.5803921568627451), 'PBE0': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), 'PBE_ref': (0.9686274509803922, 0.7137254901960784, 0.8235294117647058), 'SCAN': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), 'SCAN0': (0.7803921568627451, 0.7803921568627451, 0.7803921568627451), 'SVWN5': (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), 'TPSS': (0.8588235294117647, 0.8588235294117647, 0.5529411764705883), 'piM06-2X': (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), 'piM06-2X-DL': (0.6196078431372549, 0.8549019607843137, 0.8980392156862745), 'r2SCAN': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)}

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
                # dashes=dash_patterns[i % len(dash_patterns)],
                # marker=fm[xc],
                # markersize=6*SCALE,
                # markerfacecolor=color_func[xc],
                # markeredgecolor='black',
                # markeredgewidth=0.8,
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
        col = []
        xcz = []
        for xc, g in df.groupby('xc'):
            if xc in ('CCSD(T)', "PBE_ref", "B3LYP_ref", "O3P86"):
                continue
            grp = g.sort_values('r')
            r = grp['r'].values
            e = grp['e'].values
            spline = UnivariateSpline(r, e, k=3, s=0)
            opt = minimize_scalar(spline, bounds=(r.min(), r.max()), method='bounded')
            methods.append(xc)
            r_min.append(opt.x)
            e_min.append(sum(abs(g['e']-ref[:len(g)])))
            col.append(color_func[xc])
            xcz.append(xc)

        xcz = [x for _,x in sorted(zip(e_min,xcz))]
        col = [x for _,x in sorted(zip(e_min,col))]

        e_min.sort()

        ax.bar(
            range(len(e_min)),
            e_min,
            width=0.7,
            color=col,
            edgecolor='black'
        )
        ax.set_xticklabels(xcz, rotation=45, ha="right")

plt.tight_layout()
plt.show()

plt.savefig("figure3v2.pdf")

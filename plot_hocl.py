import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


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

xc_sh = {
    'LDA':              'LDA',
    'GGA':              'GGA',
    'meta-GGA':         'mGGA',
    'Hybrid GGA':       'h-GGA',
    'Hybrid meta-GGA':  'h-mGGA',
    'Machine Learning': 'ML',
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
plot_cats = ['Hybrid GGA', 'meta-GGA', 'Hybrid meta-GGA', 'MAE', 'GGA', 'Machine Learning']
fig, axes = plt.subplots(3, 2, figsize=(14,10))
axes = axes.flatten()

# 4) Вычислим общие пределы для всех кривых (кроме Minima)
y_vals = []
x_vals = []
for cat in plot_cats:
    if cat == 'MAE':
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
    if cat != 'MAE':
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
            if xc in ('CCSD(T)', "NN-PBE"):
                continue
            e = g['e'].values
            methods.append(xc)
            e_min.append(sum(abs(g['e']-ref[:len(g)])) / len(g))
            col.append(color_func[xc])
            xcz.append(xc)

        xcz = [x for _,x in sorted(zip(e_min,xcz))]
        col = [x for _,x in sorted(zip(e_min,col))]

        bar_colors = [color_map[category_map[xc]] for xc in xcz]


        e_min.sort()

        bars = ax.bar(
            range(len(e_min)),
            e_min,
            width=0.7,
            color=bar_colors,
            edgecolor='black'
        )
        # remove the old tick labels
        ax.set_xticks([])
        # now label each bar in the middle
        max_height = max(bar.get_height() for bar in bars)
        threshold  = max_height * 0.7  # e.g. 8% of the tallest bar

        for bar, label in zip(bars, xcz):
            h = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            
            if h >= threshold:
                # big enough → inside, centered
                y      = h / 2
                va     = 'center'
                color  = 'white'
            else:
                # too small → above the bar
                y      = h + max_height * 0.02  # 2% of max height as padding
                va     = 'bottom'
                color  = 'black'
            
            ax.text(
                x, y,
                label,
                ha='center', va=va,
                rotation=90,
                fontsize=BASE * 1,
                color=color
            )

        unique_cats = []
        for xc in xcz:
            cat = category_map[xc]
            if cat not in unique_cats:
                unique_cats.append(cat)
        print(unique_cats)
        # make one colored patch per category

        legend_patches = [
            mpatches.Patch(color=color_map[cat], label=xc_sh[cat])
            for cat in unique_cats
        ]
        ax.legend(
            handles=legend_patches,
            # title='XC category',
            frameon=False,
            loc='upper left',
            fontsize=BASE * 0.9,
            ncol=2
        )
        ax.set_title("MAE")
    # if cat == 'MAE':
    #     xlabel, ylabel = 'Exchange–correlation category', 'Mean absolute error (kcal/mol)'
    # elif cat == "Hybrid meta-GGA":
    #     xlabel, ylabel = 'Расстояние OH-Cl $r$ (Å)', 'Отклонение, ккал·моль$^{-1}$'

    #     # and now apply them with your BASE size
    #     ax.set_xlabel(xlabel,
    #                 fontsize=BASE*1.1,
    #                 fontweight='medium',
    #                 labelpad=8)
    #     ax.set_ylabel(ylabel,
    #                 fontsize=BASE*1.1,
    #                 fontweight='medium',
    #                 labelpad=8)

fig.subplots_adjust(
    # left=0.158,   # give room on the left for the y-label
    # bottom=0.05,  # give room at the bottom for the x-label
    # right=0.9,
    top=0.96
)

# add one big x-label centered under all subplots
fig.text(
    0.5,             # x = 50% of figure width
    0.02,            # y = 2% of figure height (just above the bottom)
    'Расстояние OH-Cl, Å',  # or whatever text you like
    ha='center', va='center',
    fontsize=BASE*1.5,
    # fontweight='semibold'
)

# add one big y-label centered beside all subplots
fig.text(
    0.015,            # x = 2% of figure width (just right of the left edge)
    0.5,             # y = 50% of figure height
    '$\Delta E$, ккал·моль$^{-1}$',  # or “Mean absolute error (kcal/mol)”
    ha='center', va='center',
    rotation='vertical',
    fontsize=BASE*1.5,
    # fontweight='semibold'
)

plt.tight_layout()

fig.subplots_adjust(
    # left=0.158,   # give room on the left for the y-label
    bottom=0.085,  # give room at the bottom for the x-label
    # right=0.9,
    # top=0.96
)

plt.show()

plt.savefig("figure3.pdf")

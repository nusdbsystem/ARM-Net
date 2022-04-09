# script for plotting training with partial fields (f1 & loss for both HDFS & BGL)

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.lines as mlines

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# perc0.2

f1 = [
    [0.8641, 0.9554, 0.9588, 0.9645, 0.9607, 0.9619],
    [0.5827, 0.6268, 0.6301, 0.6745, 0.6990, 0.7605]
]

loss = [
    [0.0730, 0.0332, 0.0422, 0.0282, 0.0493, 0.0251],
    [0.2038, 0.1871, 0.1759, 0.1581, 0.1363, 0.1123]
]

titles = ['HDFS', 'BGL']
colors = ['C1', 'C6', 'C2', 'C9', 'C4', 'C3']
hatches = ['---', '///', '\\\\\\', '...', 'xx', 'xxx']

width = 0.5
figs, axes = plt.subplots(nrows=1, ncols=2)
sub_figs = [axes[0], axes[1]]

yalim = [
    (0.82, 0.98), (0.5, 0.8)
]
yblim = [
    (0.01, 0.08), (0.08, 0.22)
]

for i in range(2):

    title = titles[i]
    sub_figs[i].set_title(title)
    ax2 = sub_figs[i].twinx()

    if i == 0:
        sub_figs[i].set_ylabel('F1')
    if i == 1:
        ax2.set_ylabel('Logloss')

    for j in range(6):
        sub_figs[i].bar(j, f1[i][j], width=width, edgecolor=colors[j], facecolor='white', hatch=hatches[j])
    ax2.plot(np.arange(6), loss[i], color='black', marker='o')

    sub_figs[i].set_xticks([])

    sub_figs[i].set_ylim(yalim[i])
    ax2.set_ylim(yblim[i])
    if i == 0:
        ax2.set_yticks(np.arange(0, 0.081, 0.02))

handles = [mpatches.Patch(facecolor='white', edgecolor=colors[i], hatch=hatches[i]) for i in range(6)]
fig_leg = plt.figlegend(
    labels=['Avg-1', 'Avg-2', 'Avg-3', 'Avg-4', 'Avg-5', 'All fields'],
    loc='lower center',
    handles=handles,
    ncol=6,
    handletextpad=0.3,
    columnspacing=1,
    handlelength=1.0)
fig_leg.get_frame().set_edgecolor('black')

fig_leg2 = plt.figlegend(labels=['F1 (higher is better)', 'Logloss (lower is better)'],
                         loc=(0.16, 0.16),
                         handles=[
                             mpatches.Patch(edgecolor='black', facecolor='white', hatch=''),
                             mlines.Line2D([], [], color='black', marker='o')
                         ],
                         ncol=2,
                         )
fig_leg2.get_frame().set_edgecolor('white')

figs.tight_layout(rect=[0, 0.25, 1, 0.99], h_pad=0, w_pad=1)
figs.set_figheight(2.6)
figs.set_figwidth(6)

plt.savefig('./fig/partial_perc0.2.pdf', dpi=900, bbox_inches='tight', pad_inches=0)

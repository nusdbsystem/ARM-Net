# script for plotting interpretation results (both global and local) for hdfs
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


global_neuron = [
    [-0.5540, -0.5309,  0.1801,  0.2675, -0.5165, -0.5238],
    [0.1194,  0.4117, -0.2177, -0.0597,  0.3382, -0.2098],
    [0.1318, -0.0180,  0.0798,  0.4431,  0.4836, -0.1046]
]
global_neuron = np.abs(np.array(global_neuron))

global_average = np.array([0.2719, 0.2682, 0.2209, 0.2430, 0.3301, 0.2729])

local_neuron = [
    [-0.0000e+00, -4.9947e-03, -9.1378e-02,  2.1914e-01,  3.6395e-03, 2.8912e-03],
    [-5.2395e-02, -2.4016e-03,  0.0000e+00, -3.1123e-03, -1.3144e-01, -4.2770e-03],
    [-5.5858e-02, -1.8492e-03,  4.5106e-02, -2.7576e-02, -8.3511e-02, -0.0000e+00]
]
local_neuron = np.abs(np.array(local_neuron))

local_average = np.array([0.0464, 0.0395, 0.0301, 0.0467, 0.0639, 0.0468])
local_average = (local_average-0.025)

feature_num = 6
feature_list = ['hour', 'minute', 'second', 'pid', 'level', 'component']
width = 0.35
color = 'k'
facecolors = ['#1ac9e6', '#8080ff']
# facecolors = ['#1de4bd', '#1ac9e6', '#8080ff']
hatch = ''

figs, axes = plt.subplots(nrows=4, ncols=2)

for row in range(4):
    for col in range(2):
        subfig = axes[row][col]
        if 0 == col:
            if row < 3:
                if 0 == row:
                    subfig.set_title('Global Field Importance')
                subfig.set_ylabel('Neuron-{}'.format(row+1))
                for fea in range(feature_num):
                    subfig.bar(fea, global_neuron[row][fea], width=width, edgecolor=color, facecolor=facecolors[0],
                               hatch=hatch)
                    subfig.set_ylim((0, 0.6))
                    subfig.set_xticks([])
            else:
                subfig.set_ylabel('Average')
                for fea in range(feature_num):
                    subfig.bar(fea, global_average[fea], width=width, edgecolor=color, facecolor=facecolors[1],
                               hatch=hatch)
                    subfig.set_ylim((0.18, 0.35))
                    subfig.set_xticks(range(feature_num))
                    subfig.set_xticklabels(feature_list, rotation=45)

        else:
            if row < 3:
                if 0 == row:
                    subfig.set_title('Local Field Importance')
                subfig.set_ylabel('Neuron-{}\n'.format(row+1))
                for fea in range(feature_num):
                    subfig.bar(fea, local_neuron[row][fea], width=width, edgecolor=color,
                               facecolor=facecolors[0], hatch=hatch)
                    subfig.set_ylim((0, 0.3))
                    subfig.set_xticks([])
            else:
                subfig.set_ylabel('Average')
                for fea in range(feature_num):
                    subfig.bar(fea, local_average[fea], width=width, edgecolor=color, facecolor=facecolors[1],
                               hatch=hatch)
                    subfig.set_ylim((0, 0.05))
                    subfig.set_xticks(range(feature_num))
                    subfig.set_xticklabels(feature_list, rotation=45)

figs.tight_layout(rect=[0, 0, 1, 1], h_pad=0.5, w_pad=-2)
figs.set_figheight(5)
figs.set_figwidth(10)

plt.savefig('fig/hdfs_interpretation.pdf', dpi=900, bbox_inches='tight', pad_inches=0)

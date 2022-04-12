# script for plotting interpretation results (both global and local) for hdfs
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


global_neuron = [
    [0.6004,  0.3050, -0.5111, -0.3912,  0.2041,  0.4930,  0.2257,  0.6914],
    [0.1681,  0.6374, -0.3690, -0.2411,  0.0231, -0.2288, -0.4534,  0.3655],
    [0.1493,  0.1426, -0.4507,  0.4243, -0.5284,  0.3299,  0.8326,  0.6932]
]
global_neuron = np.abs(np.array(global_neuron))

global_average = np.array([0.3585, 0.3334, 0.3923, 0.3089, 0.3005, 0.3863, 0.3868, 0.4308])

local_neuron = [
    [-0.0000e+00,  0.0000e+00,  0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,  6.9621e-05,  6.6853e-01],
    [3.7323e-03,  2.9838e-04, -6.0607e-03, -2.9756e-02,  3.2859e-05, 1.5537e-05,  1.9362e-01,  3.2316e-02],
    [4.0600e-05, -1.0818e-02,  3.9362e-03,  3.9018e-02,  0.0000e+00, 3.0049e-01, -2.3476e-04, -0.0000e+00]
]
local_neuron = np.abs(np.array(local_neuron))

local_average = np.array([0.0101, 0.0252, 0.0234, 0.0227, 0.0597, 0.0922, 0.0768, 0.1481])
local_average = (local_average)/4.

feature_num = 8
feature_list = ['hour', 'minute', 'second', 'millisecond', 'node', 'type', 'component', 'level']
width = 0.4
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
                    subfig.set_ylim((0, 1))
                    subfig.set_xticks([])
            else:
                subfig.set_ylabel('Average')
                for fea in range(feature_num):
                    subfig.bar(fea, global_average[fea], width=width, edgecolor=color, facecolor=facecolors[1],
                               hatch=hatch)
                    subfig.set_ylim((0.28, 0.5))
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
                    subfig.set_ylim((0, 0.8))
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
figs.set_figwidth(12)

plt.savefig('fig/bgl_interpretation.pdf', dpi=900, bbox_inches='tight', pad_inches=0)

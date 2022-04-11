# script for plotting training with different ratios of training:testing (f1, both HDFS & BGL)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_learning_curve(lines, shapes, colors, labels, markers,
                        save_path, title='', logy=False,
                        ms=1., linewidth=1.,
                        xlabel=None, ylabel=None,
                        ylim=None, yticks=None, xlim=None, xticks=None, xtick_label=None,
                        legend_font=6., legend_loc='upper right', x_tick_font=14, y_tick_font=14, label_font=14):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel, fontsize=label_font); plt.ylabel(ylabel, fontsize=label_font)
    if xlim or xticks: plt.xticks(xticks, xtick_label, fontsize=x_tick_font); plt.xlim(xlim)
    if ylim or yticks: plt.yticks(yticks, fontsize=y_tick_font); plt.ylim(ylim);
    # plt.grid(linestyle='dotted')

    for idx, line in enumerate(lines):
        if not logy:
            plt.plot(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                 linewidth=linewidth, marker=markers[idx], ms=ms)
        else:
            plt.semilogy(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                 linewidth=linewidth, marker=markers[idx], ms=ms)

    plt.legend(loc=legend_loc, fontsize=legend_font)
    plt.savefig(save_path, dpi=900, bbox_inches='tight', pad_inches=0)


# HDFS
line_lr = np.array([0.9342, 0.9302, 0.9189, 0.9312, 0.8885, 0.9736, 0.9928, 0.9731, 0.9992])
line_deeplog = np.array([0.8778, 0.8737, 0.8198, 0.7512, 0.7236, 0.8565, 0.8747, 0.8222, 0.8132])
line_loganomaly = np.array([0.8933, 0.5956, 0.5459, 0.6462, 0.5596, 0.9019, 0.8690, 0.8251, 0.8108])
line_robustlog = np.array([0.4227, 0.8713, 0.6539, 0.8820, 0.8664, 0.9549, 0.9881, 0.9745, 0.9524])
line_tabularlog = np.array([0.9526, 0.9530, 0.9731, 0.9656, 0.9622, 0.9849, 0.9804, 0.9812, 0.9976])

lines = [line_lr, line_deeplog, line_loganomaly, line_robustlog, line_tabularlog]
shapes = [':', '-.', ':', '--', '-']
markers = ['^', 'o', 'D', 'p', 's']
labels = ['LR', 'DeepLog', 'LogAnomaly', 'RobustLog', 'RTLog']
colors = ['#9a0eea', '#ff00ff', '#06c2ac', '#15b01a', '#0343df']
xlim = [-0.2, 8.2]
ylim = [0.5, 1.02]
xticks = range(9)
xtick_label = ['1:9', '2:8', '3:7', '4:6', '5:5', '6:4', '7:3', '8:2', '9:1']

plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                    save_path='fig/hdfs_f1_vs_train_size.pdf',
                    title='', logy=False, ms=12, linewidth=2.8,
                    xlabel=r'Different Ratios of Training:Testing', ylabel='F1 Score', ylim=ylim, yticks=None,
                    xlim=xlim, xticks=xticks, xtick_label=xtick_label, legend_font=14, legend_loc='lower right',
                    x_tick_font=18, y_tick_font=16, label_font=20)


# BGL
line_lr = np.array([0.2096, 0.2031, 0.2015, 0.2001, 0.2294, 0.2325, 0.3745, 0.2682, 0.2200])
line_deeplog = np.array([0.2127, 0.2148, 0.2052, 0.2037, 0.2230, 0.2319, 0.3440, 0.2689, 0.2228])
line_loganomaly = np.array([0.1891, 0.1792, 0.1798, 0.1846, 0.1778, 0.2440, 0.4524, 0.4854, 0.6399])
line_robustlog = np.array([0.5374, 0.5668, 0.5508, 0.5724, 0.6599, 0.5845, 0.7923, 0.9364, 0.9583])
line_tabularlog = np.array([0.7182, 0.7266, 0.7766, 0.7313, 0.7495, 0.7648, 0.9334, 0.9644, 0.9745])

lines = [line_lr, line_deeplog, line_loganomaly, line_robustlog, line_tabularlog]
shapes = [':', '-.', ':', '--', '-']
markers = ['^', 'o', 'D', 'p', 's']
labels = ['LR', 'DeepLog', 'LogAnomaly', 'RobustLog', 'RTLog']
colors = ['#9a0eea', '#ff00ff', '#06c2ac', '#15b01a', '#0343df']
xlim = [-0.2, 8.2]
ylim = [0.05, 1.02]
xticks = range(9)
xtick_label = ['1:9', '2:8', '3:7', '4:6', '5:5', '6:4', '7:3', '8:2', '9:1']

plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                    save_path='fig/bgl_f1_vs_train_size.pdf',
                    title='', logy=False, ms=12, linewidth=2.8,
                    xlabel=r'Different Ratios of Training:Testing', ylabel='F1 Score', ylim=ylim, yticks=None,
                    xlim=xlim, xticks=xticks, xtick_label=xtick_label, legend_font=14, legend_loc='best',
                    x_tick_font=18, y_tick_font=16, label_font=20)

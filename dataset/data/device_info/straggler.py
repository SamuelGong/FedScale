import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

def my_horizontal_stacked(y, datas, yticklabels,
               legend_labels, xlabel, ylabel, path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)
    colors = ['#f1a340','#fee0b6','#d8daeb','#998ec3','#542788']
    fontsize = 8

    current_left = np.zeros(len(yticklabels))
    for idx, col in enumerate(datas):

        ax.barh(y, col, left=current_left,
               color=colors[idx], label=legend_labels[idx])
        current_left += col
        if idx == len(datas) - 1:
            for yi, li in zip(y, current_left):
                ax.text(li * 1.02, yi - 0.2, "{:.0f}".format(li),
                        fontsize=fontsize)

    ax.axvline(x=current_left[0], ymin=0, ymax=1,
               color='#e0e0e0', dashes=[3, 3, 3, 3])
    ax.text(current_left[0] * 1.02, 1,
            "Round Completion", rotation=270, fontsize=fontsize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
    ax.set_xlim(0, ax.get_xlim()[1]*1.1)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc=(0.4, 0.7), fontsize=fontsize, frameon=False)
    plt.savefig(path, bbox_inches='tight')


def run():
    y = np.arange(5)
    datas = [[350, 60, 100, 100, 40],
             [66, 199, 51, 24, 29]]
    yticklabels = [str(i+1) for i in y]
    yticklabels.reverse()
    legend_labels = ["Comp.", "Comm."]
    xlabel = "Time Breakdown of a Round (s)"
    ylabel = "Client ID"
    path = "straggler"
    my_horizontal_stacked(y, datas, yticklabels, legend_labels, xlabel, ylabel, path)


run()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from matplotlib.transforms import Bbox

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)



def plot_perf_on_axis(ax, models, metric, r, ticks, title, ylabel, xlabel):
    for m in models:

        y = [a[m][metric] for a in r]
        x = list(range(len(y)))
        ax.plot(x, y, linestyle='dashed', linewidth=2,label=m)            

    data = [[a[m][metric] for m in models] for a in r]
    #     ax.boxplot(data, positions = range(len(ticks)))
    ax.plot(x, [sum(row)/len(row) for row in data], marker='o', linestyle='solid', linewidth=1, markersize=2,label="mean",color="black")


    ax.set_title(title)
    ax.grid(True)

    ax.set_yticks(ticks = [.25,.5,.75,1])
    ax.set_ylabel(ylabel)

    new_ticks = [f"[{t}]" for t in ticks]
    ax.set_xticks(ticks=[i for i in range(len(x))])
    ax.set_xticklabels(new_ticks)
    ax.set_xlabel(xlabel)
    ax.legend(prop={'size': 6})

def display_perf_by(results, ticks, models,title='Metric Performance'):
    
    r = results
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False,figsize=(25, 5))
    for i, metric in enumerate(["acc","f1","mAP", "auROC"]):
        plot_perf_on_axis(axs[i], models, metric, r, ticks, title, f'{metric} Scores', 'Head Rotation (degrees)')

        
    # fig.suptitle(title,y=0.98, fontsize=16)
    plt.show()
    
def create_paper_perf_figure(results, num_buckets=4, metric="mAP", title='Metric Performance'):
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False,figsize=(25, 5))
    ticks = results[num_buckets]["xticks"]

    if not os.path.exists(f"visuals/{title}/"):
        os.makedirs(f"visuals/{title}/")

    label="SPEECH"
    r = results[num_buckets][label]["original"]
    plot_perf_on_axis(axs[0], synthesizers, metric, r, ticks, "", f'{metric} Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[0]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Synthesizers Speech Performance.png", bbox_inches=extent)

    plot_perf_on_axis(axs[1], detectors, metric, r, ticks, "", f'{metric} Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[1]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Detectors Speech Performance.png", bbox_inches=extent)

    label="TURN"
    r = results[num_buckets][label]["original"]
    plot_perf_on_axis(axs[2], synthesizers, metric, r, ticks, "", f'{metric} Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[2]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Synthesizers Turn Performance.png", bbox_inches=extent)

    plot_perf_on_axis(axs[3], detectors, metric, r, ticks, "", f'{metric} Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[3]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Detectors Turn Performance.png", bbox_inches=extent)


    fig.suptitle(title,y=0.98, fontsize=16)
    plt.savefig(f"visuals/{title}.png")

    # plt.show()

def plot_imprv_on_axis(ax, models, metric, r0, r1, ticks, title, ylabel, xlabel):
    ys = []
    for m in models:
        y0 = [a[m][metric] for a in r0]
        y1 = [a[m][metric] for a in r1]
        y=[y1[k]-y0[k] for k in range(len(y0))]
        ys.append(y)

        x = list(range(len(y)))
        ax.plot(x, y, marker='o', linestyle='dashed', linewidth=2, markersize=2,label=m)            

    data = [list(x) for x in zip(*ys)]
    # ax.boxplot(data, positions = range(len(ticks)))
    ax.plot(x, [sum(row)/len(row) for row in data], marker='o', linestyle='solid', linewidth=1, markersize=2,label="mean",color="black")
    # ax.plot(x, [0 for i in range(len(y))],color="black")

    ax.set_title(title)
    ax.grid(True)

    ax.set_yticks(ticks = [-.1,0,.1,.2,.3])
    ax.set_ylabel(ylabel)
    new_ticks = [f"[{t}]" for t in ticks]
    ax.set_xticks(ticks=[i for i in range(len(x))])
    ax.set_xticklabels(new_ticks)
    ax.set_xlabel(xlabel)
    ax.legend(prop={'size': 6})


def display_improv_by(results, new_results, ticks, models, title='Metric Improvement'):
    
    r0 = results
    r1 = new_results
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False,figsize=(25, 5))
    for i, metric in enumerate(["acc","f1","mAP", "auROC"]):
        plot_imprv_on_axis(axs[i], models, metric, r0, r1, ticks, f"Improvement ({metric}) vs Head Rotation", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
        
    # fig.suptitle(title, fontsize=16)
    plt.show()
    
def display_improv_metric(results, models, num_buckets=5,label="SPEECH",metric="mAP", title='Metric Improvement by Gaze and Combination Type'):
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False,figsize=(30, 8))
    i = 0
    for gaze_type in ["at","ang"]:
        for mult_type in ["mult", "avg"]:
            r0 = results[num_buckets][label]["original"]
            r1 = results[num_buckets][label][gaze_type][mult_type]
            ticks = results[num_buckets]["xticks"]
            
            plot_imprv_on_axis(axs[i], models, metric, r0, r1, ticks, f"Improvement ({metric}) vs Head Rotation", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
            i+=1
            
        
    fig.suptitle(title, fontsize=16)
    plt.show()
    

def create_paper_imprv_figure(results, gaze_type="ang", mult_type="avg", num_buckets=4, metric="mAP", title='Metric Performance'):
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False,figsize=(25, 5))
    ticks = results[num_buckets]["xticks"]
    # print(ticks)

    if not os.path.exists(f"visuals/{title}/"):
        os.makedirs(f"visuals/{title}/")

    label="SPEECH"
    r0 = results[num_buckets][label]["original"]
    r1 = results[num_buckets][label][gaze_type][mult_type]
    plot_imprv_on_axis(axs[0], synthesizers, metric, r0, r1, ticks, "", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[0]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Synthesizers Speech Improvement.png", bbox_inches=extent)

    plot_imprv_on_axis(axs[1], detectors, metric, r0, r1, ticks, "", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[1]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Detectors Speech Improvement.png", bbox_inches=extent)

    label="TURN"
    r0 = results[num_buckets][label]["original"]
    r1 = results[num_buckets][label][gaze_type][mult_type]
    plot_imprv_on_axis(axs[2], synthesizers, metric, r0, r1, ticks, "", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[2]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Synthesizers Turn Improvement.png", bbox_inches=extent)

    plot_imprv_on_axis(axs[3], detectors, metric, r0, r1, ticks, "", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
    extent = full_extent(axs[3]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"visuals/{title}/Detectors Turn Improvement.png", bbox_inches=extent)

    # fig.suptitle(title,y=0.98, fontsize=16)
    plt.savefig(f"visuals/{title}.png")
    
    # plt.show()


if __name__ == '__main__':
    gaze = ["at","ang"]
    synthesizers = ["pConf","sConf"]
    detectors = ["2LAYER_TCN_SYNCNET","2LAYER_TCN_PERFECTMATCH","2LAYER_BLSTM_SYNCNET","2LAYER_BLSTM_PERFECTMATCH"]

    for trained_on in ["RFSG","FOVA"]:
        for tested_on in ["FOVA","RFSG"]:
            with open(f'scored_results/trained_on_{trained_on}_tested_on_{tested_on}_results.json', 'r') as f:
                tmp = json.loads(f.read())
                results={}
                for k, v in tmp.items():
                    results[int(k)] = v

            for mult_type in ["mult","avg"]:
                for gaze_type in ["ang","at"]:
                    print(trained_on,tested_on,mult_type,gaze_type)
                    create_paper_perf_figure(results, num_buckets=5, metric="mAP", title=f'Performance of {trained_on} on {tested_on} Dataset')
                    create_paper_imprv_figure(results, gaze_type="ang", mult_type="mult", num_buckets=5, metric="mAP", title=f'Improvement of {trained_on} on {tested_on} Dataset ({mult_type}x{gaze_type})')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json




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

        
    fig.suptitle(title,y=0.98, fontsize=16)
    plt.show()
    
def create_paper_perf_figure(results, num_buckets=4, metric="mAP", title='Metric Performance'):
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False,figsize=(25, 5))
    ticks = results[num_buckets]["xticks"]

    label="SPEECH"
    r = results[num_buckets][label]["original"]
    plot_perf_on_axis(axs[0], originals, metric, r, ticks, "Synthesizers Speech Performance", f'{metric} Scores', 'Head Rotation (degrees)')
    plot_perf_on_axis(axs[1], new, metric, r, ticks, "Fine-Tuned Speech Performance", f'{metric} Scores', 'Head Rotation (degrees)')
    
    label="TURN"
    r = results[num_buckets][label]["original"]
    plot_perf_on_axis(axs[2], originals, metric, r, ticks, "Synthesizers Turn Performance", f'{metric} Scores', 'Head Rotation (degrees)')
    plot_perf_on_axis(axs[3], new, metric, r, ticks, "Fine-Tuned Speech Performance", f'{metric} Scores', 'Head Rotation (degrees)')

    fig.suptitle(title,y=0.98, fontsize=16)
    plt.savefig(f"visuals/{title}.png")
    
    plt.show()

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
#     ax.boxplot(data, positions = range(len(ticks)))
    ax.plot(x, [sum(row)/len(row) for row in data], marker='o', linestyle='solid', linewidth=1, markersize=2,label="mean",color="black")
#     ax.plot(x, [0 for i in range(len(y))],color="black")

    ax.set_title(title)
    ax.grid(True)

    ax.set_yticks(ticks = [-.1,0,.1,.2,.3])
    ax.set_ylabel(ylabel)
    new_ticks = [f"[{t}]" for t in ticks]
    ax.set_xticks(ticks=[i for i in range(len(x))])
    ax.set_xticklabels(new_ticks)
    ax.set_xlabel(xlabel)
    ax.legend(prop={'size': 6})

def display_improv_by(results,new_results, ticks, models,title='Metric Improvement'):
    
    r0 = results
    r1 = new_results
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False,figsize=(25, 5))
    for i, metric in enumerate(["acc","f1","mAP", "auROC"]):
        plot_imprv_on_axis(axs[i], models, metric, r0, r1, ticks, f"Improvement ({metric}) vs Head Rotation", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
        
    fig.suptitle(title, fontsize=16)
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
    print(ticks)

    label="SPEECH"
    r0 = results[num_buckets][label]["original"]
    r1 = results[num_buckets][label][gaze_type][mult_type]
    plot_imprv_on_axis(axs[0], originals, metric, r0, r1, ticks, "Synthesizers Speech Improvement", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
    plot_imprv_on_axis(axs[1], new, metric, r0, r1, ticks, "Fine-Tuned Speech Improvement", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
    
    label="TURN"
    r0 = results[num_buckets][label]["original"]
    r1 = results[num_buckets][label][gaze_type][mult_type]
    plot_imprv_on_axis(axs[2], originals, metric, r0, r1, ticks, "Synthesizers Turn Improvement", f'{metric} Improv. Scores', 'Head Rotation (degrees)')
    plot_imprv_on_axis(axs[3], new, metric, r0, r1, ticks, "Fine-Tuned Turn Improvement", f'{metric} Improv. Scores', 'Head Rotation (degrees)')

    fig.suptitle(title,y=0.98, fontsize=16)
    plt.savefig(f"visuals/{title}.png")
    
    plt.show()


if __name__ == '__main__':
    with open(f'k_to_c_results.json', 'w') as f:
        results = json.loads(f.read())

    originals = list(results[0].keys())[:2]
    gaze = list(results[0].keys())[2:4]
    new = list(results[0].keys())[4:]
    create_paper_perf_figure(results, num_buckets=8, metric="mAP", title='Performance on RFSG Dataset')
    create_paper_imprv_figure(results, gaze_type="ang", mult_type="avg", num_buckets=4, metric="mAP", title='Improvement on RFSG Dataset')

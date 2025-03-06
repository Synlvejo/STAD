import argparse
from graphlstm_vae_ad import GraphLSTM_VAE_AD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Circle, Rectangle

parser = argparse.ArgumentParser(description="train")
parser.add_argument('-p', '--program', type=str, required=True, help='program')
parser.add_argument('-d', '--duration', type=str, required=True, help='duration')
parser.add_argument('-n', '--num', type=int, required=True, help='nodes num')
parser.add_argument('-per', '--percent', type=float, default=0.5, help='percent')
parser.add_argument('-i', '--num_inter', type=int, default=2, help='num_inter')
parser.add_argument('-delay', '--delay', type=int,default=5, help='delay')

parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu')
parser.add_argument('-s', '--seq', type=int, default=30, help='seq')
parser.add_argument('-l', '--lr', type=float, default=1e-3, help='lr')
parser.add_argument('-b', '--bsz', type=int, default=32, help='bsz')
parser.add_argument('-hd', '--hdim', type=int,default=8, help='hdim')
parser.add_argument('-e', '--epoch', type=int,default=200, help='epoch')
parser.add_argument('-mode', '--mode', type=int,default=0, help='mode')

args = parser.parse_args()

process = args.program
duration = args.duration
node_num = args.num

gpu = args.gpu
seq = args.seq
lr = args.lr
bsz = args.bsz
hdim = args.hdim
num_epochs=args.epoch

percent = args.percent
mode = args.mode
delay = args.delay

is_static = ""
if mode == 1:
    is_static = "_static"

print(process + "_" + duration
            + '_seq=' + str(args.seq)
            + '_lr=' + str(args.lr)
            + '_bsz=' + str(args.bsz)
            + '_hdim=' + str(args.hdim)
            + '_epoch=' + str(num_epochs))

print("loading scores...")
scores = np.load('./results/scores/' + process + "_" + duration+ is_static
            + '_delay=' + str(delay)
            + '_seq=' + str(args.seq)
            + '_lr=' + str(args.lr)
            + '_bsz=' + str(args.bsz)
            + '_hdim=' + str(args.hdim)
            + '_epoch=' + str(num_epochs) + '_score.npy')


print("caculating distribution..")
def generate_intervals(start, end, num_intervals):
    interval_width = (end - start) / (num_intervals) 
    
    intervals = []
    
    intervals.append((-np.inf, start))
    
    for i in range(num_intervals):
        interval_start = start + i * interval_width
        interval_end = interval_start + interval_width
        intervals.append((interval_start, interval_end))
    
    intervals.append((end, np.inf))
    
    return intervals

def count_elements_in_intervals(array, intervals):
    counts = []
    for interval in intervals:
        count = np.sum((array >= interval[0]) & (array < interval[1]))
        counts.append(count)
    return counts

array = scores
min_value = np.min(array)
max_value = np.max(array)
num_intervals = args.num_inter

mean = np.mean(array)
print("mean:", mean)

variance = np.var(array)
print("variance:", variance)

std_dev = np.std(array)
print("std_dev:", std_dev)

intervals = generate_intervals(min_value, max_value, num_intervals)
counts = count_elements_in_intervals(array, intervals)

for interval, count in zip(intervals, counts):
    print(f"interval {interval}: {count} ")

print("printing heatmap after filtering...")
t0 = mean + 1 * std_dev
t1 = mean + 2 * std_dev
t2 = mean + 3 * std_dev
array = array.T

def categorize(value):
    if value <= t0:
        return 0  
    elif t0 < value <= t1:
        return 1  
    elif t1 < value <= t2:
        return 2  
    else:
        return 3  

categorized_arr = np.vectorize(categorize)(array)

mycolors='Paired'
colors_map = plt.get_cmap(mycolors)(range(20))

colors = ["white", "lightblue", colors_map[6], colors_map[5]]

cmap = ListedColormap(colors)
plt.figure(figsize=(10, 4))
sns.heatmap(categorized_arr, cmap=cmap, cbar=False)

ax = plt.gca()

for i in range(categorized_arr.shape[0]):
    for j in range(categorized_arr.shape[1]):
        if categorized_arr[i, j] == 2:  
            rect = Rectangle((j - 0.25, i), 1.5, 1, edgecolor=colors[2], facecolor=colors[2])
            ax.add_patch(rect)


for i in range(categorized_arr.shape[0]):
    for j in range(categorized_arr.shape[1]):
        if categorized_arr[i, j] == 3: 
            rect = Rectangle((j - 0.25, i), 1.5, 1, edgecolor=colors[3], facecolor=colors[3])
            ax.add_patch(rect)        

legend_elements = [
    Patch(facecolor='white', edgecolor='black', label=r'$(-\infty, \mu+\sigma]$'),
    Patch(facecolor='lightblue', edgecolor='black', label=r'$(\mu+\sigma,\mu+2\sigma]$'),
    Patch(facecolor=colors[2], edgecolor='black', label=r'$(\mu+2\sigma, \mu+3\sigma]$'),
    Patch(facecolor=colors[3], edgecolor='black', label=r'$(\mu+3\sigma, +\infty)$', linewidth=2)
]

plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=4, frameon=False, fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.yticks(rotation=0)
plt.xlabel('Trace Slices', fontsize=12)
plt.ylabel('Nodes', fontsize=12)

plt.savefig('./results/heatmaps_filter/' + process + "_" + duration + is_static
            + '_delay=' + str(delay)
            + '_seq=' + str(args.seq)
            + '_lr=' + str(args.lr)
            + '_bsz=' + str(args.bsz)
            + '_hdim=' + str(args.hdim)
            + '_epoch=' + str(num_epochs) + '_interval_score.png', bbox_inches='tight')

plt.show()




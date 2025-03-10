import os
import json
from random import randint 

import numpy as np
import matplotlib as mp
mp.use('agg')
from matplotlib import pyplot as plt
from tqdm import tqdm

from cuda9_3 import SIZE, N, decode

""" Визуализация 100 случайных заполнений. Сохраняется в папку results. """

def setup_figure():
    plt.figure(figsize=(8.0, 8.0))
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)

def plot_combo(a):
    colors = [
        "",
        "navy",
        "indigo",
        "magenta",
        "red",
        "orange",
        "yellowgreen",
        "lawngreen",
        "turquoise",
        "dodgerblue",
    ]

    boxes = decode(a)
    plt.clf()
    for y, x, s in boxes:
        plt.bar(x, s, s, SIZE-(y+s), align='edge', color=colors[s], edgecolor='k')
        if s > 1:
            plt.text(x + s/2, SIZE - y - s/2, str(s), ha='center', va='center_baseline', size=10*s)

def save_figure(path):
    plt.gca().axis("off")
    plt.xlim(-0.1, SIZE+0.1)
    plt.ylim(-0.1, SIZE+0.1)
    plt.savefig(path, bbox_inches='tight')

if __name__ == '__main__':

    outdir = "results"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    json_path = '2025_all.json'
    with open(json_path, 'r') as f:
        all_combinations = json.load(f)

    setup_figure()

    p = np.random.permutation(len(all_combinations))
    for i in tqdm(range(100)):
        a = all_combinations[p[i]]
        plot_combo(a)
        save_figure(outdir + f"/image{i+1:06d}.png")
    print("Visualizations saved to folder ", outdir)

        


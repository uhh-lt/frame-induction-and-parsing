import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from itertools import cycle
import numpy as np


#DEFAULT_COLORS = cycle(['b', 'r', 'g', 'y', 'black', 'brown', '#660066', 'orange'])
DEFAULT_COLORS = ['b', 'r', 'g', 'y', 'black', 'brown', '#660066', 'orange']


def create_precision_recall_plot(precision_recall_curves: Dict[str, pd.DataFrame], 
                                 colors=DEFAULT_COLORS, output_path=None, step=True):
    """ precision_recall_curves: {exp_name : pd.DataFrame([precision, recall])} """
    
    plt.figure(dpi=200, figsize=(6.4, 4.8), facecolor='w', edgecolor='k')
    
    max_y = -1
    max_x = -1
    for i, (exp_name, curve) in enumerate(precision_recall_curves.items()):
        #tps, aps, gps = load_tps_aps_gps(os.path.join(result_path, exp_name))
        data = pd.DataFrame({'precision' : curve.precision, 
                             'recall' : curve.recall})
        
        col = colors[i]
        marker = ''
        if type(col) is tuple:
            col, marker = col
        
        if step:
            graph = plt.step(data['recall'], data['precision'], color=col, 
                             marker=marker, alpha=0.5, where='post', linewidth=1.)
        else:
            graph = plt.plot(data['recall'], data['precision'], color=col, marker=marker, alpha=0.5, linewidth=1.)
            
        graph[0].set_label(exp_name)
            
        if data['recall'].max() > max_x:
            max_x = data['recall'].max()
        
        if data['precision'].max() > max_y:
            max_y = data['precision'].max()

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, max_y + 0.025])
    plt.xlim([0.0, max_x + 0.005])
    plt.legend(prop={'size': 9})
    
    if output_path is not None:
        plt.savefig(output_path)
    
    
def create_precision_plot(precision_curves: Dict[str, pd.DataFrame], 
                          colors=DEFAULT_COLORS, output_path=None, step=True):
    """ precision_recall_curves: {exp_name : pd.DataFrame([precision, recall])} """
    
    plt.figure(dpi=200, figsize=(6.4, 4.8), facecolor='w', edgecolor='k')
    
    max_y = -1
    rng = 0
    for i, (exp_name, curve) in enumerate(precision_curves.items()):
        data = pd.DataFrame({'precision' : curve.precs_all_hard})
        
        col = colors[i]
        marker = ''
        if type(col) is tuple:
            col, marker = col
        
        rng = np.array(range(1, data.shape[0] + 1))
        if step:
            graph = plt.step(rng, data['precision'], color=col, 
                             marker=marker, alpha=0.5, where='post', linewidth=1.)
        else:
            graph = plt.plot(rng, data['precision'], 
                             color=col, marker=marker, alpha=0.5, linewidth=1.)
            
        graph[0].set_label(exp_name)
        
        if data['precision'].max() > max_y:
            max_y = data['precision'].max()

    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.ylim([0.0, max_y + 0.025])
    plt.xlim([0.0, len(rng) + 1])
    plt.legend(prop={'size': 9})
    
    if output_path is not None:
        plt.savefig(output_path)
        

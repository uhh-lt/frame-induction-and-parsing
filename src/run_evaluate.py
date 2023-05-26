import pandas as pd
import matplotlib.pyplot as plt
import pickle
import fire
import os
import json
import sys
from .evaluate import precision_recall_curve, precision_at_level, calc_MAP_at_k
from .evaluate import precision_at_all_levels_hard, calc_nDCG_score
from .evaluate import precision_recall_curve_hard, precision_at_level_hard



def load_tps_aps_gps(load_dir_path):
    with open(os.path.join(load_dir_path, 'tps.pkl'), 'rb') as f:
        tps = pickle.load(f)

    with open(os.path.join(load_dir_path, 'aps.pkl'), 'rb') as f:
        aps = pickle.load(f)

    with open(os.path.join(load_dir_path, 'gps.pkl'), 'rb') as f:
        gps = pickle.load(f)
        
    return tps, aps, gps


def calc_irmetric_for_several_exps(names, k, metric):
    results = {}
    
    for exp_name in names:
        tps, aps, gps = load_tps_aps_gps(os.path.join(load_dir_path_base, exp_name))
        score = metric(tps, k=k)
        results[exp_name] = score
    
    return results


calc_nDCG_for_several_exps = lambda names, k: calc_irmetric_for_several_exps(names, k, calc_nDCG_score)
calc_MAP_for_several_exps = lambda names, k: calc_irmetric_for_several_exps(names, k, calc_MAP_at_k)

def create_precision_recall_curve(annots, exp_name, output_file_path=None):
    tps, aps, gps = annots
    #precision, recall = precision_recall_curve(tps, aps, gps, n_jobs=6)
    precision, recall = precision_recall_curve_hard(tps, gps, n_jobs=6)
    metrics = pd.DataFrame.from_dict({'precision' : precision, 'recall' : recall})
    if output_file_path is not None:
        metrics.to_csv(output_file_path)
    return metrics
    
    
def keep_only_indexes(annot, indexes):
    res = pd.Series(annot, index=range(len(annot)))
    return res[indexes].tolist()
    
        
def main(results_path, levels=[1, 5, 10], k=50, exp_names=None, test_indexes_path=None, prefix='',
        min_cluster_size=0, max_cluster_size=0): 
    if exp_names is None:
        exp_names = os.listdir(results_path)
    elif type(exp_names) is str:
        exp_names = exp_names.split(',')
    exp_names = [exp_name for exp_name in exp_names if not exp_name.startswith('.')]
    if prefix:
        prefix += '_'
    
    if test_indexes_path is not None:
        with open(test_indexes_path, 'r') as f:
            test_indexes = json.load(f)
    
    for exp_name in exp_names:
        try:
        
            print(f'{exp_name}  ########################')

            exp_name_spl = exp_name.split(':')
            exp_path = os.path.join(results_path, exp_name_spl[0])
            if len(exp_name_spl) > 1:
                exp_path = os.path.join(exp_path, exp_name_spl[1])

            annots = load_tps_aps_gps(exp_path)
            if test_indexes_path is not None:
                annots = tuple(keep_only_indexes(a, test_indexes) for a in annots)

            annots_df = pd.DataFrame({'tps' : annots[0], 'aps' : annots[1], 'gps' : annots[2]})
            annots_df = annots_df[annots_df.gps != 0]
            if min_cluster_size!=0:
                annots_df = annots_df[annots_df.gps >= min_cluster_size]
            if max_cluster_size!=0:
                annots_df = annots_df[annots_df.gps <= max_cluster_size]
                k = max_cluster_size
            print(len(annots_df))
            annots = annots_df.tps.tolist(), annots_df.aps.tolist(), annots_df.gps.tolist()

            curve = create_precision_recall_curve(annots, exp_name, 
                                                  output_file_path=os.path.join(exp_path, 
                                                                                f'{prefix}precision_recall.csv'))
            tps, aps, gps = annots

            precs = precision_at_level_hard(tps, levels=levels)
            #precs = precision_at_level(tps, aps, levels=levels)
            #mean_av_prec = calc_MAP_score(tps, k=k)
            #mean_av_prec = calc_MAP_score(curve['precision'], curve['recall'])
            mean_av_prec = calc_MAP_at_k(tps, gps, k)

            nDCG = calc_nDCG_score(tps, k)

            metrics = {}
            metrics['precisions_at_level'] = {str(lev) : prec for lev, prec in zip(levels, precs)}
            metrics['map'] = mean_av_prec
            metrics['nDCG'] = nDCG
            with open(os.path.join(exp_path, f'{prefix}precision.json'), 'w') as f:
                json.dump(metrics, f, indent=4)

            precs_all_hard = precision_at_all_levels_hard(tps, k)
            pd.DataFrame.from_dict({'precs_all_hard' : precs_all_hard}).to_csv(os.path.join(exp_path, 
                                                                                            f'{prefix}precs_all_hard.csv'))

            print('Precision at levels:')
            print('\n'.join([str(e) for e in zip(levels, precs)]))
            print('MAP: ', metrics['map'])
#             print('Precision recall curve')
#             print(curve)
            print('################################')

        except Exception as ex:
            print(ex)
            
if __name__ == '__main__':
    fire.Fire(main)
    
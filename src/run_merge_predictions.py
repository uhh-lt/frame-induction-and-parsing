import os
import fire
import json
import pandas as pd
import pickle 
from ordered_set import OrderedSet 
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import multiprocessing
import more_itertools 
from statistics import mean

from .evaluate import calculate_tp
from .run_postprocessing_predictions import calculate_true_positives 
from .run_postprocessing_predictions import read_gold_dataset
# ---------------------------------------------------------- BERT and DT interection

def index(v, *lists):
    p =  [L.index(v)  if v in L else 1000 for L in lists]
    return p 
    
    
def ranked_intersect(*L):
    L = L[0][0:]
    sets = [OrderedSet(l) for l in L]
    final = OrderedSet.intersection(*sets)
    ranks = [index(v, *sets) for v in final]
    word_rank = [(w, r, mean(r)) for w, r in zip(final, ranks)]
    sorted_ranks = sorted(word_rank, key=lambda x: x[2])
    return [v[0] for v in sorted_ranks]

def ranked_union(*L):
    L = L[0][0:]
    sets = [OrderedSet(l) for l in L]
    final = OrderedSet.union(*sets)
    ranks = [(index(v, *sets)) for v in final]
    
    word_rank = [(w, r, mean(r)) for w, r in zip(final, ranks)]
    sorted_ranks = sorted(word_rank, key=lambda x: x[2])
    return [v[0] for v in sorted_ranks]


def merge(merge_func, k, n_jobs, *predictions):
    
    merge_func = {'union': ranked_union,
                 'intersect': ranked_intersect}[merge_func]
    progress_bar=tqdm
    L = []
    N = len(predictions)
    print(f'merge_func:{merge_func.__name__}, prediction lists: {N}, k: {k} --not used....\n')
    assert (N>=2), "Lists of predictions must be at least 2 for merging..." 
    
    for i in range(len(predictions[0])):
        temp = []
        for j in range(N):
            temp.append(predictions[j][i])
        L.append(temp)  
        
    with multiprocessing.Pool(n_jobs) as pool:
            res = pool.map(merge_func, progress_bar(L), chunksize=200)
#         return [merge_func(*L[i]) for i in progress_bar(range(len(predictions[0])))]
    return res
        
        
def exp_list(results_path, exp_names):
    
    if exp_names is None:
        exp_names = os.listdir(results_path)
    elif type(exp_names) is str:
        exp_names = exp_names.split(',')
            
    exp_names = [exp_name for exp_name in exp_names if not exp_name.startswith('.')]
   
    return exp_names


def merge_func_symbol(merge_func):
    if merge_func == 'union':
        return '+'
    if merge_func == 'intersect':
        return '&'    
    
def main(gold_path, results_path1, results_path2, save_results_path, 
         merge_func='union', 
         exp_names1=None, exp_names2=None, 
         test_indexes_path=None, 
         k=50, n_jobs=16,
         ranked = False,
         ranking_column='frameName',
         **kwargs):
        
               
    if test_indexes_path is not None:
        with open(test_indexes_path, 'r') as f:
            test_indexes = json.load(f)
    else:
        test_indexes = None
        
    gold_data = read_gold_dataset(gold_path)
    gold_dataset = gold_data['dataset']
    gold_cluster_column = gold_data['gold_cluster_column']
    
    if test_indexes is not None:
        gold_dataset = gold_dataset.loc[test_indexes].copy().reset_index(drop=True)    
    
    if ranked:
        print('drop duplicates of goldset for ranked evaluation')
        gold_dataset = gold_dataset.drop_duplicates(subset=[ranking_column]).copy()
    
    exp_names1 = exp_list(results_path1, exp_names1)
    exp_names2 = exp_list(results_path2, exp_names2)
    print('exp_names1', exp_names1)
    print('exp_names2', exp_names2)


    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
        
#  ---------------------------------------- 
    print('kwargs: ',kwargs)
    more_experiments = []
    if kwargs:
        E = len(kwargs) // 2
        for e in range(E):
            path = kwargs[f"results_path{e+3}"]
            exps = kwargs[f"exp_names{e+3}"]
            print(path)
            print(exps)
            more_experiments.append((path, exp_list(path, exps)))

        for i, exps in enumerate(more_experiments):
            predictions_list = []
            results_path, exp_names = exps

            for exp_name in exp_names:
        
                pred_path = os.path.join(results_path, exp_name, 'final_predictions.pkl')
                with open(pred_path, 'br') as f:
                    predictions = pickle.load(f)
                
                predictions_list.append(predictions)
            
            more_experiments[i] = more_experiments[i][0],  more_experiments[i][1], predictions_list
# ----------------------------------------
    for exp_name1 in exp_names1:
        
        pred_path = os.path.join(results_path1, exp_name1, 'final_predictions.pkl')
        with open(pred_path, 'br') as f:
             predictions1 = pickle.load(f)
        
        for exp_name2 in exp_names2:
            
            pred_path = os.path.join(results_path2, exp_name2, 'final_predictions.pkl')
            with open(pred_path, 'br') as f:
                 predictions2 = pickle.load(f)
            # ---------------------------------------                
            if len(more_experiments)==0:                

                print(f'merging {exp_name1} and {exp_name2} with {merge_func}...')

                predictions = merge(merge_func, k, n_jobs, predictions1, predictions2)

                symbol = merge_func_symbol(merge_func)
                merged_exp =  f'{symbol}'.join([exp_name1, exp_name2])

                print('Saving final predictions...', len(predictions))
                save_dir_path = os.path.join(save_results_path, merged_exp)

                if not os.path.exists(save_dir_path):
                    os.mkdir(save_dir_path)

                with open(os.path.join(save_dir_path, 'final_predictions.pkl'), 'wb') as f:
                    pickle.dump(predictions, f)

                calculate_true_positives(predictions, gold_dataset[gold_cluster_column].tolist(), 
                                        save_dir_path, 
                                         k, n_jobs, progress_bar=tqdm)

                print('Done....')

            else: # there are more predictions to combine
                if len(more_experiments) == 1: # 3 total predictions to combine
                    results_path3, exp_names3, predictions_list3 = more_experiments[0]
                    print('exp_names3', exp_names3)


                    for exp_name3, predictions3 in zip(exp_names3, predictions_list3):
                        
                        print(f'merging {exp_name1} and {exp_name2} and {exp_name3} using {merge_func}')

                        predictions = merge(merge_func, k, n_jobs, predictions1, predictions2, predictions3)
        
                        symbol = merge_func_symbol(merge_func)
                        merged_exp =  f'{symbol}'.join([exp_name1, exp_name2, exp_name3])

                        print('Saving final predictions...', len(predictions))
                        save_dir_path = os.path.join(save_results_path, merged_exp)

                        if not os.path.exists(save_dir_path):
                            os.mkdir(save_dir_path)

                        with open(os.path.join(save_dir_path, 'final_predictions.pkl'), 'wb') as f:
                            pickle.dump(predictions, f)

                        calculate_true_positives(predictions, gold_dataset[gold_cluster_column].tolist(), 
                                                save_dir_path, 
                                                 k, n_jobs, progress_bar=tqdm)
                            
                        print('Done....')


                    
                elif len(more_experiments) == 2: # 4 predictions to combine
                                                 
                    results_path3, exp_names3, predictions_list3 = more_experiments[0]
                    results_path4, exp_names4, predictions_list4 = more_experiments[1]
                    print('exp_names3', exp_names3)
                    print('exp_names4', exp_names4)


                    for exp_name3, predictions3 in zip(exp_names3, predictions_list3):
                                                 
                        for exp_name4, predictions4 in zip(exp_names4, predictions_list4):
                                                 
                            print(f'merging {exp_name1} and {exp_name2} and {exp_name3} and {exp_name4} using {merge_func}')

                            predictions = merge(merge_func, k, n_jobs, predictions1, predictions2, predictions3, predictions4)

                            symbol = merge_func_symbol(merge_func)
                            merged_exp =  f'{symbol}'.join([exp_name1, exp_name2, exp_name3, exp_name4])

                            print('Saving final predictions...', len(predictions))
                            save_dir_path = os.path.join(save_results_path, merged_exp)

                            if not os.path.exists(save_dir_path):
                                os.mkdir(save_dir_path)

                            with open(os.path.join(save_dir_path, 'final_predictions.pkl'), 'wb') as f:
                                pickle.dump(predictions, f)

                            calculate_true_positives(predictions, gold_dataset[gold_cluster_column].tolist(), 
                                                     save_dir_path, 
                                                     k, n_jobs, progress_bar=tqdm)


                            print('Done....')

                                                 
                                                 
if __name__ == '__main__':

    fire.Fire(main)

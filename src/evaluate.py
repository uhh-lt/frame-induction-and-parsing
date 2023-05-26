import pandas as pd
import re
from ordered_set import OrderedSet
import numpy as np
from tqdm.auto import tqdm
import multiprocessing, itertools
from functools import partial
from joblib import Parallel, delayed
from .util import strList_toList
from .rank_metrics import ndcg_at_k, mean_average_precision

# ------------------------------------------

def _precision(row, *args):
    
    k = args[0]
    
    gold_cluster = OrderedSet(list(row['gold_cluster']))
    predictions = OrderedSet(list(row['predictions']))
    predictions = predictions[:k]
     
    n_c = len(predictions & gold_cluster)
           
    return n_c / k



def _recall(row, *args):
    
    k = args[0]
    
    gold_cluster = OrderedSet(list(row['gold_cluster']))
    predictions = OrderedSet(list(row['predictions']))
    predictions = predictions[:k]

    N = len(gold_cluster)    
    n_c = len(predictions & gold_cluster)
    if N==0: return 0
    
    return n_c / N


              

def evaluate(predictions, gold_clusters, k = 1, method = 'cased_T'):
    """do all the normalization on predictions and gold clusters before calling this method, e.g. lemma, lowercase etc"""
    
    if type(predictions.iloc[0]) is not list: 
        predictions = predictions.apply(strList_toList)


    if type(gold_clusters.iloc[0]) is not list: 
        gold_clusters = gold_clusters.apply(strList_toList)

    df = pd.DataFrame()
    
    df['predictions'] = predictions
    df['gold_cluster'] = gold_clusters

 
    df['precision'] = df.apply(_precision, axis = 1, args = (k,))
    df['recall'] = df.apply(_recall, axis = 1, args = (k,))
    
    precision = df['precision'].mean() * 100
    recall = df['recall'].mean() * 100
    
#     res_df = pd.DataFrame(columns = ['method', 'k', 'precision', 'recall']) 
#     res_df.loc[len(res_df)] = [method, k, precision, recall]

    return (precision, recall)

# ------------------------------------------- 

# def calculate_tp_iter(frames, predictions, gold_clusters, k, i):
#     gold_objs = frames[gold_clusters.iloc[i].frameName] 
#     row_pred = predictions[i]
#     row_res = []
#     aps = []
#     for j in range(k):
#         if j < len(row_pred):
#             row_res.append((row_pred[j] in gold_objs))
#             aps.append(j + 1)
#         else:
#             row_res.append(False)
#             aps.append(len(row_pred))
    
#     return row_res, aps, [len(gold_objs) - 1] * k # For exclusion of seed lemma




# def calculate_tp(predictions, gold_clusters, tp_column, k, n_jobs, progress_bar=tqdm):
#     if progress_bar is None:
#         progress_bar=lambda _: _
        
#     frames = {}
#     for i in gold_clusters.index:
#         if gold_clusters.frameName.loc[i] not in frames:
#             frames[gold_clusters.frameName.loc[i]] = set(gold_clusters[tp_column].loc[i])
        
#     with multiprocessing.Pool(n_jobs) as pool:
#         res = pool.map(partial(calculate_tp_iter, frames, predictions, gold_clusters, k), 
#                        progress_bar(range(len(predictions))), 
#                        chunksize=800)
    
#     tps, aps, gps = zip(*res)
#     return tps, aps, gps
# with 

def calculate_tp_iter(gold_objs, row_pred, k):
    row_res = []
    aps = []
    for j in range(k):
        if j < len(row_pred):
            row_res.append((row_pred[j] in gold_objs))
            aps.append(j + 1)
        else:
            row_res.append(False)
            aps.append(len(row_pred))
    
    return row_res, aps, max(0, len(gold_objs) - 1)  # For exclusion of seed lemma

    
def calculate_tp(predictions, gold_clusters, k, n_jobs, progress_bar=tqdm):
#     if progress_bar is None:
#         progress_bar=lambda _: _
#     res = Parallel(n_jobs=n_jobs)(delayed(calculate_tp_iter)(gold_clusters[i], predictions[i], k) 
#                                   for i in progress_bar(range(len(gold_clusters))))   
    args = [(gold_clusters[i], predictions[i], k) for i in range(len(gold_clusters))]
    with multiprocessing.Pool(n_jobs) as pool:
        res = pool.starmap(calculate_tp_iter, progress_bar(args), chunksize=200)
                
    tps, aps, gps = zip(*res)
    return tps, aps, gps


 
# We can calculate the precision - recall curve of multiple "queries" if we aggregate them into single "curve" by
# averaging the results on each level

# We can do not penalize the precision if model does not output anything on the current and further steps simply it will keep the
# same precision and same recall till the end (the end of line). It will negatively affect map and overall curve (it will be a point)

# how we average affects not precision but recall.

def precision_recall_curve(tps, aps, gps, n_jobs):
    with multiprocessing.Pool(n_jobs) as pool:
        tps = pool.map(np.cumsum, tqdm(tps), chunksize=800)
    
    tps, aps, gps = (np.array(e).sum(axis=0) for e in (tps, aps, gps)) # aps already contains the cumulative sum
    # gps contains the number of all gold results
    return tps / aps, tps / gps

# dsfs

def precision_recall_curve_hard(tps, gps, n_jobs):
    aps = np.array(list(range(1, len(tps[0]) + 1)))
    aps = aps * len(tps)
    
    with multiprocessing.Pool(n_jobs) as pool:
        tps = pool.map(np.cumsum, tqdm(tps), chunksize=800)
        
    tps, gps = (np.array(e).sum(axis=0) for e in (tps, gps)) 
    
    return tps / aps, tps / gps



def calc_nDCG_score(tps, k):
    all_scores = []
    for r in tps:
        all_scores.append(ndcg_at_k(r, k=k))

    return np.asarray(all_scores).mean()
    

# to calculate map you also need to know number of all ground truths or a recall
# so do not use mean_average_precision from package rank_metrics since it induces the number of all gps from all the results
# We can also normalize the results according to best recall that can be acheived till k.
#def calc_MAP_score(precisions):
    # incorrect
    #return np.mean(precisions)
    
def calc_MAP_score(precisions, recalls):
    # It is just an area under the curve
    
    prev_rec = recalls[0]
    res = 0.
    for i in range(precisions.shape[0] - 1):
        res += precisions[i] * (recalls[i + 1] - prev_rec)
        prev_rec = recalls[i + 1]
    
    return res # TODO: scale it

    #return mean_average_precision((e[:k] for e in tps))
#     all_scores = []
#     for r in tps:
#         print(r)
#         print(len(r))
#         all_scores.append(mean_average_precision(r[:k]))
        
#     return np.asarray(all_scores).mean()


def precision_at_all_levels_hard(tps, n_jobs=6):
    aps = np.array(list(range(1, len(tps[0]) + 1))) * len(tps)
    
    with multiprocessing.Pool(n_jobs) as pool:
        tps = pool.map(np.cumsum, tqdm(tps), chunksize=800)
    
    tps = np.array(tps).sum(axis=0)
    
    return tps / aps


def precision_at_level(tps, aps, levels):
    result = []
    
    for lev in levels:
        all_t = sum([sum(t[:lev]) for t in tps])
        ap_t = sum([a[lev-1] for a in aps])
        result.append(all_t / ap_t)
        
    return result


def precision_at_level_hard(tps, levels):
    result = []
    
    for lev in levels:
        all_t = sum([sum(t[:lev]) for t in tps])
        ap_t = lev * len(tps)
        result.append(all_t / ap_t)
        
    return result


def calc_MAP_at_k(tps, gps, k=None):
    assert len(tps) == len(gps)
    
    if k is None:
        k = len(tps[0])
        
    if gps is None:
        gps = [len(tps[0])] * len(tps)
        
    prec_div = np.array(range(1, len(tps[0]) + 1))
    
    map_at_k = 0.
    for tp, gp in zip(tps, gps): 
        div = min(gp, k)
        ap = ((np.cumsum(tp) /  prec_div) * np.array(tp)).sum() / div
        map_at_k += ap
    
    return map_at_k / len(tps)

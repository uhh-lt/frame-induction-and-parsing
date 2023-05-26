import fire
import os
import sys
import pickle
import numpy as np
import scipy
import gensim
from pathlib import Path
import multiprocessing as mp
from ordered_set import OrderedSet
import pandas as pd
from pandarallel import pandarallel
from .logger import create_logger

main_dir = 'workdir/melamud_lexsub'
word_embedding = f'{main_dir}/lexsub_word_embeddings.txt'
context_embedding = f'{main_dir}/lexsub_context_embeddings.txt'


fname1 = f'{main_dir}/word_vectors.model'
fname2 = f'{main_dir}/context_vectors.model'

if not Path(fname1).exists():
    word_model=gensim.models.KeyedVectors.load_word2vec_format(word_embedding, binary=False)
    # filename should be absolute, to save model
    word_model.save(os.path.abspath(os.path.expanduser(os.path.expandvars(fname1))))

if not Path(fname2).exists():
    context_model=gensim.models.KeyedVectors.load_word2vec_format(context_embedding, binary=False)
    context_model.save(os.path.abspath(os.path.expanduser(os.path.expandvars(fname2))))

word_model = gensim.models.KeyedVectors.load(fname1, mmap='r')
context_model = gensim.models.KeyedVectors.load(fname2, mmap='r')
###########################################################################################
word_vocab = OrderedSet(word_model.vocab.keys())
context_vocab = OrderedSet(context_model.vocab.keys())
   
def Add(w, C):

    S_vecs = np.array([word_model[s] for s in word_vocab])
    t_vec = np.array([word_model[w]])
    target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
    C_vecs = np.array([context_model[c] for c in C])

    context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
    context_sim = context_sim.sum(axis=1)
    context_sim = context_sim.reshape(context_sim.shape[0],1)

    combine_sim = (target_sim + context_sim) / (len(C) + 1)
    similarity_dict = dict({k:v[0] for k,v in zip(word_vocab,combine_sim)})

    return similarity_dict

def BalAdd(w, C):

    S_vecs = np.array([word_model[s] for s in word_vocab])
    t_vec = np.array([word_model[w]])
    C_vecs = np.array([context_model[c] for c in C])

    target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
    target_sim = target_sim * len(C)

    context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
    context_sim = context_sim.sum(axis=1)
    context_sim = context_sim.reshape(context_sim.shape[0],1)

    combine_sim = (target_sim + context_sim) / (len(C) * 2)

    similarity_dict = dict({k:v[0] for k,v in zip(word_vocab,combine_sim)})

    return similarity_dict

def Mult(w, C):

    S_vecs = np.array([word_model[s] for s in word_vocab])
    t_vec = np.array([word_model[w]])
    C_vecs = np.array([context_model[c] for c in C])

    target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
    target_sim = (target_sim + 1)/2

    context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
    context_sim = (context_sim + 1) / 2
    context_sim = context_sim.prod(axis=1)
    context_sim = context_sim.reshape(context_sim.shape[0],1)  

    combine_sim = (target_sim * context_sim) ** (1 /(len(C) +1))

    similarity_dict = dict({k:v[0] for k,v in zip(word_vocab,combine_sim)})
    return similarity_dict


def BalMult(w, C):

    S_vecs = np.array([word_model[s] for s in word_vocab])
    t_vec = np.array([word_model[w]])
    C_vecs = np.array([context_model[c] for c in C])

    target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
    target_sim = (target_sim + 1)/2
    target_sim = target_sim ** len(C)

    context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
    context_sim = (context_sim + 1) / 2
    context_sim = context_sim.prod(axis=1)
    context_sim = context_sim.reshape(context_sim.shape[0],1)

    combine_sim = (target_sim * context_sim) ** (1 /(len(C) *2))

    similarity_dict = dict({k:v[0] for k,v in zip(word_vocab,combine_sim)})
    return similarity_dict
                
def _similar_words(word, C, n_top = 200):

    if not word in word_vocab: return ([], [])

#     drop contexts which are not present in context model vocab
    C = [c for c in C if c in context_vocab]
    if C == []: return ([], [])

    similarity_dict = METRIC_FUNC(word, C)

    res = sorted(similarity_dict.items(), key=lambda items: items[1], reverse=True)
    res = res[:n_top]
    terms = [e[0] for e in res]
    scores = [e[1] for e in res]
    
    return terms, scores


def similar_words(row):
    row['predictions'], row['scores'] = _similar_words(row['bpe_tokens'][row['masked_position']].text.lower(), row['C'], 200)
#     row['predictions'], row['scores'] = _similar_words(row['plaintext'].lower(), row['word'].lower(), row['C'], 200)
    return row

def parallel_process(df, func, num_cores =24):
    df_chunks = np.array_split(df, num_cores)
    pool = mp.Pool(num_cores)
    df = pd.concat(pool.map(func, df_chunks))
    pool.close()
    pool.join()
    return df


def parallel_similar_words(df_chunk):
    df_chunk = df_chunk.apply(similar_words, axis=1)
    return df_chunk


def main(input_file, result_dir, metric, jobs=16):
    global METRIC_FUNC
    if metric.lower() == "add": 
        METRIC_FUNC = Add
    if metric.lower() == "baladd": 
        METRIC_FUNC = BalAdd
    if metric.lower() == "mult": 
        METRIC_FUNC = Mult
    if metric.lower() == "balmult": 
        METRIC_FUNC = BalMult
    
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
        
    logger = create_logger('usem-experiments', result_dir)
    
    df = pd.read_pickle(input_file)[['bpe_tokens','masked_position', 'C']]

        
    logger.info(f'input file:{input_file}')  
    logger.info(f'result dir:{result_dir}')    

    logger.info(f'input records:{len(df)}')
    logger.info(f'metric:{METRIC_FUNC.__name__}')

    pandarallel.initialize(nb_workers=jobs, progress_bar=True)
    df = df.parallel_apply(similar_words, axis=1)

    # df = parallel_process(df, parallel_similar_words)
    logger.info(f'Saving results to {result_dir} ...')
    predictions = df['predictions'].tolist()
    scores = df['scores'].tolist()
    masked_words = df.apply(lambda row: row['bpe_tokens'][row['masked_position']].text, axis=1)
    with open(os.path.join(result_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)

    with open(os.path.join(result_dir, 'scores.pkl'), 'wb') as f:
        pickle.dump(scores, f)

    with open(os.path.join(result_dir, 'masked_words.pkl'), 'wb') as f:
        pickle.dump(masked_words, f)

    logger.info('Done.')
    
if __name__ == '__main__':
    fire.Fire(main)
    
import os
import fire
import json
import pandas as pd
import pickle 
from tqdm import tqdm as tqdm
from ordered_set import OrderedSet 
from pathlib import Path

from .evaluate import calculate_tp

from .postprocessing import unify_predictions, clean_noisy_predictions, remove_digits
from .postprocessing import remove_seedword, filter_vocab, remove_stopwords, remove_duplicate_predictions
from .postprocessing import lemmatize, lemmatize_predictions, lemmatize_lemminflect, lemmatize_predictions_lemminflect
from .postprocessing import lemmatize_nltk, lemmatize_predictions_nltk,filter_nounPOStags_nltk,filter_verbPOStags_nltk

from .postprocessing import remove_verbPOStags, remove_nounPOStags, filter_nounPOStags,filter_verbPOStags
from .postprocessing import remove_verbPOStags_lemminflect, remove_nounPOStags_lemminflect, filter_nounPOStags_lemminflect,filter_verbPOStags_lemminflect
from .postprocessing import ROLE_STOPWORDS, NOUN_STOPWORDS

PARSER_FUNCS = {'pattern':
                        {'lemmatize': lemmatize,
                       'lemmatize_predictions': lemmatize_predictions,
                       'filter_verbPOStags': filter_verbPOStags,
                       'filter_nounPOStags': filter_nounPOStags,
                       'remove_verbPOStags': remove_verbPOStags,
                       'remove_nounPOStags': remove_nounPOStags
                     },
                'lemminflect':
                        {'lemmatize': lemmatize_lemminflect,
                       'lemmatize_predictions': lemmatize_predictions_lemminflect,
                       'filter_verbPOStags': filter_verbPOStags_lemminflect,
                       'filter_nounPOStags': filter_nounPOStags_lemminflect,
                       'remove_verbPOStags': remove_verbPOStags_lemminflect,
                       'remove_nounPOStags': remove_nounPOStags_lemminflect
                    },
                 'nltk':
                        {'lemmatize': lemmatize_nltk,
                        'lemmatize_predictions': lemmatize_predictions_nltk,
                        'filter_verbPOStags': filter_verbPOStags_nltk,
                        'filter_nounPOStags': filter_nounPOStags_nltk,
#                         'remove_verbPOStags': remove_verbPOStags_nltk,
#                         'remove_nounPOStags': remove_nounPOStags_nltk
                }
           }
    
# ------------------------------------------- read gold datasets and few essential columns 
def read_gold_dataset(gold_path, parser='pattern'):
    
    with open(gold_path, 'br') as f:
        gold_dataset = pickle.load(f)
        
    if 'luName' in gold_dataset.columns:
        seed_word_column = 'luName'
        gold_cluster_column = 'gold_cluster_processed'
        
    else:
        seed_word_column = 'feText'
        gold_cluster_column = 'gold_cluster_patternlemmatized'
        if parser == 'lemminflect':
            gold_cluster_column = 'gold_cluster_lemminflectlemmatized'
        if parser == 'nltk':
            gold_cluster_column = 'gold_cluster_nltklemmatized'    
        
    return {'dataset':gold_dataset,
            'seed_word_column':seed_word_column,
            'gold_cluster_column':gold_cluster_column
           }


def postprocess(predictions, seed_words, proc_funcs, vocabulary=None, stopwords=None, parser='pattern', dataset_type='verbs', reprocess=False):
    # only useful for nltk
    POS = {'nouns': {'pattern':None, 'lemminflect':None, 'nltk': 'n'},
           'verbs': {'pattern':None, 'lemminflect':None, 'nltk': 'v'},
           'roles': {'pattern':None, 'lemminflect':None, 'nltk': 'n'}
          }[dataset_type][parser]
    
    print(f'parser:{parser} \t dataset: {dataset_type}\t proc_funcs: {proc_funcs}\nreprocess:{reprocess}\n')            
    if not reprocess:
        print('unifying predictions')
        predictions = unify_predictions(predictions)
    
    if 'clean_noisy' in proc_funcs:
        print('clean noisy')
        predictions = clean_noisy_predictions(predictions)
        
    if 'remove_digits' in proc_funcs:
        print('remove digits')
        predictions = remove_digits(predictions)
            
    if 'lemmatize' in proc_funcs:
        print('lemmatization of predictions')
        predictions = [[pred.lower() for pred in predictions[i]]
            for i in range(len(predictions))]
        
        predictions = PARSER_FUNCS[parser]['lemmatize_predictions'](predictions, [POS]*len(predictions))
        
        print('lemmatization of seed words')
        seed_words = [w.lower() for w in seed_words]
        seed_words = PARSER_FUNCS[parser]['lemmatize'](seed_words, POS)
    
    if 'remove_role_stopwords' in proc_funcs:
        print('filter stopwords')
        predictions = remove_stopwords(predictions, OrderedSet(stopwords))
                                 
    if 'remove_noun_stopwords' in proc_funcs:
        print('filter stopwords')
        predictions = remove_stopwords(predictions, OrderedSet(stopwords))
       
    if 'filter_vocab' in proc_funcs:
        print('filter predictions using vocabulary list')
        predictions = filter_vocab(predictions, vocabulary)
        
    if 'filter_verbs' in proc_funcs:
        print('filter verbs using POS tags')
        predictions = PARSER_FUNCS[parser]['filter_verbPOStags'](predictions)
    
    if 'filter_nouns' in proc_funcs:
        print('filter nouns using POS tags')
        predictions = PARSER_FUNCS[parser]['filter_nounPOStags'](predictions)
   
#     if 'remove_verbs' in proc_funcs:
#         print('remove verbs using postags')
#         predictions = PARSER_FUNCS[parser]['remove_verbPOStags'](predictions)
    
#     if 'remove_nouns' in proc_funcs:
#         print('remove nouns using postags')
#         predictions = PARSER_FUNCS[parser]['remove_nounPOStags'](predictions)
 
    if not reprocess:
                        
        print('removing seeding lemma')
        predictions = remove_seedword(seed_words, predictions)

        print('removing duplicates')
        predictions = remove_duplicate_predictions(predictions)


    return predictions


def calculate_true_positives(predictions, gold_clusters, save_dir_path, 
                             k=50, n_jobs=16, progress_bar=tqdm):
               
    print('Calculating true positives')
    tps, aps, gps = calculate_tp(predictions, 
                                 gold_clusters, 
                                 k=k, n_jobs=n_jobs, progress_bar=progress_bar)

    print('Saving results...')
    with open(os.path.join(save_dir_path, 'tps.pkl'), 'wb') as f:
        pickle.dump(tps, f)

    with open(os.path.join(save_dir_path, 'aps.pkl'), 'wb') as f:
        pickle.dump(aps, f)

    with open(os.path.join(save_dir_path, 'gps.pkl'), 'wb') as f:
        pickle.dump(gps, f)
        
    print('Done.')
    

def postprocess_results(gold_path, exp_results_path, exp_name, 
                        proc_funcs, vocabulary_path=None, stopwords_path=None, parser='pattern', dataset_type='verbs',
                        test_indexes_path=None,
                        save_results_path=None, k=50, n_jobs=16, progress_bar=tqdm):
    
    if vocabulary_path is None:
        vocabulary = None
    else:
        with open(vocabulary_path, 'r') as f:
            vocabulary = f.readlines()
        vocabulary = set([v.replace('\n', '').strip().lower() for v in vocabulary])
    
    stopwords = None
    if stopwords_path is None:
        if dataset_type == 'nouns':
            stopwords = NOUN_STOPWORDS
        if dataset_type == 'roles':
            stopwords = ROLE_STOPWORDS           
    else:
        with open(stopwords_path, 'r') as f:
            stopwords = f.readlines()
        stopwords = set([v.replace('\n', '').strip().lower() for v in stopwords])
    
    gold_data = read_gold_dataset(gold_path, parser)
    gold_dataset = gold_data['dataset']
    seed_word_column = gold_data['seed_word_column']
    gold_cluster_column = gold_data['gold_cluster_column']

    pred_path = Path(exp_results_path) / exp_name
    reprocess = False
    print(f'pred_path:{pred_path}')
    if (pred_path / 'final_predictions.pkl').exists():
        with open(pred_path / 'final_predictions.pkl', 'br') as f:
            predictions = pickle.load(f)
            reprocess = True
            
    elif (pred_path / 'predictions.pkl').exists():
        with open(pred_path / 'predictions.pkl', 'br') as f:
            predictions = pickle.load(f)
            
    elif (pred_path / 'results.csv').exists():
        preds_df = pd.read_csv(pred_path / 'results.csv', sep=',', converters={'pred_substitutes': eval})
        predictions = [[s for s in l] for l in preds_df.pred_substitutes]
#         assert len(predictions)==len(gold_dataset), f'{len(predictions)}!={len(gold_dataset)}'
#         print(preds_df[['pred_substitutes','context']].head())    
    else:
        raise Exception(f'Prediction file not found in {pred_path}')
        
    if test_indexes_path is not None:
        with open(test_indexes_path, 'r') as f:
            test_indexes = json.load(f)
            if len(predictions)!= len(test_indexes):
                gold_dataset['preds'] = predictions
    else:
        test_indexes = None
    
    if test_indexes is not None:
        gold_dataset = gold_dataset.loc[test_indexes].copy().reset_index(drop=True)    
        
        if len(predictions)!= len(test_indexes):
            predictions = gold_dataset['preds'].tolist()

    seed_words = gold_dataset[seed_word_column].tolist()        
    
    predictions = postprocess(predictions, seed_words, proc_funcs, 
                              vocabulary, stopwords, parser=parser, dataset_type=dataset_type, reprocess=reprocess)
           
    print('Saving final predictions...', len(predictions))
    if save_results_path:
        save_dir_path = os.path.join(save_results_path, exp_name)
    else:
        save_dir_path = os.path.join(exp_results_path, exp_name)

    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    
    with open(os.path.join(save_dir_path, 'final_predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
        
    if 'remove_role_stopwords' in proc_funcs and seed_word_column=="feText":
        print('Filter stopwords from gold cluster')
        gold_dataset[gold_cluster_column] = remove_stopwords(gold_dataset[gold_cluster_column].tolist(), stopwords)

                 
    calculate_true_positives(predictions, gold_dataset[gold_cluster_column].tolist(), 
                             save_dir_path, 
                             k=k, n_jobs=n_jobs, progress_bar=tqdm)
                        
#---------------------------------
def main(gold_path, results_path, proc_funcs, parser='pattern', dataset_type='verbs', exp_names=None, 
         save_results_path=None, vocabulary_path=None, stopwords_path=None,
         test_indexes_path=None, 
         k=50, n_jobs=16):
    
    if type(proc_funcs) is str:
        proc_funcs = [proc_funcs]

    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
        
    if exp_names is None:
        exp_names = os.listdir(results_path)
    elif type(exp_names) is str:
        exp_names = exp_names.split(',') # TODO: check why sometimes comma separated args are tuples and sometimes are strings
            
    exp_names = [exp_name for exp_name in exp_names if not exp_name.startswith('.')]
    
    for exp_name in exp_names:
        try:
            print(f'Postprocessing experiment {exp_name}.')

            postprocess_results(gold_path, results_path, exp_name, 
                                proc_funcs=proc_funcs,
                                vocabulary_path=vocabulary_path,
                                stopwords_path=stopwords_path,
                                parser=parser,dataset_type=dataset_type,
                                test_indexes_path=test_indexes_path,
                                save_results_path=save_results_path, k=k, n_jobs=n_jobs)

            print(f'Postprocessing experiment {exp_name} finished.')
        except Exception as ex:
            print(ex)
                    
        
if __name__ == '__main__':
    fire.Fire(main)
    
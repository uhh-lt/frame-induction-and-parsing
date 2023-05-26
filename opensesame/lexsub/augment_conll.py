import os
from itertools import cycle
from ordered_set import OrderedSet
import copy
import shutil
import pandas as pd 
from tqdm import tqdm
import sys
import numpy as np

from sesame.dataio import read_conll
from sesame.conll09 import CoNLL09Example, CoNLL09Element
from sesame.sentence import Sentence
from lexsub.conll_helper import is_multitokenLU, reconstruct_example
from lexsub.conll_helper import get_frameName
from lexsub.conll_helper import conll_to_sentence, sort_sentencewise, reset_sentnum, example_to_str

import src
from src.run_predict import load_predictor
from src.predict import analyze_tagged_text
from src.predict import get_masked_tokens_from_tagged_text

from src.postprocessing import unify_predictions, clean_noisy_predictions, lemmatize_predictions, lemmatize, lemmatize_predictions_nltk, lemmatize_nltk,lemmatize_lemminflect, lemmatize_predictions_lemminflect
from src.postprocessing import remove_seedword, remove_nolemma_seedword, remove_stopwords, remove_duplicate_predictions
from src.postprocessing import remove_digits, remove_vocab, filter_vocab, match_inflection, match_POStags 
from src.postprocessing import filter_verbPOStags_nltk, filter_nounPOStags_nltk
from src.postprocessing import remove_verbPOStags, remove_nounPOStags, filter_nounPOStags,filter_verbPOStags
from src.postprocessing import remove_verbPOStags_lemminflect, remove_nounPOStags_lemminflect, filter_nounPOStags_lemminflect,filter_verbPOStags_lemminflect

from src.postprocessing import correct_predictions, correct_nolemma_predictions, correct_nolemma_predictions_nltk, correct_nolemma_predictions_lemminflect
from src.lemmatize_util import pattern_lemma, pattern_parse, nltk_lemma, nltk_poslemma, nltk_postag, nltk_parse

from src.postprocessing import ROLE_STOPWORDS, NOUN_STOPWORDS, NOUN_POSTAGS
# ----------------------------------------------------------------------

DATA_DIR = 'data/open_sesame_v1_data/fn1.7'

DEV_FILE = 'fn1.7.dev.syntaxnet.conll'
TEST_FILE = 'fn1.7.test.syntaxnet.conll'
TRAIN_FILE = 'fn1.7.fulltext.train.syntaxnet.conll'

vocabulary_path='../workdir/resources/verbs_list.txt'
with open(vocabulary_path, 'r') as fp:
    vocabulary = fp.readlines()

VERB_VOCAB = set([v.replace('\n', '').lower() for v in vocabulary])


PROC_FUNCS_OPTIONS={
'nolemma' : {
            'lu_v': 'clean_noisy,filter_vocab,match_inflection',
            'lu_n': 'clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns,match_inflection' ,
            'role' : 'clean_noisy',
            'noun':'clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns,match_inflection'
            },
    
'lemma' : {
            'lu_v': 'lemmatize,clean_noisy,filter_vocab',
            'lu_n': 'lemmatize,clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns',
            'role' : 'lemmatize,clean_noisy',
            'noun':'lemmatize,clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns'
            },

'nolemma_role_stopwords' : {
                    'lu_v': 'clean_noisy,filter_vocab,match_inflection',
                    'lu_n': 'clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns,match_inflection',
                    'role' : 'clean_noisy,remove_role_stopwords',
                    'noun':'clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns,match_inflection'
                    },
'lemma_role_stopwords' : {
                    'lu_v': 'lemmatize,clean_noisy, filter_vocab',
                    'lu_n': 'lemmatize,clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns',
                    'role' : 'lemmatize,clean_noisy,remove_role_stopwords',
                    'noun':'lemmatize,clean_noisy,remove_digits,remove_noun_stopwords,filter_nouns'
                    },

}
# BIOS scheme settings for roles
BEGINNING = 0
INSIDE = 1
OUTSIDE = 2
SINGULAR = 3

MODEL_CONFIGS = {'bert-large-cased': 'blc',
                }

POS_MAP = {'lu_v':'v',
           'lu_n':'n',
          'noun':'n', 
          'ibnoun': 'n',
          'role': 'n'}

# -----------------------------------------------
def read_goldclusters(gold_clusters_verbs_path, gold_clusters_nouns_path, gold_clusters_roles_path):
    gold_clusters_verbs = None
    gold_clusters_nouns = None
    gold_clusters_roles = None

    if gold_clusters_verbs_path:
        
        gold_clusters_verbs = pd.read_pickle(gold_clusters_verbs_path)
        gold_clusters_verbs.drop_duplicates(subset=['frameName'], inplace=True)
        gold_clusters_verbs.set_index(keys=['frameName'], inplace=True)
        gold_clusters_verbs.sort_index(inplace=True)
        
    if gold_clusters_nouns_path:
        
        gold_clusters_nouns = pd.read_pickle(gold_clusters_nouns_path)
        gold_clusters_nouns.drop_duplicates(subset=['frameName'], inplace=True)
        gold_clusters_nouns.set_index(keys=['frameName'], inplace=True)
        gold_clusters_nouns.sort_index(inplace=True)
        
    if gold_clusters_roles_path:
        gold_clusters_role = pd.read_pickle(gold_clusters_roles_path)
        gold_clusters_role.drop_duplicates(subset=['frameName', 'feName'], inplace=True)
        gold_clusters_role.set_index(keys=['frameName', 'feName'], inplace=True)
        gold_clusters_role.sort_index(inplace=True)
 
    GOLD_CLUSTERS = {
        'lu_v': gold_clusters_verbs,
        'lu_n': gold_clusters_nouns,
        'role': gold_clusters_role,
        'noun': None
    }   
    return GOLD_CLUSTERS
    
def fill_goldclusters(word_types, identifiers, parser, gold_clusters_dfs):
    assert len(word_types)==len(identifiers), print(f'len of word_types != len of identifiers')
    def get_cluster(word_type, identifier):
        
        if word_type == 'noun': return None
        elif word_type == 'roles' and type(gold_clusters_dfs[word_type])!=None: 
            column = f'gold_cluster_{parser}lemmatized'
            return gold_clusters_dfs[word_type].loc[identifier][column]
        elif word_type in ('lu_v', 'lu_n') and  type(gold_clusters_dfs[word_type])!=None:
            frame = identifier[0]
            return gold_clusters_dfs[word_type].loc[frame]['gold_cluster_processed']
        else:
            return None
        
    gold_clusters = gold_clusters = [get_cluster(word_types[i],  identifiers[i]) for i in tqdm(range(len(word_types)))]
    
    return  gold_clusters        
    

def predict(tagged_text, predictor, use_tokenizer=False, 
            n_top = 200,
            n_units=1, 
            n_tokens=[1], 
            mask_token=False, 
            max_multiunit = 1,
            batch_size=16, 
            do_lemmatize=False, do_lowercase=True, 
            n_jobs=1, mask_in_multiunit=False, other_params=None):
    
    
    predictions, scores, masked_words = analyze_tagged_text(tagged_text, predictor, use_tokenizer, 
                                                        n_top=n_top, 
                                                        n_tokens=n_tokens,
                                                        n_units=n_units,
                                                        progress_bar=tqdm,
                                                        mask_token=mask_token,
                                                        max_multiunit=max_multiunit)
    return predictions



PARSER_FUNCS = {'pattern':
                {'lemmatize': lemmatize,
                       'lemmatize_predictions': lemmatize_predictions,
                       'filter_verbPOStags': filter_verbPOStags,
                       'filter_nounPOStags': filter_nounPOStags,
                      'correct_nolemma_predictions': correct_nolemma_predictions
                 },
            'lemminflect':
                {'lemmatize': lemmatize_lemminflect,
                       'lemmatize_predictions': lemmatize_predictions_lemminflect,
                       'filter_verbPOStags': filter_verbPOStags_lemminflect,
                       'filter_nounPOStags': filter_nounPOStags_lemminflect,
                       'correct_nolemma_predictions': correct_nolemma_predictions_lemminflect
                },
             'nltk':
                {'lemmatize': lemmatize_nltk,
                        'lemmatize_predictions': lemmatize_predictions_nltk,
                        'filter_verbPOStags': filter_verbPOStags_nltk,
                        'filter_nounPOStags': filter_nounPOStags_nltk,
                        'correct_nolemma_predictions': correct_nolemma_predictions_nltk
                }
           }

def postprocess_fsp(predictions, seed_words, word_types, proc_funcs='clean_noisy', vocabulary=[], seed_postags=None, parser='nltk', verbose=False): #pipeline options lu, role, noun
    print(f'parser:{parser} \t proc_funcs: {proc_funcs}\n')
    print('unifying predictions')
    predictions = unify_predictions(predictions)
    
    if 'clean_noisy' in proc_funcs:
        if verbose: print('clean noisy')
        predictions = clean_noisy_predictions(predictions)
               
    if 'remove_digits' in proc_funcs:
        if verbose: print('remove digits')
        predictions = remove_digits(predictions)
        
    if 'filter_vocab' in proc_funcs:
        if verbose: print('filter predictions using vocabulary list')
        predictions = filter_vocab(predictions, vocabulary)

    if 'filter_verbs' in proc_funcs:
        print('filter verbs using POS tags')
        predictions = PARSER_FUNCS[parser]['filter_verbPOStags'](predictions)
    
    if 'filter_nouns' in proc_funcs:
        if verbose: print('filter nouns using POS tags')
        predictions = PARSER_FUNCS[parser]['filter_nounPOStags'](predictions)
    
    if 'filter_roles' in proc_funcs:
        if verbose: print('match roles using POS tags')
        predictions = PARSER_FUNCS[parser]['match_POStags'](predictions, seed_postags)

    if 'remove_noun_stopwords' in proc_funcs:
        if verbose: print('remove noun stopwords')
        predictions = remove_stopwords(predictions, OrderedSet(NOUN_STOPWORDS))   
        
    if 'remove_role_stopwords' in proc_funcs:
        if verbose: print('remove role stopwords')
        predictions = remove_stopwords(predictions, OrderedSet(ROLE_STOPWORDS))
    
    
    if 'lemmatize' in proc_funcs:
        if verbose: print('lemmatization of seed words')
        seed_postypes = [POS_MAP[w] for w in word_types] 
        PARSER_FUNCS[parser]['lemmatize'](seed_words, seed_postypes)    
        
        if verbose: print('lemmatization of predictions')
        predictions_postypes = [[POS_MAP[word_types[i]] for p in preds] for i, preds in enumerate(predictions)]
        predictions = PARSER_FUNCS[parser]['lemmatize_predictions'](predictions, predictions_postypes)
        
        if verbose: print('Removing seeding lemma') 
        predictions = remove_seedword(seed_words, predictions)
    else:
        if verbose: print('Removing seeding lemma') 
        predictions = remove_nolemma_seedword(seed_words, predictions)
        
    # for verbs
    if 'match_inflection' in proc_funcs and seed_postags:
        if verbose: print('matching inflection form')
        if not 'lemmatize' in proc_funcs: #inflection needs lemma form
            if verbose: print('lemmatization of predictions before inflection')
            predictions_postypes = [[POS_MAP[word_types[i]] for p in preds] for i, preds in enumerate(predictions)]
            predictions = PARSER_FUNCS[parser]['lemmatize_predictions'](predictions, predictions_postypes)
        if verbose: print('inflecting')
        predictions = match_inflection(predictions, seed_postags) 
     
    if verbose: print('removing duplicates')
    predictions = [[p.lower() for p in predictions[i]] for i in tqdm(range(len(predictions)))]
    predictions = remove_duplicate_predictions(predictions)
                    
    return predictions

# -----------------------------------
def to_df(config_dicts):
    
    sentences = [config_dict[key]['sentence'] for config_dict in config_dicts for key in config_dict.keys()]
    masked_sentences = [config_dict[key]['masked_sent'] for config_dict in config_dicts for key in config_dict.keys()]
    indices = [config_dict[key]['index'] for config_dict in config_dicts for key in config_dict.keys()]

    seed_words = [config_dict[key]['seed_word'] for config_dict in config_dicts for key in config_dict.keys()]
    gold_clusters = [config_dict[key]['gold_cluster'] for config_dict in config_dicts for key in config_dict.keys()]
    postags = [config_dict[key]['postag'] for config_dict in config_dicts for key in config_dict.keys()]
    word_types = [config_dict[key]['word_type'] for config_dict in config_dicts for key in config_dict.keys()]
    
    identifiers = [config_dict[key]['identifier'] for config_dict in config_dicts for key in config_dict.keys()]
    e_ids = [key for config_dict in config_dicts for key in config_dict.keys()]
    ex_ids = [i+1 for i, config_dict in enumerate(config_dicts) for key in config_dict.keys()]

    df = pd.DataFrame()
    df['sentence'] = sentences
    df['masked_sent'] = masked_sentences
    df['index'] = indices
    df['seed_word'] = seed_words
    df['gold_cluster'] = gold_clusters
    df['word_type'] = word_types
    df['postag'] = postags

    df['identifier'] = identifiers
    df['e_id'] = e_ids
    df['ex_id'] = ex_ids

    
    return df
    
def to_dict(examples, df):
    config_dicts = []
    config_dict = {}
    for ex_id in range(len(examples)):
        subdf = df.loc[df['ex_id']==ex_id+1].copy()
        config_dict = {}
        
        for index, row in subdf.iterrows():
            if 'preds' in df.columns:
                config_dict[row['e_id']] = {'sentence': row['sentence'],
                                                 'masked_sent': row['masked_sent'], 
                                                'index': row['index'], 
                                                'identifier': row['identifier'], 
                                                'seed_word': row['seed_word'],
                                                'gold_cluster': row['gold_cluster'],
                                                'postag': row['postag'],
                                                'word_type': row['word_type'],
                                                'preds':row['preds']
                                               } 
                

            else:
                config_dict[row['e_id']] = {'sentence': row['sentence'],
                                                 'masked_sent': row['masked_sent'], 
                                                'index': row['index'], 
                                                'identifier': row['identifier'], 
                                                'seed_word': row['seed_word'],
                                                'gold_cluster': row['gold_cluster'],
                                                'postag': row['postag'],
                                                'word_type': row['word_type']
                                               } 
        config_dicts.append(config_dict)
        
    return config_dicts
    

def mask_sentences(examples, substitute_lu=True, 
                   substitute_role=False, role_tokens=[1], role_postags=None,
                   noun_max=0, ibn=False,
                   parser='nltk',
                   add_goldclusters=True,
                   gold_clusters_verbs_path='../workdir/data/swv_gold_dataset.pkl',
                   gold_clusters_nouns_path='../workdir/data/swn_gold_dataset.pkl',
                   gold_clusters_roles_path='../workdir/data/swr_gold_dataset.pkl',
                   verbose=False):
    
    GOLD_CLUSTERS = read_goldclusters(gold_clusters_verbs_path, gold_clusters_nouns_path, gold_clusters_roles_path)

    MAX_RT = max(role_tokens)
#     if MAX_RT>1:
#         gold_clusters_role = pd.read_pickle('../workdir/data/mwr_gold_dataset.pkl')
#         gold_clusters_role.drop_duplicates(subset=['frameName', 'feName'], inplace=True)
#         gold_clusters_role.set_index(keys=['frameName', 'feName'], inplace=True)
#         gold_clusters_role.sort_index(inplace=True)
#         GOLD_CLUSTERS['role'] = gold_clusters_role
        
    config_dicts = []
    print('tagging targets')
    for example in tqdm(examples):    
        
        frame = get_frameName(example)
        elements = example._elements
        noun_n=0
        config_dict = {}
        role_elements = []
        if type(role_postags) is str:
            role_postags = role_postags.split(',')
            
#         lu_marked_already = False
        for i, e in enumerate(elements):
            if e.is_pred and not is_multitokenLU(example) and substitute_lu:
                
                # sometimes a lexical unit is single token, but annotations may have multiple words, then consider the first one
                # or may be altogether ignore such case
#                 if lu_marked_already: continue
                    
                if verbose: print('lu:',e._form)

                tagged_text = ' '.join([ex._form if ex!=e else f'__{ex._form}__' for ex in elements ])
                index, clean_text  = get_masked_tokens_from_tagged_text(tagged_text)
                identifier = frame
                seedword = e._form
                gold_cluster = None#GOLD_CLUSTERS[f'lu_{e._lupos}'].loc[identifier]['gold_cluster_processed']
                postags = e._pos


                if verbose: print(postags)
                config_dict[e.id] = {'sentence':clean_text,
                                    'masked_sent':tagged_text,
                                    'index':index,
                                    'identifier': (frame, '.'.join([e._lu, e._lupos])), 
                                    'seed_word':seedword,
                                    'gold_cluster':gold_cluster,
                                    'postag': postags,
                                    'word_type':f'lu_{e._lupos}'
                                    }
#                 lu_marked_already = True
                break
        if substitute_role:
            for i, e in enumerate(elements):
                
                if e.id in config_dict.keys() or e.is_pred: continue # already masked for some word_type
                
                if e.is_arg and e.argtype == SINGULAR and 1 in role_tokens:
                    
                    if role_postags and not e._pos in role_postags: continue
                        
                    if verbose: print('role:',e._form)
                    tagged_text = ' '.join([ex._form if ex!=e else f'__{ex._form}__' for ex in elements])
                    index, clean_text  = get_masked_tokens_from_tagged_text(tagged_text)
                    identifier = (frame, e._role)
                    seedword = e._form
#                     it appears, instead of using nltk_lemmatized gold cluster, pattern_lemmatized has been used here for singular roles of verbs based fsp experiments
                    gold_cluster = None# GOLD_CLUSTERS['role'].loc[identifier]['gold_cluster_nltklemmatized']
#                     if parser == 'pattern':
#                         gold_cluster =  GOLD_CLUSTERS['role'].loc[identifier]['gold_cluster_patternlemmatized']
#                     if parser == 'lemminflect':
#                         gold_cluster =  GOLD_CLUSTERS['role'].loc[identifier]['gold_cluster_lemminflectlemmatized']

                    postags = e._pos

                    if verbose: print(postags)
                    config_dict[e.id] = {'sentence':clean_text,
                                        'masked_sent':tagged_text,
                                        'index':index, 
                                        'identifier': identifier, 
                                        'seed_word':seedword,
                                        'gold_cluster':gold_cluster,
                                        'postag': postags,
                                        'word_type':'role'
                                       }

                elif e.is_arg and e.argtype != SINGULAR and role_tokens!=[1] and len(role_elements) < MAX_RT:
                    if not role_elements:
                        role_elements.append(e)
                    else:
                        role_elements.append(e)
                        identifier = (frame, e._role)
                        if i+1==len(elements) or elements[i+1].argtype != INSIDE: # next element does not belong to this role                    
                            seedword = ' '.join([e._form for e in role_elements])
                            if verbose: print('role:', seedword)

                            tagged_text = ' '.join([ex._form if not ex in [role_elements[0], role_elements[-1]] else f'__{ex._form}' if ex == role_elements[0] else f'{ex._form}__' for ex in elements])

                            index, clean_text  = get_masked_tokens_from_tagged_text(tagged_text)
                            tagged_text = tagged_text.replace(f'__{seedword}__', '__-__')
#                             in current mw_T.pkl, gold_cluster for multitoken roles may refer to nltk_lemmatized
                            gold_cluster =  None#GOLD_CLUSTERS['role'].loc[identifier]['gold_cluster_nltklemmatized']
#                             if parser == 'pattern':
#                                 gold_cluster =  GOLD_CLUSTERS['role'].loc[identifier]['gold_cluster_patternlemmatized']
#                             if parser == 'lemminflect':
#                                 gold_cluster =  GOLD_CLUSTERS['role'].loc[identifier]['gold_cluster_lemminflectlemmatized']
                            
                            postags = [e._pos for e in role_elements]
                            postags = ' '.join(postags)

                            if verbose: print(postags)
                            config_dict['-'.join([f'{e.id}' for e in role_elements])] = {'sentence':clean_text,
                                                'masked_sent':tagged_text,
                                                'index':index, 
                                                'identifier': identifier, 
                                                'seed_word':seedword,
                                                'gold_cluster':gold_cluster,
                                                'postag': postags,
                                                'word_type':'role'
                                               }
                            role_elements = []
        if noun_max>0:
            for i, e in enumerate(elements):
                
                if e.id in config_dict.keys() or e.is_pred: continue # already masked for some word_type
                
                if e._pos in ['NN', 'NNS', 'NNP', 'NNPS'] and len(e._form)>2 and noun_n < noun_max*len(elements):
                    if e.is_arg: 
                        if ibn:
                            if verbose: print('noun:',e._form)
                            noun_n = noun_n + 1
                            tagged_text = ' '.join([ex._form if ex!=e else f'__{ex._form}__' for ex in elements])
                            index, clean_text  = get_masked_tokens_from_tagged_text(tagged_text)
                            seedword = e._form
                            postags = e._pos

                            if verbose: print(postags)
                            config_dict[e.id]= {'sentence':clean_text,
                                                'masked_sent':tagged_text,
                                                'index':index, 
                                                'identifier': None, 
                                                'seed_word':seedword,
                                                'gold_cluster':None,
                                                'postag': postags,
                                                'word_type':'ibnoun'
                                               }


                    else:
                        if verbose: print('noun:',e._form)
                        noun_n = noun_n + 1
                        tagged_text = ' '.join([ex._form if ex!=e else f'__{ex._form}__' for ex in elements])
                        index, clean_text  = get_masked_tokens_from_tagged_text(tagged_text)
                        seedword = e._form
                        postags = e._pos

                        if verbose: print(postags)
                        config_dict[e.id] = {'sentence':clean_text,
                                             'masked_sent':tagged_text, 
                                            'index':index, 
                                            'identifier': None, 
                                            'seed_word':seedword,
                                            'gold_cluster':None,
                                            'postag': postags,
                                            'word_type':'noun'
                                           }   
                         
        # ------------------------------------
        config_dicts.append(config_dict) 
        # ------------------------------------
    df = to_df(config_dicts)    
    print(f'{len(df)} words were tagged in {len(examples)}')
    if add_goldclusters:
        print('~'*50, '\nfilling gold_cluster column...\n', '~'*50)
        GOLD_CLUSTERS = read_goldclusters(gold_clusters_verbs_path, gold_clusters_nouns_path, gold_clusters_roles_path)
        df['gold_cluster'] = fill_goldclusters(df['word_type'].tolist(), df['identifier'].tolist(), parser, 
                                           GOLD_CLUSTERS)
        config_dicts = to_dict(examples, df)
        
    return df, config_dicts

def postprocess_predictions(df, predictions, 
                            proc_funcs=PROC_FUNCS_OPTIONS['lemma'],
                            parser='nltk',
                            verbose=False):
    #     -------------
    
    assert len(df) == len(predictions), f'df length: {len(df)} predictions length: {len(predictions)}'
    df['preds'] = predictions    
    lus_v   = df.loc[df['word_type']=='lu_v']['preds'].tolist()
    lus_n   = df.loc[df['word_type']=='lu_n']['preds'].tolist()

    roles = df.loc[df['word_type']=='role']['preds'].tolist()
    nouns = df.loc[df['word_type'].apply(lambda x: x in ['noun', 'ibnoun'])]['preds'].tolist() 

    # for lu
    if lus_v:
        
        subdf = df.loc[df['word_type']=='lu_v'].copy()
        _seed_words = subdf['seed_word'].tolist()
        _gold_clusters = subdf['gold_cluster'].tolist()
        _postags = subdf['postag'].tolist()
        _proc_funcs = proc_funcs['lu_v']
        _vocabulary = VERB_VOCAB
        _word_types = subdf['word_type'].tolist()
        lus_v = postprocess_fsp(lus_v, _seed_words, _word_types, _proc_funcs, _vocabulary, _postags, parser, verbose)
    
    if lus_n:
        
        subdf = df.loc[df['word_type']=='lu_n'].copy()
        _seed_words = subdf['seed_word'].tolist()
        _gold_clusters = subdf['gold_cluster'].tolist()
        _postags = subdf['postag'].tolist()
        _proc_funcs = proc_funcs['lu_n']
        _vocabulary = None
        _word_types = subdf['word_type'].tolist()
        lus_n = postprocess_fsp(lus_n, _seed_words, _word_types, _proc_funcs, _vocabulary, _postags, parser, verbose)
        
#for roles
    if roles:
        word_type = 'role'
        subdf = df.loc[df['word_type']=='role'].copy()
        _seed_words = subdf['seed_word'].tolist()
        _gold_clusters = subdf['gold_cluster'].tolist()
        _postags = subdf['postag'].tolist()
        _proc_funcs = proc_funcs['role']
        _vocabulary = None
        _word_types = subdf['word_type'].tolist()
        roles = postprocess_fsp(roles, _seed_words, _word_types, _proc_funcs, _vocabulary, _postags, parser, verbose)
          
    #for nouns
    if nouns:
        word_type = 'noun'
        subdf = df.loc[df['word_type'].apply(lambda x: x in ['noun', 'ibnoun'])].copy()
        _seed_words = subdf['seed_word'].tolist()
        _gold_clusters = subdf['gold_cluster'].tolist()
        _postags = subdf['postag'].tolist()
        _proc_funcs = proc_funcs['noun']
        _vocabulary = None
        _word_types = subdf['word_type'].tolist()
        nouns = postprocess_fsp(nouns, _seed_words, _word_types, _proc_funcs, _vocabulary, _postags, parser, verbose)

              
    df.loc[df['word_type']=='lu_v', 'final_preds'] = np.array(lus_v, dtype='object')
    df.loc[df['word_type']=='lu_n', 'final_preds'] = np.array(lus_n, dtype='object')
    df.loc[df['word_type']=='role', 'final_preds'] = np.array(roles, dtype='object')
    df.loc[df['word_type'].apply(lambda x: x in ['noun', 'ibnoun']), 'final_preds'] = np.array(nouns, dtype='object')
    
    return df['final_preds'].tolist()
    

def matchgold_predictions(df, predictions, 
                            match_lugold=True, match_rolegold=True,
                            proc_funcs=PROC_FUNCS_OPTIONS['lemma'],
                            parser='nltk'
                            ):
    #     -------------
    
    assert len(df) == len(predictions), f'df length: {len(df)} predictions length: {len(predictions)}'
    df['preds'] = predictions    
    lus_v   = df.loc[df['word_type']=='lu_v']['preds'].tolist()
    lus_n   = df.loc[df['word_type']=='lu_n']['preds'].tolist()
    roles = df.loc[df['word_type']=='role']['preds'].tolist()
    nouns = df.loc[df['word_type'].apply(lambda x: x in ['noun', 'ibnoun'])]['preds'].tolist() 

    # for lu
    if lus_v:
        
        subdf = df.loc[df['word_type']=='lu_v'].copy()
        _gold_clusters = subdf['gold_cluster'].tolist()
        _proc_funcs = proc_funcs['lu_v']
        if match_lugold and 'lemmatize' in _proc_funcs:
            lus_v = correct_predictions(_gold_clusters, lus_v)
        elif match_lugold:
            lus_v = PARSER_FUNCS[parser]['correct_nolemma_predictions'](_gold_clusters, lus_v, [POS_MAP['lu_v'] for w in lus_v])
    
    if lus_n:
        
        subdf = df.loc[df['word_type']=='lu_n'].copy()
        _gold_clusters = subdf['gold_cluster'].tolist()
        _proc_funcs = proc_funcs['lu_n']
        if match_lugold and 'lemmatize' in _proc_funcs:
            lus_n = correct_predictions(_gold_clusters, lus_n)
        elif match_lugold:
            lus_n = PARSER_FUNCS[parser]['correct_nolemma_predictions'](_gold_clusters, lus_n, [POS_MAP['lu_n'] for w in lus_n])
#for roles
    if roles:
        word_type = 'role'
        subdf = df.loc[df['word_type']=='role'].copy()
        _gold_clusters = subdf['gold_cluster'].tolist()
        _proc_funcs = proc_funcs['role']
        if match_rolegold and 'lemmatize' in _proc_funcs:
            roles = correct_predictions(_gold_clusters, roles)
        elif match_rolegold:
            roles = PARSER_FUNCS[parser]['correct_nolemma_predictions'](_gold_clusters, roles, [POS_MAP['role'] for w in roles])   
            
    df.loc[df['word_type']=='lu_v', 'final_preds'] = np.array(lus_v, dtype='object')
    df.loc[df['word_type']=='lu_n', 'final_preds'] = np.array(lus_n, dtype='object')
    df.loc[df['word_type']=='role', 'final_preds'] = np.array(roles, dtype='object')
    df.loc[df['word_type'].apply(lambda x: x in ['noun', 'ibnoun']), 'final_preds'] = np.array(nouns, dtype='object')
    
    
    return df['final_preds'].tolist()
    
def capitalize_rule(seed_word, seed_idx, word_type, pred):
#     for nouns
    if seed_word[0].isupper() and word_type in set(['noun', 'innoun']):
        return True
#     for roles
    if seed_word[0].isupper() and seed_idx==1:
        return True
    if seed_word[0].isupper() and nltk_postag(pred) in NOUN_POSTAGS:
        return True
    
    return False

def augment_sentence(seed_text, eids, preds):
    tokens = seed_text.split(" ")
    MAX_P = max([len(ps) for ps in preds])
    expanded_texts = OrderedSet()
    for n in range(MAX_P):
        for eid, ps in zip(eids, preds):
            if ps:
                D = MAX_P - len(ps)
                if D>0:
                    ps.extend([p for p in ps for i in range(D)])
                
                tokens[eid-1] = ps[n]

        new_text = " ".join(tokens)
        expanded_texts.add(new_text)
    return list(expanded_texts)


def generate_newExamples(examples, 
                         df, predictions, 
                         match_lugold=True, match_rolegold=True,
                         proc_funcs=PROC_FUNCS_OPTIONS['lemma'],
                         N=2,
                         E=-1,
                         parser='nltk',
                         verbose=False):
    
    df['preds'] = predictions    
    if match_lugold or match_rolegold:
        df['preds'] = matchgold_predictions(df, predictions, 
                                            match_lugold=match_lugold, match_rolegold=match_rolegold,
                                            proc_funcs=proc_funcs,
                                            parser=parser)

    df['preds'] = df['preds'].apply(lambda x: x[:N])
    df['preds'] = df.apply(lambda row:[w.capitalize() if capitalize_rule(row['seed_word'], row['e_id'], row['postag'], w) else w for w in row['preds']], axis=1)

    config_dicts = to_dict(examples, df)
    config_dicts = config_dicts[:E] if E!=-1 else config_dicts
    examples = examples[:E] if E!=-1 else examples

    all_new_examples = []

    print('final expansions')
    for ex_id in tqdm(range(len(examples))):
        expanded_sents = []
        new_examples = []

        subdf = df.loc[df['ex_id']==ex_id+1].copy()
        lu_eid = 0
        lu_pos= ''
        if len(subdf) >0:
            
            if 'lu_v' in subdf['word_type'].unique():
                lu_eid = subdf.loc[subdf['word_type']=='lu_v']['e_id'].iloc[0]
                lu_pos='v'
            
            elif 'lu_n' in subdf['word_type'].unique():
                lu_eid = subdf.loc[subdf['word_type']=='lu_n']['e_id'].iloc[0]
                lu_pos = 'n'
                
            expanded_sents = augment_sentence(subdf['sentence'].iloc[0], 
                                              subdf['e_id'].tolist(), 
                                              subdf['preds'].tolist()
                                              )
 
                
        for sent in expanded_sents:
            
            parse_output = nltk_parse(sent)
            forms = [token[0] for token in parse_output]
            ppos = [token[1] for token in parse_output]
            plemmas = [token[2] for token in parse_output]
            
            lu = None
            if lu_eid != 0:
                lu = sent.split(' ')[lu_eid-1]
                lu = nltk_poslemma(lu.lower(), lu_pos)
                
            new_examples.append(reconstruct_example(examples[ex_id], forms, plemmas, ppos, lu_lemma=lu))


        all_new_examples.append(new_examples)
    

    return all_new_examples, config_dicts

 
    
def augment_conllExamples(examples, preds_model, use_tokenizer=False, 
                          substitute_lu=True, 
                          substitute_role=False, role_tokens=[1], role_postags=None,
                          noun_max=0, ibn=False, 
                          proc_funcs=PROC_FUNCS_OPTIONS['lemma'], match_lugold=True, match_rolegold=False, 
                          N=2,
                          parser='nltk',
                          verbose=False):
    
    df, config_dicts = mask_sentences(examples, 
                                  substitute_lu=substitute_lu, 
                                  substitute_role=substitute_role, role_tokens=role_tokens, role_postags=role_postags,
                                  noun_max=noun_max, ibn=ibn,
                                  parser=parser,
                                  verbose=verbose)
    
#     -----------------------------------------------------------------------
    print('predicting substitutes')
    use_tokenizer = False
    predictor = preds_model
    if predictor.__class__.__name__ == 'str':
        predictor = load_predictor(preds_model)
        
    print('predictor :', predictor.__class__.__name__)

    predictions = predict(df['masked_sent'].tolist(), predictor, use_tokenizer, n_tokens=role_tokens) 
#     -----------------------------------------------------------------------
    
    final_predictions = postprocess_predictions(df.copy(), predictions,                                           
                                          proc_funcs=proc_funcs,
                                          parser=parser,
                                          verbose=verbose)
    
    
    all_new_examples, config_dicts =  generate_newExamples(examples, df, final_predictions,
                                                           match_lugold=match_lugold, match_rolegold=match_rolegold,
                                                           proc_funcs=proc_funcs,
                                                           N=N,
                                                           parser=parser,
                                                           verbose=verbose)
    
    return {'augmented_examples':all_new_examples, 
            'config_dicts':config_dicts,
            'predictions':predictions,
            'final_predictions':final_predictions}


# ================================
def write_conll(augmented_examples, output_conll_file):
    
    augmented_examples = sort_sentencewise(augmented_examples)
    sentences = []
    for example in augmented_examples:
        sent = conll_to_sentence(example)        
        if not sent in sentences:
            sentences.append(sent)
    
    augmented_examples = reset_sentnum(augmented_examples, sentences)

    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in augmented_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
    
    with open(output_conll_file.replace('.conll', '.conll.sents'), "w", encoding="utf-8") as cf:
        for sent in sentences:
            cf.write(sent)
            cf.write('\n')
            
def copy_dev_test_files(input_dir, output_dir):

    print(f'Copying dev and test files from {input_dir} to {output_dir}')

    shutil.copyfile(f'{input_dir}/{TEST_FILE}', f'{output_dir}/{TEST_FILE}')
    shutil.copyfile(f'{input_dir}/{DEV_FILE}', f'{output_dir}/{DEV_FILE}')
    
    
# ================================
def join_examples(examples, augmented_data):
    augmented_examples = copy.deepcopy(examples)
    for l in augmented_data: augmented_examples.extend(l)
    return augmented_examples

    
def augment_conllFile(input_file, output_file, preds_model='bert-large-cased', 
                      substitute_lu=True, 
                      substitute_role=False, role_tokens=[1], role_postags=None,
                      noun_max=0, ibn=False, 
                      proc_funcs=PROC_FUNCS_OPTIONS['lemma'], match_lugold=True, match_rolegold=False, 
                      N=2,
                      parser='nltk',
                      verbose=False):
    
    examples, __, __ = read_conll(input_file)
        
    results = augment_conllExamples(examples, 
                                              preds_model=preds_model, 
                                              N=N, 
                                              substitute_lu=substitute_lu,
                                              substitute_role=substitute_role, role_tokens=role_tokens, role_postags=role_postags,
                                              noun_max=noun_max, ibn=ibn, 
                                              proc_funcs=proc_funcs,
                                              match_lugold=match_lugold,
                                              match_rolegold=match_rolegold,
                                              parser=parser,
                                              verbose=verbose)
    
    all_examples = join_examples(examples, results['augmented_examples'])
    write_conll(all_examples, output_file)
    return results
    
def main(input_exp, output_exp, data_dir=DATA_DIR, preds_model='bert-large-cased', 
         substitute_lu=True, 
         substitute_role=False, role_tokens="1", role_postags=None,
         noun_max=0, ibn=False, 
         proc_funcs='lemma', match_lugold=True, match_rolegold=False, 
         N=2,
         parser='nltk',
         verbose=False):
    
    if type(role_postags) is str:
            role_postags = role_postags.split(',')
    if type(role_tokens) is str:
            role_tokens = role_tokens.split(',')
    role_tokens = [int(r) for r in role_tokens]
    
    
    if verbose: 
        print('cuda device:', os.environ['CUDA_VISIBLE_DEVICES'] )
        print('exp to expand:', input_exp)
        print('substitute_lu:', substitute_lu)
        print('substitute_role:', substitute_role)
        print('role_tokens:', role_tokens)
        print('role_postags:', role_postags)
        print('max noun percentage:', noun_max)
        print('in_boundary noun substitution:', ibn)
        print('predictor model:', preds_model)
        print('proc_funcs:', PROC_FUNCS_OPTIONS[proc_funcs])
        print('match_lugold:', match_lugold)
        print('match_rolegold:', match_rolegold)
        print('N:', N)

        print('output_exp:', output_exp)

    input_exp = os.path.join(data_dir, input_exp)
    output_exp = os.path.join(data_dir, output_exp)
    input_file = os.path.join(input_exp, TRAIN_FILE)
    output_file = os.path.join(output_exp,TRAIN_FILE)
    
    examples, __, __ = read_conll(input_file)
    results = augment_conllExamples(examples,
                                              preds_model=preds_model, 
                                              substitute_lu=substitute_lu,
                                              substitute_role=substitute_role, role_tokens=role_tokens, role_postags=role_postags,
                                              noun_max=noun_max, ibn=ibn, 
                                              proc_funcs=PROC_FUNCS_OPTIONS[proc_funcs],
                                              match_lugold=match_lugold,
                                              match_rolegold=match_rolegold,
                                              N=N,
                                              parser=parser,
                                              verbose=verbose)
    
    
    all_examples = join_examples(examples, results['augmented_examples'])
    
    if not os.path.exists(output_exp): 
        print(f'creating output_dir: {output_exp}')
        os.makedirs(output_exp, exist_ok=True)
    print('writing train file...')
    write_conll(all_examples, output_file)
    print('writing dev and test file...')
    copy_dev_test_files(input_exp, output_exp)
    print('done...')
# ===========================================
import fire
if __name__ == '__main__':
    fire.Fire(main)
    
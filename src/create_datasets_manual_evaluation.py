import os
import csv
import pickle
import json
import random
import itertools
import pandas as pd
from ordered_set import OrderedSet
from lemminflect import getInflection, getAllInflections, getAllLemmas
from src.lemmatize_util import pattern_lemma, pattern_parse, nltk_postag
from src.run_evaluate import load_tps_aps_gps

UTAGS={
'JJ':'ADJ',   
'JJR':'ADJ',  
'JJS':'ADJ',

'RB':'ADV',
'RBR':'ADV',
'RBS':'ADV',

'NN':'NOUN',
'NNS':'NOUN',
'NNP':'PROPN',
'NNPS':'PROPN',

# upos = 'VERB', 'AUX'
'VB':'VERB',
'VBD':'VERB',
'VBG':'VERB',
'VBN':'VERB',
'VBP':'VERB',
'VBZ':'VERB',
'MD':'VERB'
}


def _match_inflection(pred, postag):
    if postag in UTAGS.keys():
        inflected_form = getInflection(pred, tag=postag)
        if inflected_form:
            return inflected_form[0]
    return pred


def match_inflection(predictions, postags):
    return [[_match_inflection(pred, postags[i]) for pred in predictions[i]] for i in tqdm(range(len(predictions)))]



def capitalize_rule(seed_word, seed_index, pred):
    NOUN_POSTAGS = set(['NN', 'NNS','NNP','NNPS'])
#     for roles
    if seed_word[0].isupper() and seed_index[0]==0:
        return True
    if seed_word[0].isupper() and nltk_postag(pred) in NOUN_POSTAGS:
        return True
    
    return False


def get_frame_maps(frame_description_file='workdir/framenet_data/frame_info.json'):
    
    with open(frame_description_file, 'r') as f:
        frame_desc = json.load(f)

    frame_info = {}
    for fns in frame_desc:
        frame_info[fns['fname']] = {'definition': fns['definition']}

    frameroles_info = {}
    for fns in frame_desc:
        frameroles_info[fns['fname']] = {'definition': fns['definition'],
                                    'FEs':{fe['name']:{'definition':fe['defn']} for fe in fns['FEs']}
                                    }
        
    return frame_info, frameroles_info



def create_predictions_dataframe(results_path,
                                predictors,
                                gold_dataset_path,
                                test_indexes_path=None,
                                columns=None,
                                P=10):
    
    """
    results_path: a directory with subdirectories that contains predictions
    predictors: a dictionary {key:exp_name}, where exp_name is a directory within results_path, and key is a short name to be used as final name for the exp_name
    test_indexes_path: a json file with test indices
    P: number of final predictions to add to output dataset
    """
    gold_df = pd.read_pickle(gold_dataset_path)
    
    if test_indexes_path is not None:
        with open(test_indexes_path, 'r') as f:
            test_indexes = json.load(f)
        test_df = gold_df.loc[test_indexes].copy().reset_index(drop=True)
    else:
        test_df = gold_df
        
    if columns is not None:
        test_df = test_df[columns].copy()
        
    df = pd.DataFrame()
    for k, v in predictors.items():
        df['preds'] = pd.read_pickle(f'{results_path}/{v}/final_predictions.pkl')
        test_df[k] = df['preds'].apply(lambda x: x[:P])
    
    return test_df
    
    
def create_verbs_dataset(results_path,
                            predictor,
                            # save_path,
                            gold_dataset_path='../workdir/data/swv_gold_dataset.pkl',
                            test_indexes_path='../workdir/data/swv_gold_dataset_test_split.json',
                            gold_cluster_col='gold_cluster_processed', 
                            frame_description_file='workdir/framenet_data/frame_info.json',
                            P=10,
                            sample_size=50,
                            seed=11,
                            verbose=False):
    """
    results_path: a directory with subdirectories that contains predictions
    predictor: a tuple (key,exp_name), where exp_name is a directory within results_path, and key is a short name to be used as final name for the exp_name
    test_indexes_path: a json file with test indices
    gold_cluster_col: column name within gold_dataset which contains gold set
    frame_description_file: a json file which contains frame descriptions, represents frame files from FrameNet data
    P: number of final predictions to add to output dataset
    """

    gold_df = create_predictions_dataframe(results_path, {predictor[0]:predictor[1]}, gold_dataset_path, test_indexes_path,
                                          columns=['frameName', 'luName', 'luText', 'luIndex', 'sentence', 'gold_cluster_processed'] )
    
    load_dir_path = f'{results_path}/{predictor[1]}'
    tps, aps, gps = load_tps_aps_gps(load_dir_path)

    gold_df['tps'] = tps
    gold_df['aps'] = aps
    gold_df['gps'] = gps
    gold_df['tps'] = gold_df['tps'].apply(lambda x: x[:P])
    gold_df['aps'] = gold_df['aps'].apply(lambda x: x[:P])
    
   
    random.seed(seed)
    sample = random.sample([i for i in range(0,len(gold_df))], sample_size)
    df = gold_df.iloc[sample].copy().reset_index(drop=True)
    
    frame_info, frameroles_info = get_frame_maps(frame_description_file)

    
    def match_verb_inflection(df,
                              predictor_name,
                              gold_cluster_col='gold_cluster_processed',
                              name_col = 'luName',
                              index_col = 'luIndex',
                              text_col = 'luText'):

        columns=['frameName','frame_description', 'gold_terms','original_term','sentence', 'match gold']
        new_df = pd.DataFrame(columns=columns)

        for gd, fn, luName, luText, luIndex, sentence, predictions, match_gold in zip(df[gold_cluster_col], df['frameName'], df[name_col], df[text_col], df[index_col], df['sentence'], df[predictor_name], df['tps']):

            inflections = getAllInflections(luName)
            postag = None
            inflected_predictions = predictions

            for key,form in inflections.items():
                if form[0] == luText.lower():
                    postag = key
                    if postag in set(['VB','VBD','VBN','VBP','VBZ']):
                        break
            if postag:
                # there can be errors with lemmatization with Pattern for words used as both nouns and verbs like (used, reading), which can lead to invalid inflections
                lemmas = [getAllLemmas(p)['VERB'][0] for p in predictions]
                inflected_predictions = [getInflection(p, tag=postag)[0] for p in lemmas]
            else:
                print('cannot inflect')
                print(luName, luText, inflections)

        # --------------------------------------------------------------- just to check and correct (if possible) if lemmas do not match predictions
            lemmas = [getAllLemmas(p)['VERB'][0] for p in predictions]
            if set(lemmas) - set(predictions) != set():
                if verbose: 
                    print('*** discrepancy in lemmas and preds ***')
                    print(set(lemmas) - set(predictions) )
                    print(f'{fn}-{luName} ----> {luText},{postag}')
                    print(f'predictions:{predictions}')
                    print(f'lemmas:     {lemmas}')
                    print(f'inflected:  {inflected_predictions}')
        # ---------------------------------------------------------------        

            temp_df=pd.DataFrame()
            temp_df['sentence'] = [f'{sentence[:luIndex[0][0]]}<{p}>{sentence[luIndex[0][1]+1:]}' for p in inflected_predictions]
            temp_df['frameName'] = [f'{fn}' for p in predictions]
            temp_df['frame_description'] = [frame_info[fn]['definition'],'','','','','','','','','']
            temp_df['gold_terms'] = [f'{gd}' for p in predictions]

            temp_df['original_term'] = [f'{luText}' for p in predictions]
            temp_df['match gold'] = [1 if g else 0 for g in match_gold]
            new_df = pd.concat([new_df,temp_df])
        #     break
        new_df['gold_terms'] = new_df['gold_terms'].apply(lambda x: eval(x))
        new_df['gold_terms'] = new_df['gold_terms'].apply(lambda x: ','.join(sorted(x)))
#             
        new_df

        return new_df
    
    
    new_df = match_verb_inflection(df, predictor[0])
    return df, new_df


def create_nouns_dataset(results_path,
                            predictor,
                            # save_path,
                            gold_dataset_path='../workdir/data/swn_gold_dataset.pkl',
                            test_indexes_path='../workdir/data/swn_gold_dataset_test_split.json',
                            gold_cluster_col='gold_cluster_processed', 
                            frame_description_file='workdir/framenet_data/frame_info.json',
                            P=10,
                            sample_size=50,
                            seed=11,
                            verbose=False):
    """
    results_path: a directory with subdirectories that contains predictions
    predictor: a tuple (key,exp_name), where exp_name is a directory within results_path, and key is a short name to be used as final name for the exp_name
    test_indexes_path: a json file with test indices
    gold_cluster_col: column name within gold_dataset which contains gold set
    frame_description_file: a json file which contains frame descriptions, represents frame files from FrameNet data
    P: number of final predictions to add to output dataset
    """

    gold_df = create_predictions_dataframe(results_path, {predictor[0]:predictor[1]}, gold_dataset_path, test_indexes_path,
                                          columns=['frameName', 'luName', 'luText', 'luIndex', 'sentence', 'gold_cluster_processed'] )
    
    load_dir_path = f'{results_path}/{predictor[1]}'
    tps, aps, gps = load_tps_aps_gps(load_dir_path)

    gold_df['tps'] = tps
    gold_df['aps'] = aps
    gold_df['gps'] = gps
    gold_df['tps'] = gold_df['tps'].apply(lambda x: x[:P])
    gold_df['aps'] = gold_df['aps'].apply(lambda x: x[:P])
    
   
    random.seed(seed)
    sample = random.sample([i for i in range(0,len(gold_df))], sample_size)
    df = gold_df.iloc[sample].copy().reset_index(drop=True)

    
    def match_noun_inflection(df,
                              predictor_name,
                              gold_cluster_col='gold_cluster_processed',
                              name_col = 'luName',
                              index_col = 'luIndex',
                              text_col = 'luText'):

        columns=['frameName','frame_description', 'gold_terms','original_term','sentence', 'match gold']
        new_df = pd.DataFrame(columns=columns)

        for gd, fn, luName, luText, luIndex, sentence, predictions, match_gold in zip(df[gold_cluster_col], df['frameName'], df[name_col], df[text_col], df[index_col], df['sentence'], df[predictor_name], df['tps']):

            inflections = getAllInflections(luName)
            postag = None
            inflected_predictions = predictions
        
            for key,form in inflections.items():
                if form[0] == luText.lower():
                    postag = key
                    if postag in set(['NN', 'NNS', 'NNP', 'NNPS']):
                        break
            all_lemmas = {p:getAllLemmas(p) for p in predictions}
            if postag:
                # there can be errors with lemmatization with Pattern for words used as both nouns and verbs like (used, reading), which can lead to invalid inflections
                lemmas = [lemma['NOUN'][0] if 'NOUN' in lemma.keys() else p for p,lemma in all_lemmas.items()]
                inflected_predictions = [getInflection(p, tag=postag)[0] for p in lemmas]
            else:
                print('cannot inflect')
                print(luName, luText, inflections)
        # --------------------------------------------------------------- to check and correct (if possible) if lemmas do not match predictions
            lemmas = [lemma['NOUN'][0] if 'NOUN' in lemma.keys() else p for p,lemma in all_lemmas.items()]
            if set(lemmas) - set(predictions) != set():
                if verbose: 
                    print('*** discrepancy in lemmas and preds ***')
                    print(set(lemmas) - set(predictions) )
                    print(f'{fn}-{luName} ----> {luText},{postag}')
                    print(f'predictions:{predictions}')
                    print(f'lemmas:     {lemmas}')
                    print(f'inflected:  {inflected_predictions}')
        # ---------------------------------------------------------------        

            temp_df=pd.DataFrame()
            temp_df['sentence'] = [f'{sentence[:luIndex[0][0]]}<{p}>{sentence[luIndex[0][1]+1:]}' for p in inflected_predictions]
            temp_df['frameName'] = [f'{fn}' for p in predictions]
            temp_df['frame_description'] = [frame_info[fn]['definition'],'','','','','','','','','']
            temp_df['gold_terms'] = [f'{gd}' for p in predictions]

            temp_df['original_term'] = [f'{luText}' for p in predictions]
            temp_df['match gold'] = [1 if g else 0 for g in match_gold]
            new_df = pd.concat([new_df,temp_df])

        new_df['gold_terms'] = new_df['gold_terms'].apply(lambda x: eval(x))
        new_df['gold_terms'] = new_df['gold_terms'].apply(lambda x: ','.join(sorted(x)))

        #             
        new_df

        return new_df
    
    
    new_df = match_noun_inflection(df, predictor[0])
    return df, new_df
    
def create_roles_dataset(results_path,
                            predictor,
                            # save_path,
                            gold_dataset_path='../workdir/data/swr_gold_dataset.pkl',
                            test_indexes_path='../workdir/data/swr_gold_dataset_test_split.json',
                            gold_cluster_col='gold_cluster_patternlemmatized', 
                            frame_description_file='workdir/framenet_data/frame_info.json',
                            P=10,
                            sample_size=50,
                            seed=11,
                            verbose=False):
    """
    results_path: a directory with subdirectories that contains predictions
    predictor: a tuple (key,exp_name), where exp_name is a directory within results_path, and key is a short name to be used as final name for the exp_name
    test_indexes_path: a json file with test indices
    gold_cluster_col: column name within gold_dataset which contains gold set
    frame_description_file: a json file which contains frame descriptions, represents frame files from FrameNet data
    P: number of final predictions to add to output dataset
    """

    gold_df = create_predictions_dataframe(results_path, {predictor[0]:predictor[1]}, gold_dataset_path, test_indexes_path,
                                          columns=['frameName', 'feName', 'feText', 'feIndex', 'sentence', 'gold_cluster_patternlemmatized'] )
    
    load_dir_path = f'{results_path}/{predictor[1]}'
    tps, aps, gps = load_tps_aps_gps(load_dir_path)

    gold_df['tps'] = tps
    gold_df['aps'] = aps
    gold_df['gps'] = gps
    gold_df['tps'] = gold_df['tps'].apply(lambda x: x[:P])
    gold_df['aps'] = gold_df['aps'].apply(lambda x: x[:P])
    
   
    random.seed(seed)
    sample = random.sample([i for i in range(0,len(gold_df))], sample_size)
    df = gold_df.iloc[sample].copy().reset_index(drop=True)

    
    def match_role_inflection(df,
                              predictor_name,
                              gold_cluster_col='gold_cluster_patternlemmatized',
                              name_col = 'feName',
                              index_col = 'feIndex',
                              text_col = 'feText'):

        columns=['frameName','frame_description', 'gold_terms','original_term','sentence', 'match gold']
        new_df = pd.DataFrame(columns=columns)

        for gd, fn, luName, luText, luIndex, sentence, predictions, match_gold in zip(df[gold_cluster_col], df['frameName'], df[name_col], df[text_col], df[index_col], df['sentence'], df[predictor_name], df['tps']):
        
            possible_lemmas = list(itertools.chain(*getAllLemmas(luText).values()))
            inflections = getAllInflections(luText)
            if inflections == {} and possible_lemmas != []:
                inflections = getAllInflections(possible_lemmas[0])
            if verbose: print('--------------------------------')
            if verbose: print(luText, ':\t', inflections)

            inflected_predictions = predictions
            # match case
            inflected_predictions = [w.capitalize() if capitalize_rule(luText, luIndex[0], w) else w for w in inflected_predictions]

            postag = None
            need_inflection = False
            possible_postags = set(inflections.keys())
#             print(f'possible_postags:{possible_postags}')
            if possible_postags - set(['VB','VBD','VBG','VBN','VBP','VBZ','MD', 'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']) == set():
                need_inflection = True
                for key,form in inflections.items():
                    if form[0] == luText.lower():
                        postag = key            
                        break
 
            all_lemmas = {p:getAllLemmas(p) for p in predictions}
            if postag:
                if verbose: print('\n',postag)
                for k,v in all_lemmas.items(): print(k, ':\t', v)
                # there can be errors with lemmatization with Pattern for words used as both nouns and verbs like (used, reading), which can lead to invalid inflections
                lemmas = [lemma['NOUN'][0] if 'NOUN' in lemma.keys() else p for p,lemma in all_lemmas.items()]
                inflected_predictions = [getInflection(p, tag=postag)[0] for p in lemmas]

            else:
                print('cannot inflect')
                print(luText, inflections)

            
            if verbose: print(inflected_predictions)

            temp_df=pd.DataFrame()
            temp_df['sentence'] = [f'{sentence[:luIndex[0][0]]}<{p}>{sentence[luIndex[0][1]+1:]}' for p in inflected_predictions]
            temp_df['frameName'] = [f'{fn}' for p in predictions]
            temp_df['frame_description'] = [frame_info[fn]['definition'],'','','','','','','','','']
            temp_df['feName'] = [f"{luName}: {frameroles_info[fn]['FEs'][luName]['definition']}" ,'','','','','','','','','']
            temp_df['gold_terms'] = [f'{gd}' for p in predictions]

            temp_df['original_term'] = [f'{luText}' for p in predictions]
            temp_df['match gold'] = [1 if g else 0 for g in match_gold]

            new_df = pd.concat([new_df,temp_df])
        new_df['gold_terms'] = new_df['gold_terms'].apply(lambda x: eval(x))
        new_df['gold_terms'] = new_df['gold_terms'].apply(lambda x: ','.join(sorted(x)))

        new_df
        # tinkled, tapering,children

        return new_df
    
    
    new_df = match_role_inflection(df, predictor[0])
    return df, new_df

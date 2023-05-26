import pandas as pd
import re
from ordered_set import OrderedSet 
from pathlib import Path
import string
from sklearn.model_selection import train_test_split
import json
import pickle
import os
from .io_util import read_file, write_file
from .lemmatize_util import lemma
# ------------------------------------------------
def not_quoted(word):
    return word== remove_quotes(word)


def remove_quotes(word):
    """remove single, double and white space characters from start and end of word"""
    regex_s =  r"(^[\"\'\s?]*)"
    regex_e =  r"([\"\'\s?]*$)" 
    
    word = re.sub(regex_s, '', word)
    word = re.sub(regex_e, '', word) 
    
    return word


def luIndex_toList(text):
    "convert string-list of string-tuples from dataframe column to python-list of integer-tuples"
    """
    >>> to_list([(2, 5), (6, 10)])
    [(2, 5), (6, 10)]
    >>> to_list([('2', '5'), ('6', '10')])
    [(2, 5), (6, 10)
    """
    
    text=text.replace("\'","")
    e = r"\(([0-9]+), ?([0-9]+)\)"
    tokens = re.findall(e, text)
    tokens = [(int(pair[0]), int(pair[1])) for pair in tokens]
    
    return tokens


def strList_toList(text):
    "convert string-list from dataframe column to python list"
    """
    >>> to_list(['word', 'word'])
    [word, word]
    
    """

    tokens = text[1:-1].split(",")
    tokens =[remove_quotes(tok) for tok in tokens]

    return tokens
    
# ------------------------------------------- pre-process fn_verbs
# 1. if verb contains a token in (), remove the ()
# 2. if verb contains a token in [], remove the [token]

def remove_brackets(verb):
    v = verb
    if '(' in verb or '[' in verb:
        
#         parentheses = r"\(.*?\)"

        brackets = r"\[.*?\]"
        v = re.sub(brackets, "", v)
        
        s, e = v.find('('), v.find(')')
        if s == -1: return [v.strip()]
        sub = v[s:e+1]
        
        bfr, sub, aft = v.partition(sub)
        
        sub = sub.replace('(', '')
        sub = sub.replace(')', '')
        
        tokens = sub.split('/')
            
        v = []
        for t in tokens:
            v.append(''.join([bfr, t, aft]))  
            
        v = [t.strip() for t in v]
                    
        return v
    
    else: return [verb]

def substitute_pronouns(verb):
    """
    one's --> my, your
    oneself -->myself, yourself
    """
    vs = [verb]
    
    if "one's" in verb:
        
        bfr, pron, aft = verb.partition("one's")
        pronouns = ['my', 'your', 'her', 'his', 'their', 'our']
        for p in pronouns:
            vs.append(''.join([bfr, p, aft]))
        
    if 'oneself' in verb:    
        bfr, pron, aft = verb.partition("oneself")
        pronouns = ['myself', 'yourself', 'herself', 'himself', 'themselves', 'ourselves']
    
        for p in pronouns:
            vs.append(''.join([bfr, p, aft]))
    
    return vs

def preprocess_fn_verbs(verbs):
    if type(verbs) is not list:
        verbs = strList_toList(verbs)
    
    # 1. if verbs contain .v annotations    
    verbs = [v.split('.v')[0] for v in verbs]   
    # 2. handle verbs with () and []
    L1 = []
    for v in verbs:
        L1.extend(remove_brackets(v)) #
    
    # 3. handle pronoun forms
    L2 = []
    for v in L1:
        L2.extend(substitute_pronouns(v))
        
        
    return list(OrderedSet(L2))  
# -------------------------------------------
def expand_contractions(text):
    contractions_dict = {
    r"([\s?]*'re)": " are",
    r"([\s?]*'m)": " am",
    r"([\s?]*'ve)": " have",
    r"([\s?]*'ll)": " will",
    r"(ai[\s?]*n't)": "is not",
    r"(sha[\s?]*n't)": "shall not",
    r"(ca[\s?]*n't)": "can not",
    r"(wo[\s?]*n't)": "will not", # will not or would not

    r"([\s?]*n't)": " not",
    r"([\s?]*'d)": " had", # had or would
    r"([\s?]*'s)": " is", # is or 's (posessive)
    
    r"(y[\s?]*'all)": "you all",
    r"(y[\s?]*'alls)": "you alls",
    r"(how[\s?]*'d)": "how did",
    r"(how[\s?]*'d[\s?]*'y)": "how do you",
 
    r"(y[\s?]*'cause)": "because",
    r"(y[\s?]*'alls)": "you alls",    
    }

    newtext = text
    for k,v in contractions_dict.items():
        newtext = re.sub(k, v, newtext)
    return newtext
# ------------------------------------------- just consider first K predictions
def cutoff_atK(predictions, K):

    if type(predictions.iloc[0]) is not list: 
        predictions = predictions.apply(strList_toList)

    predictions = predictions.apply(lambda x: x[:K])
    return predictions    

# -------------------------------------------
def extract_framenetVocabulary(fn_verb_cluster_file, preprocess=True):
    
    df = read_file(fn_verb_cluster_file)
    if type(df['fn_verbs_withoutchild'].iloc[0]) is not list: 
        df['fn_verbs_withoutchild'] = df['fn_verbs_withoutchild'].apply(strList_toList)
     
    if type(df['fn_verbs_withchild'].iloc[0]) is not list: 
        df['fn_verbs_withchild'] = df['fn_verbs_withchild'].apply(strList_toList)
    
    vocab = set()
    for verbs in df['fn_verbs_withoutchild']:
        for v in verbs:
            vocab.add(v.split('.v')[0])

    for verbs in df['fn_verbs_withchild']:
        for v in verbs:
            vocab.add(v.split('.v')[0])
    
    vocab = list(vocab)
    if preprocess:
        vocab= preprocess_fn_verbs(vocab)
    
    vocab.sort()
    return vocab

# ---------------------------------------------------------- FN Vocabulary
def _remove_OOVpredictions(Ps, V):
    """"P is lemmatized list of predictions"""
    P = OrderedSet(Ps)
    inVocPreds = list(P.intersection(V))

    return inVocPreds 



def remove_OOVpredictions(predictions, V):
        
    """ retain all P whose lemma is present in V, sorted by order of P"""

    if type(predictions.iloc[0]) is not list: 
        predictions = predictions.apply(strList_toList)

    predictions = predictions.apply(lambda Ps: _remove_OOVpredictions(Ps, V))
    min_k = min([len(x) for x in predictions])
    print('min # of prediction in fn_vocabulary', min_k)
    return predictions


# ---------------------------------------------------------- BERT and DT interection
def filter_withDT(bert_file, dt_file):
    """ take the intersection of BERT predictions with DT similar terms"""
    
    df = read_file(bert_file)
    dt = read_file(dt_file)

    df['bert_terms'] = df['predictions']
    df['dt_terms'] = dt['predictions']
    df['predictions'] = intersect(df['bert_terms'], df['dt_terms'])

    return df

# assuming if intersection is empty then column2 would be considered, assuming its DT
def intersect(column1, column2):
    """if intersection is empty then consider column2"""
    if type(column1.iloc[0]) is not list: 
        column1 = column1.apply(strList_toList)
    
    if type(column2.iloc[0]) is not list: 
        column2 = column2.apply(strList_toList)

    
    column3 = []
    for L1, L2 in zip(column1, column2):
        
        s1 = OrderedSet(L1)
        s2 = OrderedSet(L2)

        s3 = s1.intersection(s2)
        if len(s3) == 0:
            s3 = s2
            
        column3.append(list(s3))   
 
    return column3
    
# ---------------------------------------------------------- Rank framewise predictions 

def get_predictions_freq(frame_predictions):

    prediction_freq = {}
    
    for ps in frame_predictions:
        for p in ps:
            if p in prediction_freq.keys():
                prediction_freq[p] = prediction_freq[p]+1
            else:
                prediction_freq[p] = 1
        
    return prediction_freq



def rank_framewise(df, frame_col='frameName', predictions_col='predictions', drop_duplicates=True): 
    df = df.copy()
    if type(df[predictions_col].iloc[0]) is not list: 
            df[predictions_col] = df[predictions_col].apply(strList_toList)

    df['ranked_predictions'] = [[] for i in range(len(df))]

    for frame in df[frame_col].unique():

        frame_predictions = df[df[frame_col]==frame][predictions_col]

        prediction_freq = get_predictions_freq(frame_predictions)
       
        ranks_df = pd.DataFrame(prediction_freq.items(),  columns=['key', 'value'])
        ranks_df = ranks_df.sort_values(by='value', ascending=False)


        ranked_predictions = list(ranks_df['key'])
        
        df.loc[df[frame_col]==frame,'ranked_predictions'] = str(ranked_predictions)
    
    if  drop_duplicates:
        df = df.drop_duplicates(subset=['frameName']).copy()

    df[predictions_col] = df['ranked_predictions'].apply(strList_toList)
    df = df.drop(columns=['ranked_predictions'])
    return df

# -------------------------------------------------
def min_predictions(predictions):
    min_np = 1000
    min2 = min_np
    ps = []
    row = 0
    for i, preds in enumerate(predictions):
        np = len(preds)
        if np<min_np:
#             ps = preds
#             row = i
#             min2 = min_np
            min_np = np
            
    return min_np 

# -------------------------------------------------
def generate_random_token_masks(bpe_tokens):
    return np.array([np.random.randint(0, len(bpe_tokens[i])) 
                     for i in range(len(bpe_tokens))])




def make_test_dev_split(dataset_path, result_path, test_size, 
                        dataset_size=None, name=None):
    if dataset_path is not None:
        with open(dataset_path, 'rb') as f:
            dataset_size = len(pickle.load(f))
        
        if name is None:
            name = dataset_path.split('/')[-1]
    
    train, dev = train_test_split(range(dataset_size), test_size=test_size)
    test = train

    test = sorted(test)
    dev = sorted(dev)

    print(f'{name}_dev_split:', len(dev))
    print(f'{name}_test_split:', len(test))

    with open(os.path.join(result_path, f'{name}_dev_split.json'), 'w') as f:
        json.dump(dev, f)

    with open(os.path.join(result_path, f'{name}_test_split.json'), 'w') as f:
        json.dump(test, f)

    return dev, test
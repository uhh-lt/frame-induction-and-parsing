import re
import string
from tqdm import tqdm as tqdm
from ordered_set import OrderedSet 
from itertools import chain
import itertools
from lemminflect import getInflection, getAllInflections, getAllLemmas
import multiprocessing

from .lemmatize_util import lemma, pattern_lemma, pattern_postag, pattern_parse, nltk_lemma, nltk_poslemma, nltk_postag

    
# same as resources/noun_stopwords
NOUN_STOPWORDS=['about', 'above', 'after', 'again', 'against', 'all', 'also', 'an', 'and', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'but', 'by', 'down', 'during', 'each', 'for', 'from', 'further', 'here',  'how', 'if', 'in', 'into', 'just', 'most',  'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'out', 'over', 'own', 'same', 'so', 'such', 'than', 'the',  'then', 'there',  'through', 'to', 'too', 'under', 'until', 'up', 'very', 'when', 'where',  'while', 'with',  
                
 'any','both', 'either', 'each', 'which', 'who', 'whom', 'whose', 'neither', 'none', 'one', 'some',
'everybody', 'anybody', 'nobody', 
                
'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves' ,'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',  'hers', 'herself', 'it', 'its', 'itself', 'they', 'their', 'theirs', 'them', 'themselves',
'this', 'these', 'that', 'those',
                
'is', 'am', 'are', 'can', 'could', 'do', 'did', 'does', 'has', 'have', 'having', 'had', 'shall', 'should', 'would', 'must', 'may', 'was', 'were',
                
"'m", "'r", "'s", "'re", "'ll", "'d", "'ve", "won't",
"isn't", "aren't", "can't", "couldn't", "don't", "doesn't", "didn't", "hasn't", "haven't", "hadn't", "mustn't", "shouldn't" , "shan't", "wasn't", "weren't" ,"wouldn't"              
                
'ah', 'aren', 'arent', 'com', 'eg', 'en', 'et', 'et-al', 'etc', 'ex', 'hi', 'id', 'ie', 'inc', "'ml", 'na', 'nd', 'oh', 'ok', 'okay', 're','thi','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '']


# same as resources/role_stopwords2
ROLE_STOPWORDS=['also', 'an', 'and', 'as', 'at', 'be', 'been','being','by', 'for', 'if', 'in', 'no', 'nor', 'not','of', 'off', 'on', 'or', 'so', 'the','to', 'too', 'with',
 
'about', 'above', 'after', 'again', 'against', 'all',  'because', 'before', 'below', 'between', 'but', 'down', 'during', 'each', 'from', 'further', 'here', 'how', 'into', 'just', 'more', 'most', 'now', 'once', 'only', 'out', 'over', 'own', 'same', 'such', 'than', 'then', 'there', 'through', 'under', 'until', 'very' , 'when', 'where', 'while', 'why',
    
'ah','aren', 'arent', 'com', 'eg', 'en', 'et', 'et-al', 'etc', 'ex', 'hi', 'id', 'ie', 'inc', "'ml", 'na', 'nd', 'oh', 'ok', 'okay', 're','thi','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '',
]

NOUN_POSTAGS = set(['NN', 'NNS','NNP','NNPS'])
VERB_POSTAGS =         set(['VB','VBD','VBG','VBN','VBP','VBZ','MD'])


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
EN_CH = set(string.ascii_letters + string.digits + string.punctuation)

# pre-process multiword predictions, convert list of predictions to space joined sequence
def unify(preds):
    result = []
    for pred in preds:
        if type(pred) is list:
            result.append(' '.join(pred))
        else:
            result.append(pred)
    return result


def is_noisyword(text):
    special_tokens = set(["[unk]", "<unk>"])
    text = text.strip()
    if text == '': return True
    if text.lower() in special_tokens: 
        return True #UNKOWN token
    if "#" in text: 
        return True # subword
    
    for character in text: # filter out any word which contains any non-english character
        if not character in EN_CH:
            return True
    
    for character in text: # filter out any word which only contains punctuation or whitespaces
        if character in string.ascii_letters or character in string.digits:
            return False
       
    return True

def _remove_2letterwords(predictions):
    ps = [x for x in predictions if len(x)!=2]
#     ps = [x for x in ps if not x in ['we', 'it', 'my', 'he', 'me', 'up', 'do']]
#     ps = [x for x in ps if not re.match(r'.*\d+.*', x)] #don't remove digits
    
    return ps #[p for p in predictions if not p in ps]


def remove_noisywords(phrases):
    return [phrase for phrase in phrases if not is_noisyword(phrase)]

def _remove_digits(predictions):
    return [p for p in predictions if not re.match(r'.*\d+.*', p)]

def remove_duplicates(phrases):
    return list(OrderedSet(phrases))

def lemmatize(phrases, postags=None): # just to have same signature
    result = []
    for phrase in phrases:
        result.append(pattern_lemma(phrase))
    return result

# for fsp task, postag is used
def lemmatize_nltk(phrases, postags=None):
    result = []
    if not postags:
        for phrase in phrases:
            result.append(nltk_lemma(phrase))
        
    else:
        if type(postags) is str: 
            postags = [postags]*len(phrases)
        for phrase, pos in zip(phrases, postags):
            result.append(nltk_poslemma(phrase, pos))
        
    return result

def lemmatize_lemminflect(phrases, postags=None):
    result = []
    if type(postags) is str or postags is None: 
        postags = [postags]*len(phrases)
        
    for phrase, pos in itertools.zip_longest(phrases, postags, fillvalue=None):
        lemmas = get_allLemmas(phrase, pos)
        if len(lemmas)>1 and phrase in lemmas:
            lemmas = [l for l in lemmas if l!=phrase]
        if lemmas == []:
            raise Exception(f'no lemma found for --- {phrase} !!!')
        result.append(lemmas[0])
    return result

# all possible postags for current word form
def leminflect_postag(word):
    inflections = getAllInflections(word)
    # try with lemma
    if inflections == {}:
        inflections = getAllInflections(get_allLemmas(word)[0])
    postags = []
    for k,v in inflections.items():
        if v[0] == word:
            postags.append(k)
            
    return postags

#  all possible postags for all word forms 
def get_allPOStags(word):
    all_tags = list(OrderedSet(getAllInflections(word).keys()))
    # try with lemma
    if all_tags == []: 
        all_tags = list(OrderedSet(getAllInflections(get_allLemmas(word)[0]).keys()))
    return all_tags

def get_allLemmas(word, postag=None):
    all_lemmas = getAllLemmas(word)
    if postag in all_lemmas.keys(): 
        all_lemmas = list(OrderedSet(all_lemmas[postag]))
    else: all_lemmas = list(OrderedSet(chain(*all_lemmas.values())))
    if all_lemmas == []: all_lemmas = [word]
    
    return all_lemmas


def get_allInflections(words):
    if type(words) is str:
        words = [words]
        
    all_forms = [{word:list(chain(*getAllInflections(word).values()))} for word in words]
    all_forms = [values if values else [word] for entry in all_forms for word, values in entry.items()]
    inflected_forms = list(OrderedSet(chain(*all_forms)))
    return inflected_forms

def _match_inflection(predictions, postag):
    inflected_forms = []
    if postag in UTAGS.keys():
        for pred in predictions:
            inflected_form = getInflection(pred, tag=postag)
            if inflected_form:
                inflected_forms.append(inflected_form[0])
            else:
                inflected_forms.append(pred)
        return inflected_forms
    return predictions

def _correct_predictions(gold_cluster, predictions):
    if gold_cluster:
        return list(OrderedSet(predictions) & OrderedSet(gold_cluster))
    return predictions

def _correct_nolemma_predictions(gold_cluster, predictions, preds_postags=None):
    if gold_cluster:
        return [pred for pred in predictions if pattern_lemma(pred) in OrderedSet(gold_cluster)] 
#         return list(OrderedSet(predictions) & OrderedSet(get_allInflections(gold_cluster)))
    return predictions

def _correct_nolemma_predictions_nltk(gold_cluster, predictions, preds_postag):
    if gold_cluster:
        return [pred for pred in predictions if nltk_poslemma(pred, preds_postag).lower() in OrderedSet(gold_cluster)] 
    return predictions

def _correct_nolemma_predictions_lemminflect(gold_cluster, predictions, preds_postag=None):
    if gold_cluster:
        return [pred for pred in predictions if OrderedSet(get_allLemmas(pred).lower()) & OrderedSet(gold_cluster) != set()] 
    return predictions

# ---------------------------------------- list of lists
def unify_predictions(predictions):
    return [unify(e) for e in tqdm(predictions)]

def remove_digits(predictions):
    return [_remove_digits(p) for p in tqdm(predictions) ]

def clean_noisy_predictions(predictions):
    return [remove_noisywords(e) for e in tqdm(predictions)]                        

def remove_2letterwords(predictions):
    return [_remove_2letterwords(e) for e in tqdm(predictions)]

def remove_duplicate_predictions(predictions):
    return [remove_duplicates(e) for e in tqdm(predictions)]

def remove_stopwords(predictions, vocabulary):
    return [[pred for pred in predictions[i] if not pred.lower() in vocabulary]
            for i in tqdm(range(len(predictions)))]

def remove_vocab(predictions, vocabulary):
    return [[pred for pred in predictions[i] if not pred.lower() in vocabulary]
            for i in tqdm(range(len(predictions)))]

def filter_vocab(predictions, vocabulary):
    return [[pred for pred in predictions[i] if pred.lower() in vocabulary or set(pred.lower().split(' ')).intersection(vocabulary)!=set([])]
            for i in tqdm(range(len(predictions)))]
    
# # -------------------------------------------- lemminflect based    
# in current implementation, some words can have [] postags from lemminflect 
def remove_verbPOStags_lemminflect(predictions):
    ps = set(chain(*predictions))
    all_tags = {p:set(leminflect_postag(p)) for p in ps}
    all_tags = {p:set([UTAGS[p] for p in all_tags[p]]) for p in ps}
    return [[pred for pred in predictions[i] if not all_tags[pred] == {'VERB'}] # remove if VERB is the only POS
            for i in tqdm(range(len(predictions)))]

def remove_nounPOStags_lemminflect(predictions):
    ps = set(chain(*predictions))
    all_tags = {p:set(leminflect_postag(p)) for p in ps}
    all_tags = {p:set([UTAGS[p] for p in all_tags[p]]) for p in ps}

    return [[pred for pred in predictions[i] if not all_tags[pred] == {'NOUN'}] # remove if NOUN is the only POS
            for i in tqdm(range(len(predictions)))]

def filter_verbPOStags_lemminflect(predictions):
    ps = set(chain(*predictions))
    all_tags = {p:set(leminflect_postag(p)) for p in ps}
    return [[pred for pred in predictions[i] if all_tags[pred] & VERB_POSTAGS != set() ] # keep if any POS is VERB
            for i in tqdm(range(len(predictions)))]

def filter_nounPOStags_lemminflect(predictions):
    ps = set(chain(*predictions))
    all_tags = {p:set(leminflect_postag(p)) for p in ps}
    return [[pred for pred in predictions[i] if all_tags[pred] & NOUN_POSTAGS != set() ] # keep if any POS is NOUN
            for i in tqdm(range(len(predictions)))]

def match_POStags_lemminflect(predictions, postags):
    ps = set(chain(*predictions))
    all_tags = {p:set(leminflect_postag(p)) for p in ps}
    return [[pred for pred in predictions[i] if postags[i] in tags[pred]]
             for i in tqdm(range(len(predictions)))]

# -------------------------------------------- pattern.en based
def remove_verbPOStags(predictions):
    ps = set(chain(*predictions))  
    tags = {p:pattern_postag(p).split('-')[0] for p in ps} # NNP-LOC etc
    return [[pred for pred in predictions[i] if not tags[pred] in VERB_POSTAGS ]
            for i in tqdm(range(len(predictions)))]

def remove_nounPOStags(predictions):
    ps = set(chain(*predictions))
    tags = {p:pattern_postag(p).split('-')[0] for p in ps} # NNP-LOC etc
    return [[pred for pred in predictions[i] if not all_tags[pred] in NOUN_POSTAGS]
            for i in tqdm(range(len(predictions)))]

def filter_verbPOStags(predictions):
    ps = set(chain(*predictions))
    tags = {p:pattern_postag(p).split('-')[0] for p in ps} # NNP-LOC etc
    return [[pred for pred in predictions[i] if tags[pred] in VERB_POSTAGS ]
            for i in tqdm(range(len(predictions)))]

def filter_nounPOStags(predictions):
    ps = set(chain(*predictions))
    tags = {p:pattern_postag(p).split('-')[0] for p in ps} # NNP-LOC etc
    return [[pred for pred in predictions[i] if tags[pred] in NOUN_POSTAGS ]
            for i in tqdm(range(len(predictions)))]

def match_POStags(predictions, postags):
    ps = set(chain(*predictions))
    tags = {p:pattern_postag(p) for p in ps}
    return [[pred for pred in predictions[i] if tags[pred] == postags[i]]
             for i in tqdm(range(len(predictions)))]
# --------------------------------------------
def filter_verbPOStags_nltk(predictions):
    ps = set(chain(*predictions))
    tags = {p:nltk_postag(p).split('-')[0] for p in ps} # NNP-LOC etc
    return [[pred for pred in predictions[i] if tags[pred] in VERB_POSTAGS ]
            for i in tqdm(range(len(predictions)))]

def filter_nounPOStags_nltk(predictions):
    ps = set(chain(*predictions))
    tags = {p:nltk_postag(p).split('-')[0] for p in ps} # NNP-LOC etc
    return [[pred for pred in predictions[i] if tags[pred] in NOUN_POSTAGS ]
            for i in tqdm(range(len(predictions)))]

def match_POStags_nltk(predictions, postags):
    ps = set(chain(*predictions))
    tags = {p:nltk_postag(p) for p in ps}
    return [[pred for pred in predictions[i] if tags[pred] == postags[i]]
             for i in tqdm(range(len(predictions)))]

# --------------------------------------------   
def lemmatize_predictions(predictions, postags=None, n_jobs=16):
    with multiprocessing.Pool(n_jobs) as pool:
            predictions = pool.map(lemmatize, tqdm(predictions), chunksize=200)
#     predictions = [lemmatize(e) for e in tqdm(predictions)]
    return predictions

def lemmatize_predictions_nltk(predictions, postags=None, n_jobs=16):
    if postags:
        args = [*zip(predictions, postags)]
        with multiprocessing.Pool(n_jobs) as pool:
            predictions = pool.starmap(lemmatize_nltk, tqdm(args), chunksize=200)
    else:
        with multiprocessing.Pool(n_jobs) as pool:
            predictions = pool.map(lemmatize_nltk, tqdm(predictions), chunksize=200)
    
    return predictions

def lemmatize_predictions_lemminflect(predictions, postags=None, n_jobs=16):
    with multiprocessing.Pool(n_jobs) as pool:
            predictions = pool.map(lemmatize_lemminflect, tqdm(predictions), chunksize=200)
    return predictions

def match_inflection(predictions, postags, n_jobs=16):
    return [_match_inflection(predictions[i], postags[i]) for i in tqdm(range(len(predictions)))]


def remove_seedword(words, predictions):
    """ remove target word itself from predictions"""
    return [[pred for pred in predictions[i] if pred.lower() != words[i].lower()]
            for i in tqdm(range(len(predictions)))]
    
def remove_nolemma_seedword(words, predictions):
    """ remove target word itself from predictions"""
    inflected_forms = [OrderedSet([w.lower() for w in get_allInflections(get_allLemmas(word))]) for word in words]
    return [[pred for pred in predictions[i] if not pred.lower() in inflected_forms[i]]
            for i in tqdm(range(len(predictions)))]

def correct_predictions(gold_cluster, predictions):
    return [_correct_predictions(gold_cluster[i], predictions[i]) for i in tqdm(range(len(predictions)))]
    
def correct_nolemma_predictions(gold_cluster, predictions, preds_postags=None):
    return [_correct_nolemma_predictions(gold_cluster[i], predictions[i]) for i in tqdm(range(len(predictions)))]

def correct_nolemma_predictions_nltk(gold_cluster, predictions, preds_postags):
    return [_correct_nolemma_predictions_nltk(gold_cluster[i], predictions[i], preds_postags[i]) for i in tqdm(range(len(predictions)))]

def correct_nolemma_predictions_lemminflect(gold_cluster, predictions, preds_postags=None):
    return [_correct_nolemma_predictions_lemminflect(gold_cluster[i], predictions[i], preds_postags[i]) for i in tqdm(range(len(predictions)))]
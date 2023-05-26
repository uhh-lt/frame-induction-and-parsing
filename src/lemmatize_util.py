from pattern.en import lemma as _pattern_lemma
from pattern.en import parse, tag
# import spacy
# from spacy.lang import en
# ------------------------------------------------ spacy
# spacy_poslemmatizer = en.EnglishDefaults.create_lemmatizer()
# spacy_model = spacy.load('en', disable=['parser', 'ner'])


def spacy_lemma(text):
    
    doc = spacy_model(text)
    lemmas = " ".join([token.lemma_ if token.lemma_ !="-PRON-" else token.lower_ for token in doc])
    tags = " ".join([token.tag_ for token in doc])    
    
    lemmas = lemmas.replace(" - ", "-") #spacy lemmatize twenty-five —> twenty - five 
    return (lemmas, tags)[0]


def spacy_poslemma(text, pos='verb'):
        
    return spacy_poslemmatizer(text, pos)[0]



def spacy_postag(text):
    doc = spacy_model(text)
    tags = " ".join([token.tag_  for token in doc])
    return tags



def spacy_parse(text):
    """ for (lemma, postag) """
    doc = spacy_model(text)
    tokens=[(token.lemma_ if token.lemma_ !="-PRON-" else token.lower_, token.tag_) for token in doc]
#     tokens = tokens.replace(" - ", "-") #spacy lemmatize twenty-five —> twenty - five 
    return tokens


# ------------------------------------------------ pattern https://github.com/clips/pattern    
def pattern_lemma(text):
   
    tokens = text.split(' ')
    lemmas = " ".join([_pattern_lemma(token) if token!="" else token for token in tokens])
    return lemmas


def pattern_postag(text):
    tagged_text = tag(text, tokenize = False)
    tags = " ".join([token[1].split('-')[0] for token in tagged_text])
    return tags



# parse(string, 
#    tokenize = True,         # Split punctuation marks from words?
#        tags = True,         # Parse part-of-speech tags? (NN, JJ, ...)
#      chunks = True,         # Parse chunks? (NP, VP, PNP, ...)
#   relations = False,        # Parse chunk relations? (-SBJ, -OBJ, ...)
#     lemmata = False,        # Parse lemmata? (ate => eat)
#    encoding = 'utf-8'       # Input string encoding.
#      tagset = None)         # Penn Treebank II (default) or UNIVERSAL.

def pattern_parse(text):
    tagged_text = parse(text, tokenize = False, lemmata=True, tags=True, chunks=False)
    docs = tagged_text.split()        
    output = [(token[0], token[1].split('-')[0], token[2]) for doc in docs for token in doc]
    return output



# #  ------------------------------------------------ nltk
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet 

nltk_lemmatizer = WordNetLemmatizer() 

def nltk_poslemma(text, pos='n'):
    if pos == None: pos='n'
    return nltk_lemmatizer.lemmatize(text, pos)
    
        
def get_wordnet_pos(nltk_tag):
    nltk_tag = nltk_tag.upper()
    if nltk_tag.startswith('J'): return wordnet.ADJ
    elif nltk_tag.startswith('V'): return wordnet.VERB
    elif nltk_tag.startswith('N'): return wordnet.NOUN
    elif nltk_tag.startswith('R'): return wordnet.ADV
    # if no mapping found
    return wordnet.NOUN

def nltk_lemma(text, pos=None):
        
    tokens =  text.split(' ') #nltk.word_tokenize(text)
    if pos == None:
        tags = nltk.pos_tag(tokens)
    else:
        tags = pos
        if type(tags) is str:
            tags = tags.split(' ')
            
    lemmas = [nltk_lemmatizer.lemmatize(token_tag[0], get_wordnet_pos(token_tag[1])) for token_tag in tags]
    lemmas = ' '.join(lemmas)
    return lemmas

def nltk_postag(text):
    tokens =  text.split(' ') #nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in tags]
    return ' '.join(tags)
 
def nltk_parse(text):
    tokens = text.split(' ')
    postags = [p[1] for p in nltk.pos_tag(tokens)]
    lemmas = [nltk_poslemma(token, 'v') if pos.startswith("V") else nltk_poslemma(token) for token, pos in zip(tokens, postags)]
    output = [(tokens[i], postags[i].split('-')[0], lemmas[i]) for i in range(len(tokens))]
    return output

# ------------------------------------------------
def lemma(text, lemmatizer='pattern'):
   
    if lemmatizer == 'pattern': 
        return pattern_lemma(text)
    else: 
        return spacy_lemma(text)

def postag(text, parser='pattern'):
    if parser == 'pattern': 
        return pattern_postag(text)
    else: 
        return spacy_postag(text)

# ------------------------------------------------

# Fixing pattern bug
from .lemmatize_util import lemma
try:
    lemma('Progress', lemmatizer='pattern')
except:
    pass
# -------------------------------------------------

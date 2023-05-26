import sys
import copy
from sesame.dataio import read_conll
from sesame.conll09 import CoNLL09Example, CoNLL09Element
from sesame.sentence import Sentence


# ID    FORM    LEMMA   PLEMMA  POS PPOS    SENT#   PFEAT   HEAD    PHEAD   DEPREL  PDEPREL LU  FRAME ROLE
# 0     1       2       3       4   5       6       7       8       9       10      11      12  13    14

def get_frameName(example):
    elements = example._elements
    
    for e in elements:
        if e.is_pred:
            return e._frame

def get_luName(example):
    elements = example._elements
    
    for e in elements:
        if e.is_pred:
            return e._lu

def get_feName(example):
    elements = example._elements
    
    for e in elements:
        if e.is_arg:
            return e._role

def conll_to_sentence(example):
    elements = example._elements
    tokens = []
    
    for i, e in enumerate(elements):
        tokens.append(e._form)
    return ' '.join(tokens)


def example_to_str(example):
    lines = []
    for i, e in enumerate(example._elements):
        lines.append(element_to_str(e))
    
    return '\n'.join(lines)+'\n'

def element_to_str(element):

    text = element.get_str()
    tokens = text.replace('\n', '').split('\t')
    tokens[1] = element._form
    return '\t'.join(tokens)
        
    
def sort_sentencewise(examples):
    examples = copy.deepcopy(examples)
    examples.sort(key=lambda x: x.sent_num, reverse=False)              
    return examples


def reset_sentnum(examples, sentences):
    examples = copy.deepcopy(examples)
    all_sentences =  [conll_to_sentence(example) for example in examples] 
    sent_number = {sentences[i]:i for i in range(len(sentences))}
    new_examples = []
    
    for example, sent in zip(examples, all_sentences):
        i = sent_number[sent]
        elements = example._elements
        for j, e in enumerate(elements):
            elements[j].sent_num = i
        new_examples.append(example) 
                    
    return new_examples



def is_multitokenLU(example):
    elements = example._elements
    preds = []
    for e in elements:
        if e.is_pred:
            preds.append(e)
            text = element_to_str(e)
            luName = text.replace('\n', '').split('\t')[12]    
            
    multi=  len(luName.split(' '))>1 or len(preds)>1
    return multi
               # annotations may be missed for some tokens, thus counting tokens with e.is_pred is error prone              


    
def substitute(example, eid, word, plemma=None, ppos=None):
    """
    replace given original(word) of example with predicted(word), assuming it's a single token
    """
    example = copy.deepcopy(example)
    elements = example._elements
    if plemma is None: lpemma = word
    for i, e in enumerate(elements):
        if e.id==eid: 
            text = element_to_str(e)# e.get_str()
            tokens = text.replace('\n', '').split('\t')
            tokens[1] = word
            tokens[3] = plemma
            tokens[5] = ppos if ppos else tokens[5]
            if e.is_pred:
                tokens[12] = f'{plemma}.{e._lupos}'
            elements[i] =CoNLL09Element('\t'.join(tokens))
            
    return CoNLL09Example(Sentence(None, elements), elements)


def reconstruct_example(example, words, plemmas, ppos, lu_lemma=None):
    """
    reconstruct whole example number of tokens remain same
    """
    example = copy.deepcopy(example)
    elements = example._elements
    for i, e in enumerate(elements):
        text = element_to_str(e)
        tokens = text.replace('\n', '').split('\t')
        tokens[1] = words[i]
        tokens[3] = plemmas[i] 
        tokens[5] = ppos[i] 
        if e.is_pred and lu_lemma:
            if e._form==words[i]: continue
            tokens[12] = f'{lu_lemma}.{e._lupos}'
        elements[i] =CoNLL09Element('\t'.join(tokens))
            
    return CoNLL09Example(Sentence(None, elements), elements)

# ==========
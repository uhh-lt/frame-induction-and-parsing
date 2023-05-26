import fire
import re
import os
import sys
from pathlib import Path
import multiprocessing as mp
from ordered_set import OrderedSet
import gensim
from pandarallel import pandarallel
import pandas as pd
from tqdm import tqdm
tqdm().pandas()

from pandarallel import pandarallel
from .extract_deps import dep_rels
from .predict import preprocess_tagged_text
# ------------
use_tokenizer = False

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

from nltk.parse.corenlp import CoreNLPDependencyParser

depparse_model={"depparse.model":"edu/stanford/nlp/models/parser/nndep/english_SD.gz",
               "tokenize.whitespace": str(True)}

# """
# # latest version of stanfordCoreNLP throw error if sentences contain % sign, with some versions they work fine, like 3.9.2 
# # can't remember, if previously used version was really 3.5.1, as it does not contain CoreNLPServer class (may be, used a different version or missing something completely here)
# """

def _preprocess_tagged_text(row):
    sent = row['masked_sent'].replace('\xa0',' ')#.replace('__%__','__percent__').replace('%', '')
# TODO
#     regex = re.compile('__\\$[^ ].*?[^ ]__')
#     sent = re.sub(regex, "__$__", sent)
#     regex = re.compile('__\\£[^ ].*?[^ ]__')
#     sent = re.sub(regex, "__£__", sent)
    # numbers starting with $ or £ are sometimes very large, does not apear in vocab
    try:
        row['masked_position'], row['bpe_tokens'] = preprocess_tagged_text(sent, use_tokenizer)
    except Exception as e:
        print('\nERROR...\n')
        print(sent)
        print(row['masked_sent'])
        print(e)
    return row

def _context_elements(row):
    try:
        return context_elements(row['bpe_tokens'], row['masked_position'])
    except Exception as e:
        print('\nERROR...\n')
        print(row['masked_sent'], '\t--\t',row['masked_position'])
        print(e)
        raise e#return []
    
def context_elements(sentence, masked_position):
    sentence, target_word = sentence.to_original_text().lower(), sentence[masked_position].text.lower()
    
    parse_tree,  = dependency_parser.raw_parse(sentence, properties = depparse_model)
    conll_output = parse_tree.to_conll(style=10)
#     print(conll_output)
    collapsed_deprel = dep_rels(conll_output, word_vocab, format='conll')
    context_elements = []
    for ind, word, context in collapsed_deprel:
        if ind == masked_position+1 :
            context_elements.append(context) 

    context_elements = [c for c in context_elements if not c.startswith('ROOT')]
    return context_elements

# not used after PaM
def process_sents(sentences, masked_position):
    if type(masked_position) is not list:
        masked_position = [masked_position]

    words = []
    for sent, masked_pos in zip(sentences, masked_position):
        words.append(sent[masked_pos].text)
        
    
    sentences = [sent.to_original_text() for sent in sentences]
    
    return sentences, words

def main(input_file, output_file, jobs=16, port=9000):
    
    print('input file:', input_file)  
    print('output file:', output_file)    
    print('stanford server port:', port)    

    df = pd.read_pickle(input_file)
    
    print('input records:', len(df))
    
    pandarallel.initialize(nb_workers=jobs, progress_bar=True)
    df = df.parallel_apply(_preprocess_tagged_text, axis=1)
    global dependency_parser
    dependency_parser = CoreNLPDependencyParser(url=f'http://localhost:{port}')

    pandarallel.initialize(nb_workers=jobs, progress_bar=True)
    df['C'] = df.parallel_apply(_context_elements, axis=1)
    df.to_pickle(f'{output_file}')
    print('Done...')
    

if __name__ == '__main__':
    fire.Fire(main)
    
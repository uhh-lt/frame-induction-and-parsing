import os
from itertools import cycle
from ordered_set import OrderedSet
import copy
import shutil
import pandas as pd 
from tqdm import tqdm
import sys
import numpy as np
import json
from sesame.dataio import read_conll
from lexsub.conll_helper import is_multitokenLU
from lexsub.conll_helper import get_frameName, conll_to_sentence
from src.predict import get_masked_tokens_from_tagged_text

# ----------------------------------------------------------------------
DEV_FILE = 'fn1.7.dev.syntaxnet.conll'
TEST_FILE = 'fn1.7.test.syntaxnet.conll'
TRAIN_FILE = 'fn1.7.fulltext.train.syntaxnet.conll'

INPUT_DIR = '../../workdir/framenet'
# ============================ candidate frames

LU_MAPS = {'fn1.5': f'{INPUT_DIR}/fn1.5/lu_map.json',
           'fn1.7': f'{INPUT_DIR}/fn1.7/lu_map.json',}

 # -----------------------------------
    
def to_csv(examples):
    
    df = pd.DataFrame(columns = ['frameName', 'sentence', 'luName', 'luText', 'luIndex'])    
    config_dicts = []
    row = 0
    for example in tqdm(examples):    
#         print(conll_to_sentence(example))
        frame = get_frameName(example)
        elements = example._elements
        for i, e in enumerate(elements):
            if e.is_pred and not is_multitokenLU(example):
                                    
                tagged_text = ' '.join([ex._form if ex!=e else f'__{ex._form}__' for ex in elements ])
                index, clean_text  = get_masked_tokens_from_tagged_text(tagged_text)
                index = [(p[0], p[1]-1) for p in index]
                identifier = frame
                seedword = e._form
                df.loc[row] = [identifier, clean_text, '.'.join([e._lu, e._lupos]), seedword, index ]
                row += 1
                break
                                                                 
        
    print(f'{len(df)} single token lu examples were processed...')

    return df

            
def get_frameMap(frames):
    frame2id = {}
    id2frame = {}

    frames = OrderedSet(frames)
    for i, fr in enumerate(frames):
        frame2id[fr] = i
        id2frame[i] = fr
    return frame2id, id2frame


def get_indices(text, indices):
    
    """get start, end index of word to be masked"""
    # for single and multi-token but contigous chunk
    start, end = indices[0], indices[1]

    oldstr = text[int(start): int(end)+1]
    """halfcarried --> half carried"""
    bfr = ''
    aft = ''
    if int(start)!=0:
        if text[int(start) -1] != ' ': bfr = ' '
    if len(text) != int(end)+1: 
        if text[int(end) +1] != ' ': aft = ' '    

    return oldstr, start, end, bfr, aft


def mask_indices(text, indices):
    """mask single word verb"""
    for i, index in enumerate(indices):
        oldstr, start, end, bfr, aft = get_indices(text, (index[0]+i*6, index[1]+i*6))
        text = '{}{}[s]{}[e]{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])
    
    return text 

def mask_lu(row):
    """mask lus"""

    text = row['sentence']
    index = row['luIndex']
    try:
        text = mask_indices(text, index)
        return text 
    except Exception as ex:
        print(text)
        print(index)
        raise ex

def _convert_conll_to_csv(input_file, output_dir, file_prefix='mc', fn='fn1.7'):
    
    examples = read_conll(input_file)[0]
    input_df =  to_csv(examples)
    
    
    luFrameMap = {}
    with open(LU_MAPS[fn],'r') as fp:
        luFrameMap = json.load(fp) 
        
    input_df['sentenceT'] = input_df.apply(mask_lu, axis=1)
    
    data = input_df[['frameName', 'luName', 'luText', 'sentence', 'sentenceT']].copy()
    data['labelFrame'] = data['frameName']
    data['candidateFrames'] = data['luName'].apply(lambda x: luFrameMap[x])
    
    
    data['label'] = data['frameName']
    data['candidateFrames'] = data['candidateFrames'].apply(lambda x: ','.join(x))    
    
    sent_columns = ['sentence', 'sentenceT']
    for col_a in sent_columns:
        
        df = pd.DataFrame()
        
        df["text"] = data[col_a]
        df["labels"] = data['label']
        df['luText'] = data['luText']
        df['candidates'] = data['candidateFrames']

        df.to_csv(f'{output_dir}/{file_prefix}_ic-{col_a}.csv', index=False)

def convert_conll_to_csv(input_dir, output_dir):
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
    print('output_dir:', output_dir)  
    _convert_conll_to_csv(f'{input_dir}/{TRAIN_FILE}', output_dir, file_prefix = 'ft-train_mc')
    _convert_conll_to_csv(f'{input_dir}/{TEST_FILE}', output_dir, file_prefix = 'ft-test_mc')
    _convert_conll_to_csv(f'{input_dir}/{DEV_FILE}', output_dir, file_prefix = 'ft-dev_mc')

# ===========================================
import fire
if __name__ == '__main__':
    fire.Fire(convert_conll_to_csv)
    
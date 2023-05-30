import fire 
import pickle
import pandas as pd 
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

from gensim.models import KeyedVectors, FastText

from .masked_token_predictor_bert import MaskedTokenPredictorBert
from .masked_token_predictor_dsm import MaskedTokenPredictorDsm
from .masked_token_predictor_dt import MaskedTokenPredictorDt
from .masked_token_predictor_melamudv2 import MaskedTokenPredictorMelamud2 

from .masked_token_predictor_melamud import MaskedTokenPredictorMelamud 
from .util import remove_quotes, luIndex_toList, strList_toList
from .bert_utils import load_bert_model_and_tokenizer
from .predict import analyze_tagged_text

from .logger import create_logger


def load_predictor(model_type, 
                   mask_in_multiunit=False,
                   do_lemmatize=False, 
                   do_lowercase=True, 
                   max_len=250,
                   n_jobs=1 
                   ):
    predictor = None
    if model_type.startswith('bert'):
        model, bpe_tokenizer = load_bert_model_and_tokenizer(model_config=model_type, cache_dir='workdir/cache')
        predictor = MaskedTokenPredictorBert(model, bpe_tokenizer, max_len=max_len, 
                                             mask_in_multiunit=mask_in_multiunit)
    elif model_type.startswith('dsm'):
        model_path = model_type.split('+')[1]
        is_binary = os.path.splitext(model_path)[1] != '.txt'
        model = KeyedVectors.load_word2vec_format(model_path, binary=is_binary)
        predictor = MaskedTokenPredictorDsm(model, n_jobs=n_jobs, 
                                            do_lemmatize=do_lemmatize,
                                            do_lowercase=do_lowercase)
    elif model_type.startswith('fasttext'):
        from gensim.models.wrappers import FastText
        model_path = model_type.split('+')[1]
        model = FastText.load_fasttext_format(model_path, encoding = 'utf8')
        predictor = MaskedTokenPredictorDsm(model, n_jobs=n_jobs, 
                                            do_lemmatize=do_lemmatize,
                                            do_lowercase=do_lowercase)
        
    elif model_type.startswith('dt'):
        dt_path = model_type.split('+')[1]
        dt = pd.read_csv(dt_path, index_col='w1') # TODO: there can be just w1
        dt.dropna(inplace=True)
        predictor = MaskedTokenPredictorDt(dt, n_jobs=n_jobs, 
                                           do_lemmatize=do_lemmatize, 
                                           do_lowercase=do_lowercase)
    
    elif model_type.startswith('melamud2'):
        word_model_path = model_type.split('+')[1]
        context_model_path = model_type.split('+')[2]
        word_model = KeyedVectors.load(word_model_path, mmap='r')
        context_model = KeyedVectors.load(context_model_path, mmap='r')
        predictor = MaskedTokenPredictorMelamud2((word_model, context_model), n_jobs=n_jobs, 
                                            do_lemmatize=do_lemmatize,
                                            do_lowercase=do_lowercase,
                                            metric = other_params['metric'])
    
    elif model_type.startswith('melamud'):
        word_model_path = model_type.split('+')[1]
        context_model_path = model_type.split('+')[2]
        

        fname1 = f'{Path(word_model_path).parent}/word_vectors.model'
        fname2 = f'{Path(word_model_path).parent}/context_vectors.model'

        if not Path(fname1).exists():
            word_model=gensim.models.KeyedVectors.load_word2vec_format(word_model_path, binary=False)
            # filename should be absolute, to save model
            word_model.save(os.path.abspath(os.path.expanduser(os.path.expandvars(fname1))))

        if not Path(fname2).exists():
            context_model=gensim.models.KeyedVectors.load_word2vec_format(context_model_path, binary=False)
            context_model.save(os.path.abspath(os.path.expanduser(os.path.expandvars(fname2))))

        word_model = gensim.models.KeyedVectors.load(fname1, mmap='r')
        context_model = gensim.models.KeyedVectors.load(fname2, mmap='r')
        
        predictor = MaskedTokenPredictorMelamud((word_model, context_model), n_jobs=n_jobs, 
                                            do_lemmatize=do_lemmatize,
                                            do_lowercase=do_lowercase,
                                            metric = other_params['metric'])
        
        
    else:
        raise(ValueError('Unknown predictor'))
        
    return  predictor  

def main(model_type, dataset, proc_column, result_dir, n_top=100, 
         n_units=1, n_tokens=1, mask_token=True, batch_size=10, 
         max_multiunit=10, do_lemmatize=False, do_lowercase=True, 
         n_jobs=1, mask_in_multiunit=False, use_tokenizer=False, other_params=None):

    if type(other_params) is str:
        
        other_params = other_params.split(',')
        other_params = [tuple(kv.split(':')) for kv in other_params]
        other_params = {kv[0]:kv[1] for kv in other_params}

    if type(n_tokens) is not tuple:
        n_tokens = [n_tokens]
    else:
        n_tokens = list(n_tokens)
    if n_tokens!=1:
        print('n_tokens:',n_tokens)
        
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
        
    logger = create_logger('usem-experiments', result_dir)

    max_len = 250
    if 'swr' in dataset or 'mwr' in dataset:
        max_len = 300
    if 'swn' in dataset or 'mwn' in dataset:
        max_len = 350    

    predictor = load_predictor(model_type, 
                   mask_in_multiunit=mask_in_multiunit,
                   do_lemmatize=do_lemmatize, 
                   do_lowercase=do_lowercase, 
                   max_len=max_len,
                   n_jobs=n_jobs)
    

    import subprocess

    proc = subprocess.Popen('pwd', stdout=subprocess.PIPE)
    output = proc.stdout.read()
    print(output)
    
    with open(dataset, 'rb') as f:
        df = pickle.load(f)
    
    logger.info('Processing...')
         
#     predictions, scores, masked_words = analyze_tagged_text(df[proc_column].tolist(), predictor, use_tokenizer, 
#                                                         n_top=n_top, 
#                                                         n_tokens=n_tokens,
#                                                         n_units=n_units,
#                                                         progress_bar=tqdm,
#                                                         mask_token=mask_token,
#                                                         max_multiunit=max_multiunit,
#                                                         batch_size=batch_size)

#     to cater melamud where C is pre-computed
    predictions, scores, masked_words = analyze_tagged_text(df[proc_column].tolist(), predictor, use_tokenizer, 
                                                        n_top=n_top, 
                                                        n_tokens=n_tokens,
                                                        n_units=n_units,
                                                        progress_bar=tqdm,
                                                        mask_token=mask_token,
                                                        max_multiunit=max_multiunit,
                                                        batch_size=batch_size,
                                                        contexts=df['C'].tolist() if 'C' in df.columns else None)
    logger.info('Done.')

    logger.info(f'Saving results to {result_dir} ...')
    with open(os.path.join(result_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)

    with open(os.path.join(result_dir, 'scores.pkl'), 'wb') as f:
        pickle.dump(scores, f)

    with open(os.path.join(result_dir, 'masked_words.pkl'), 'wb') as f:
        pickle.dump(masked_words, f)
    logger.info('Done.')


if __name__ == '__main__':
    fire.Fire(main)
    
import pandas as pd
import multiprocessing as mp
import numpy as np
from .lemmatize_util import lemma


class MaskedTokenPredictorDt:
    def __init__(self, dt, n_jobs=1, do_lemmatize=True, do_lowercase=True):  
        self._dt = dt
        self._n_jobs = n_jobs # TODO: use it to speed up processing
        self._do_lemmatize = do_lemmatize
        self._do_lowercase = do_lowercase
            
    def __call__(self, sentences, masked_position, n_top, *args, **kwargs):
        if type(masked_position) is not list:
            bpe_tokens = [bpe_tokens]
            masked_position = [masked_position]
            
        words = []
        for sent, masked_pos in zip(sentences, masked_position):
            words.append(sent[masked_pos].text)
        
        predictions = []
        scores = []
        
        if self._do_lowercase:
            words = [word.lower() for word in words]

        if self._do_lemmatize:
            words = [lemma(word, lemmatizer='pattern') for word in words]

        similarity_dict = self.similar_words(list(set(words)), n_top)
        
        for w in words:
            terms, sim_scores = similarity_dict[w]
            
            predictions.append(terms)
            scores.append(sim_scores)

        return predictions, scores
    
    
    def similar_words(self, words, n_top):
        print('# of words to lookup:', len(words))    
        similarity_dict = {}
        for word in words:
            try:
                tmp = self._dt.loc[[word], :].dropna()                    
                similar_words = tmp.sort_values(by='score', ascending=False)
                top_n = similar_words.iloc[:n_top]
                terms = top_n['w2'].tolist()
                scores = top_n['score'].tolist()
            
            except KeyError:
                terms = []
                scores = []

            similarity_dict[word] = (terms, scores)


        return similarity_dict

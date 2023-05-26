from gensim.models import KeyedVectors
from ordered_set import OrderedSet
from nltk.parse.corenlp import CoreNLPDependencyParser
import os
import numpy as np
import  scipy
from .lemmatize_util import lemma
from .extract_deps import dep_rels
# os.environ['CLASSPATH'] = "/home/anwar/Melamud_lexicalSubstitutions/stanford-corenlp-full-2015-01-29/stanford-corenlp-3.5.1.jar:/home/anwar/Melamud_lexicalSubstitutions/stanford-corenlp-full-2015-01-29/stanford-corenlp-3.5.1-models.jar"

class MaskedTokenPredictorMelamud2:
    def __init__(self, model, n_jobs=10, do_lemmatize=False, do_lowercase=True, whitespace_tokenizer=True, metric='Mult'):
        self.word_model = model[0]
        self.context_model = model[1]
        self._n_jobs = n_jobs # TODO: use this to speed up processing
        self._do_lemmatize = do_lemmatize
        self._do_lowercase = do_lowercase    
#         self.depparse_model={"depparse.model":"edu/stanford/nlp/models/parser/nndep/english_SD.gz"}
#         self.depparse_model["tokenize.whitespace"] = str(whitespace_tokenizer)
#         self.dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')      
        self.word_vocab = OrderedSet(self.word_model.vocab.keys())
        self.context_vocab = OrderedSet(self.context_model.vocab.keys())      
        if metric.lower() == "add": 
            self.metric = self.Add
        if metric.lower() == "baladd": 
            self.metric = self.BalAdd
        if metric.lower() == "mult": 
            self.metric = self.Mult
        if metric.lower() == "balmult": 
            self.metric = self.BalMult
        print('metric:',self.metric.__name__)            
    def __call__(self, sentences, masked_position, n_top, *args, **kwargs):
        
        predictions = []
        scores = []
        
        if type(masked_position) is not list:
            masked_position = [masked_position]
            
        words = []
        for sent, masked_pos in zip(sentences, masked_position):
            words.append(sent[masked_pos].text)
        
#         sentences = [sent.__str__() for sent in sentences]
        sentences = [sent.to_original_text() for sent in sentences]

        
        if self._do_lowercase:
            words = [word.lower() for word in words]
            sentences = [sent.lower() for sent in sentences]
            
        Cs = kwargs['Cs']  
#         Cs = [self.context_elements(sent, word) for sent, word in zip(sentences, words)]
        res = [self.similar_words(sent, word, C, n_top) for sent, word, C in zip(sentences, words, Cs)]
                
        predictions = [pair[0] for pair in res]
        scores = [pair[1] for pair in res]
        return predictions, scores
    
#     def context_elements(self, sentence, target_word):

#         parse_tree,  = self.dependency_parser.raw_parse(sentence, properties = self.depparse_model)
#         conll_output = parse_tree.to_conll(style=10)
#         collapsed_deprel = dep_rels(conll_output, self.word_vocab, format='conll')
#         context_elements = []
#         for i, (word, context) in enumerate(collapsed_deprel):
#             if word in target_word:
#                 context_elements.append(context) 

#         context_elements = [c for c in context_elements if not c.startswith('ROOT')]
#         return context_elements

    def similar_words(self, sentence, word, C, n_top = 200):

        if not word in self.word_vocab: return ([], [])
        
    #     drop contexts which are not present in context model vocab
        C = [c for c in C if c in self.context_vocab]
        if C == []: return ([], [])
            
        similarity_dict = self.metric(word, C)

        res = sorted(similarity_dict.items(), key=lambda items: items[1], reverse=True)
        res = res[:n_top]
        terms = [e[0] for e in res]
        scores = [e[1] for e in res]
        return (terms, scores)
    
    def Add(self, w, C):

        S_vecs = np.array([self.word_model[s] for s in self.word_vocab])
        t_vec = np.array([self.word_model[w]])
        target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
        C_vecs = np.array([self.context_model[c] for c in C ])

        context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
        context_sim = context_sim.sum(axis=1)
        context_sim = context_sim.reshape(context_sim.shape[0],1)

        combine_sim = (target_sim + context_sim) / (len(C) + 1)
        similarity_dict = dict({k:v[0] for k,v in zip(self.word_vocab,combine_sim)})

        return similarity_dict

    def BalAdd(self, w, C):

        S_vecs = np.array([self.word_model[s] for s in self.word_vocab])
        t_vec = np.array([self.word_model[w]])
        C_vecs = np.array([self.context_model[c] for c in C ])

        target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
        target_sim = target_sim * len(C)

        context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
        context_sim = context_sim.sum(axis=1)
        context_sim = context_sim.reshape(context_sim.shape[0],1)

        combine_sim = (target_sim + context_sim) / (len(C) * 2)

        similarity_dict = dict({k:v[0] for k,v in zip(self.word_vocab,combine_sim)})

        return similarity_dict

    def Mult(self, w, C):

        S_vecs = np.array([self.word_model[s] for s in self.word_vocab])
        t_vec = np.array([self.word_model[w]])
        C_vecs = np.array([self.context_model[c] for c in C ])

        target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
        target_sim = (target_sim + 1)/2

        context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
        context_sim = (context_sim + 1) / 2
        context_sim = context_sim.prod(axis=1)
        context_sim = context_sim.reshape(context_sim.shape[0],1)  

        combine_sim = (target_sim * context_sim) ** (1 /(len(C) +1))

        similarity_dict = dict({k:v[0] for k,v in zip(self.word_vocab,combine_sim)})
        return similarity_dict


    def BalMult(self, w, C):

        S_vecs = np.array([self.word_model[s] for s in self.word_vocab])
        t_vec = np.array([self.word_model[w]])
        C_vecs = np.array([self.context_model[c] for c in C ])

        target_sim = 1 - scipy.spatial.distance.cdist(S_vecs, t_vec, 'cosine')
        target_sim = (target_sim + 1)/2
        target_sim = target_sim ** len(C)

        context_sim = 1 - scipy.spatial.distance.cdist(S_vecs, C_vecs, 'cosine')
        context_sim = (context_sim + 1) / 2
        context_sim = context_sim.prod(axis=1)
        context_sim = context_sim.reshape(context_sim.shape[0],1)

        combine_sim = (target_sim * context_sim) ** (1 /(len(C) *2))

        similarity_dict = dict({k:v[0] for k,v in zip(self.word_vocab,combine_sim)})
        return similarity_dict
from gensim.models import KeyedVectors
from ordered_set import OrderedSet
from nltk.parse.corenlp import CoreNLPDependencyParser
import os
import numpy as np
import  scipy
from .lemmatize_util import lemma
from .extract_deps import dep_rels
os.environ['CLASSPATH'] = "/home/anwar/Melamud_lexicalSubstitutions/stanford-corenlp-full-2015-01-29/stanford-corenlp-3.5.1.jar:/home/anwar/Melamud_lexicalSubstitutions/stanford-corenlp-full-2015-01-29/stanford-corenlp-3.5.1-models.jar"

class MaskedTokenPredictorMelamud:
    def __init__(self, model, n_jobs=10, do_lemmatize=False, do_lowercase=True, whitespace_tokenizer=True, metric='Mult'):
        self.word_model = model[0]
        self.context_model = model[1]
        self._n_jobs = n_jobs # TODO: use this to speed up processing
        self._do_lemmatize = do_lemmatize
        self._do_lowercase = do_lowercase
        self.depparse_model={"depparse.model":"edu/stanford/nlp/models/parser/nndep/english_SD.gz"}
        self.depparse_model["tokenize.whitespace"] = str(whitespace_tokenizer)
        self.dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')
        self.word_vocab = OrderedSet(self.word_model.vocab.keys())
        self.context_vocab = OrderedSet(self.context_model.vocab.keys())
        print('metric:', metric)
        if metric.lower() == "add": 
            self.metric = self.Add
        if metric.lower() == "baladd": 
            self.metric = self.BalAdd
        if metric.lower() == "mult": 
            self.metric = self.Mult
        if metric.lower() == "balmult": 
            self.metric = self.BalMult
        
        
    def __call__(self, sentences, masked_position, n_top, *args, **kwargs):
        if type(masked_position) is not list:
            bpe_tokens = [bpe_tokens]
            masked_position = [masked_position]
            
        words = []
        for sent, masked_pos in zip(sentences, masked_position):
            words.append(sent[masked_pos].text)
        
        predictions = []
        scores = []

#         sentences = [sent.__str__() for sent in sentences]
        sentences = [sent.to_original_text() for sent in sentences]

        if self._do_lowercase:
            words = [word.lower() for word in words]
            sentences = [sent.lower() for sent in sentences]
            
#         if self._do_lemmatize:
#             words = [lemma(word, lemmatizer='pattern') for word in words]
        
        for sent, word in zip(sentences, words):
        
            if  word in self.word_vocab:
                terms, sim_scores = self.similar_words(sent, word, n_top)
            
                predictions.append(terms)
                scores.append(sim_scores)
                
            else:
                predictions.append([])
                scores.append([])
                
        return predictions, scores
    
    
    def similar_words(self, sentence, word, n_top):

        if not word in self.word_vocab: return ([], [])

        C = self.context_elements(sentence, word)
    #     drop contexts which are not present in context model vocab
        C = [c for c in C if c in self.context_vocab]
        if C == []: return ([], [])

        similarity_dict = self.metric(word, C)

        res = sorted(similarity_dict.items(), key=lambda items: items[1], reverse=True)
        res = res[:n_top]
        terms = [e[0] for e in res]
        scores = [e[1] for e in res]

        return (terms, scores)
    
    def cos(self, vec1, vec2):
        return 1 -  scipy.spatial.distance.cosine(vec1, vec2)

    def pcos(self, vec1, vec2):
        return (self.cos(vec1, vec2)+1)/2


    def Add(self, w, C):
        t = self.word_model[w]
        similarity_dict = {}
        for s in self.word_vocab:    
            target_sim = self.cos(t, self.word_model[s])
            context_sim = np.array([self.cos(self.word_model[s], self.context_model[c]) for c in C ])
            combine_sim = (target_sim + context_sim.sum()) / (len(C) + 1)
            similarity_dict[s] = combine_sim       

        return similarity_dict

    def BalAdd(self, w, C):
        t = self.word_model[w]
        similarity_dict = {}
        for s in self.word_vocab:   
            target_sim = self.cos(t, self.word_model[s]) * len(C)
            context_sim = np.array([self.cos(self.word_model[s], self.context_model[c]) for c in C ])
            combine_sim = (target_sim + context_sim.sum()) / (len(C) * 2)
            similarity_dict[s] = combine_sim

        return similarity_dict

    def Mult(self, w, C):
        t = self.word_model[w]
        similarity_dict = {}
        for s in self.word_vocab:    
            target_sim = self.pcos(t, self.word_model[s])
            context_sim = np.array([self.pcos(self.word_model[s], self.context_model[c]) for c in C ])
            combine_sim = (target_sim * context_sim.prod()) ** (1 /(len(C) + 1))
            similarity_dict[s] = combine_sim

        return similarity_dict  

    def BalMult(self, w, C):
        t = self.word_model[w]
        similarity_dict = {}
        for s in self.word_vocab:    
            target_sim = self.pcos(t, self.word_model[s]) ** len(C)
            context_sim = np.array([self.pcos(self.word_model[s], self.context_model[c])  for c in C ])
            combine_sim = (target_sim * context_sim.prod()) ** (1 /(len(C) * 2))
            similarity_dict[s] = combine_sim

        return similarity_dict        
        

    def context_elements(self, sentence, target_word):

        parse_tree,  = self.dependency_parser.raw_parse(sentence, properties = self.depparse_model)
        conll_output = parse_tree.to_conll(style=10)
        collapsed_deprel = dep_rels(conll_output, self.word_vocab, format='conll')
        context_elements = []
        for i, (word, context) in enumerate(collapsed_deprel):
            if word in target_word:
                context_elements.append(context) 

        context_elements = [c for c in context_elements if not c.startswith('ROOT')]
        return context_elements

    
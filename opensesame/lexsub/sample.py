import sys
import os
import random
import copy
import shutil

from sesame.dataio import read_conll
from sesame.conll09 import CoNLL09Example, CoNLL09Element
from sesame.sentence import Sentence

from lexsub.conll_helper import get_frameName, get_luName, conll_to_sentence, sort_sentencewise, reset_sentnum
from lexsub.conll_helper import element_to_str, example_to_str


dev_file = 'fn1.7.dev.syntaxnet.conll'
test_file = 'fn1.7.test.syntaxnet.conll'
train_file = 'fn1.7.fulltext.train.syntaxnet.conll'



def get_example_dicts(examples):
    
    frame_lus = {}
    lu_examples = {}
    frame_examples = {}
    
    for example in examples:
        elements = example._elements
        frame = get_frameName(example)
        lu = get_luName(example)
                
        if frame in frame_lus.keys():
            if not lu in frame_lus[frame]:
                frame_lus[frame].append(lu)
        else:
            frame_lus[frame] = [lu]

        if frame in frame_examples.keys():
            frame_examples[frame].append(example)
        else:
            frame_examples[frame] = [example]  

        if lu in lu_examples.keys():
            lu_examples[lu].append(example)
        else:
            lu_examples[lu] = [example]

#                 break
                
    return frame_lus, frame_examples, lu_examples

def get_unique_examples_for_frame(examples):
    
    frame_examples = {}
    frame_sents = set()
    
    for example in examples:
        elements = example._elements
        sent = conll_to_sentence(example)
        for e in elements:
            if e.is_pred:
                frame = e._frame
                    
        if not sent in frame_sents:

            if frame in frame_examples.keys():
                frame_examples[frame].append(example)
            else:
                frame_examples[frame] = [example] 

            frame_sents.add(sent)    

                
    return frame_examples

def examples_perSentence_dict(examples):
    
    examples_perKey = {}
    for example in examples:
        key = conll_to_sentence(example)
        if not key in examples_perKey.keys():
            examples_perKey[key] = [example]
        else:
            examples_perKey[key].append(example)
            
    return examples_perKey
            
def examples_perFrame_dict(examples, unique=False):
    
    examples_perKey = {}
    sents = set()
                
    for example in examples:
        key = get_frameName(example)
        sent = conll_to_sentence(example)
        if unique and not sent in sents:
            sents.add(sent)
            if not key in examples_perKey.keys():
                examples_perKey[key] = [example]
            else:
                examples_perKey[key].append(example)
        else:
            if not key in examples_perKey.keys():
                examples_perKey[key] = [example]
            else:
                examples_perKey[key].append(example)    
    print(len(sents))
    return examples_perKey


def examples_perLU_dict(examples, unique=False):
    
    examples_perKey = {}
    sents = set()
    for example in examples:
        key = get_luName(example)
        sent = conll_to_sentence(example)
        if unique and not sent in sents:
            sents.add(sent)
            if not key in examples_perKey.keys():
                examples_perKey[key] = [example]
            else:
                examples_perKey[key].append(example)
        else:
            if not key in examples_perKey.keys():
                examples_perKey[key] = [example]
            else:
                examples_perKey[key].append(example)
            
    return examples_perKey

# ============= filter verb examples
def filter_examples(input_conll_file, output_conll_file, pos='v', syn_type = None):
    
    examples, __, __ = read_conll(input_conll_file, syn_type=syn_type)
    
    sub_examples = []
    missingargs2 = 0
    for example in examples:
        elements = example._elements
        for e in elements:
            if e.is_pred:
                if e._lupos==pos: 
                    if not example in sub_examples:
                        sub_examples.append(example)
                        if example.numargs == 0:
                            missingargs2 += 1
                            
                               
    sentences = []
    for example in sub_examples:
        sent = conll_to_sentence(example)        
        if not sent in sentences:
            sentences.append(sent)
    
    output_dir = '/'.join(output_conll_file.split('/')[:-1])
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in sub_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
        cf.close()
              
    with open(output_conll_file.replace('.conll', '.conll.sents'), "w", encoding="utf-8") as cf:
        for sent in sentences:
            cf.write(sent)
            cf.write('\n')
        cf.close() 

    sys.stderr.write("\nAfter filtering verb examples....\n")
    examples, __, __ = read_conll(output_conll_file, syn_type=syn_type)
    
# --------------------------------------------------------------------
#sample n example per sentence
def sample_perSentence(input_conll_file, output_conll_file, sample_size = 1, syn_type = None):

    examples, missingargs, totalexamples = read_conll(input_conll_file, syn_type=syn_type)

    
    sub_examples = []
    for example in examples:
        if not conll_to_sentence(example) in sentences.keys():
            sub_examples.append(example)
            sentences[conll_to_sentence(example)] = 1
            
        else:
            if sentences[conll_to_sentence(example)] == sample_size: continue
            else:
                sub_examples.append(example)
                sentences[conll_to_sentence(example)] = sentences[conll_to_sentence(example)]+1
    
    
    sub_examples = sort_sentencewise(sub_examples)
            
    output_dir = '/'.join(output_conll_file.split('/')[:-1])
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in sub_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
        cf.close()
        
    input_dir = '/'.join(input_conll_file.split('/')[:-1])

    print(f'Copying dev and test files from {input_dir} to {output_dir}')

    shutil.copyfile(f'{input_dir}/{test_file}', f'{output_dir}/{test_file}')
    shutil.copyfile(f'{input_dir}/{dev_file}', f'{output_dir}/{dev_file}')

        
    frame_lus, frame_examples, lu_examples = get_example_dicts(examples)
   
    print('\n--inputfile stats: \n')
    print('# of input FSPs: ', len(examples))
    print('# of input frames: ', len(frame_lus.keys()))
    print('# of input lu: ', len(lu_examples.keys()))
    
    frame_lus, frame_examples, lu_examples = get_example_dicts(sub_examples)
    
    print('\n--outputfile stats: \n')
    print('# of sampled FSPs: ', len(sub_examples))
    print('# of sampled frames: ', len(frame_lus.keys()))
    print('# of sampled lu: ', len(lu_examples.keys()))
    
    
def random_sample_perSentence(input_conll_file, output_conll_file, sample_size = 1, seed=None, syn_type = None):
    
    random.seed(seed)
    examples, missingargs, totalexamples = read_conll(input_conll_file, syn_type=syn_type)
    examples_perSent = examples_perSentence_dict(examples)
    
    sub_examples = []
    for key, key_examples in examples_perSent.items():
        if sample_size < len(key_examples):
            sub_examples.extend(random.sample(key_examples, sample_size))
        else:
            sub_examples.extend(key_examples)
             
    
    sub_examples = sort_sentencewise(sub_examples)
            
    output_dir = '/'.join(output_conll_file.split('/')[:-1])
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in sub_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
        cf.close()
        
    input_dir = '/'.join(input_conll_file.split('/')[:-1])

    print(f'Copying dev and test files from {input_dir} to {output_dir}')

    shutil.copyfile(f'{input_dir}/{test_file}', f'{output_dir}/{test_file}')
    shutil.copyfile(f'{input_dir}/{dev_file}', f'{output_dir}/{dev_file}')


        
    frame_lus, frame_examples, lu_examples = get_example_dicts(examples)
   
    print('\n--inputfile stats: \n')
    print('# of input FSPs: ', len(examples))
    print('# of input frames: ', len(frame_lus.keys()))
    print('# of input lu: ', len(lu_examples.keys()))
    
    frame_lus, frame_examples, lu_examples = get_example_dicts(sub_examples)
    
    print('\n--outputfile stats: \n')
    print('# of sampled FSPs: ', len(sub_examples))
    print('# of sampled frames: ', len(frame_lus.keys()))
    print('# of sampled lu: ', len(lu_examples.keys()))
    

# --------------------------------------------------------------------

# sample examples per frame
def sample_perFrame(input_conll_file, output_conll_file, sample_size=1, unique=False ,syn_type = None):
    
    
    examples, missingargs, totalexamples = read_conll(input_conll_file, syn_type=syn_type)
    
    frame_lus, frame_examples, lu_examples = get_example_dicts(examples)
   
    
    examples_perFrame = examples_perFrame_dict(examples, unique=unique)
    
    sub_examples = []
    for key, key_examples in examples_perFrame.items():
        sub_examples.extend(key_examples[:sample_size])
                  
            
    sub_examples = sort_sentencewise(sub_examples)
    
                
    output_dir = '/'.join(output_conll_file.split('/')[:-1])
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in sub_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
        cf.close()
        
    
    input_dir = '/'.join(input_conll_file.split('/')[:-1])

    print(f'Copying dev and test files from {input_dir} to {output_dir}')

    shutil.copyfile(f'{input_dir}/{test_file}', f'{output_dir}/{test_file}')
    shutil.copyfile(f'{input_dir}/{dev_file}', f'{output_dir}/{dev_file}')

    
    frame_lus, frame_examples, lu_examples = get_example_dicts(examples)
   
    print('\n--inputfile stats: \n')
    print('# of input FSPs: ', len(examples))
    print('# of input frames: ', len(frame_lus.keys()))
    print('# of input lu: ', len(lu_examples.keys()))
    
    frame_lus, frame_examples, lu_examples = get_example_dicts(sub_examples)
    
    print('\n--outputfile stats: \n')
    print('# of sampled FSPs: ', len(sub_examples))
    print('# of sampled frames: ', len(frame_lus.keys()))
    print('# of sampled lu: ', len(lu_examples.keys()))

    

def random_sample_perFrame(input_conll_file, output_conll_file, sample_size=1, unique=False, seed=None, syn_type = None):
    
    random.seed(seed)
    examples, missingargs, totalexamples = read_conll(input_conll_file, syn_type=syn_type)
    print('unique:', unique)
    examples_perFrame = examples_perFrame_dict(examples, unique=unique)
    
    sub_examples = []
    
    for key, key_examples in examples_perFrame.items():
        if sample_size < len(key_examples):
            sub_examples.extend(random.sample(key_examples, sample_size))
        else:
            sub_examples.extend(key_examples)
             
    
    sub_examples = sort_sentencewise(sub_examples)   
        
    output_dir = '/'.join(output_conll_file.split('/')[:-1])
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in sub_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
        cf.close()
        
    input_dir = '/'.join(input_conll_file.split('/')[:-1])

    print(f'Copying dev and test files from {input_dir} to {output_dir}')

    shutil.copyfile(f'{input_dir}/{test_file}', f'{output_dir}/{test_file}')
    shutil.copyfile(f'{input_dir}/{dev_file}', f'{output_dir}/{dev_file}')


        
    frame_lus, frame_examples, lu_examples = get_example_dicts(examples)
   
    print('\n--inputfile stats: \n')
    print('# of input FSPs: ', len(examples))
    print('# of input frames: ', len(frame_lus.keys()))
    print('# of input lu: ', len(lu_examples.keys()))
    
    frame_lus, frame_examples, lu_examples = get_example_dicts(sub_examples)
    
    print('\n--outputfile stats: \n')
    print('# of sampled FSPs: ', len(sub_examples))
    print('# of sampled frames: ', len(frame_lus.keys()))
    print('# of sampled lu: ', len(lu_examples.keys()))


def random_sample_perLU(input_conll_file, output_conll_file, sample_size=1, unique=False, seed=None, syn_type = None):
    
    random.seed(seed)
    examples, missingargs, totalexamples = read_conll(input_conll_file, syn_type=syn_type)
    
    examples_perLU = examples_perLU_dict(examples, unique=unique)
    
    sub_examples = []
    
    for key, key_examples in examples_perLU.items():
        if sample_size < len(key_examples):
            sub_examples.extend(random.sample(key_examples, sample_size))
        else:
            sub_examples.extend(key_examples)
             
    
    sub_examples = sort_sentencewise(sub_examples)   
        
    output_dir = '/'.join(output_conll_file.split('/')[:-1])
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in sub_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
        cf.close()
        
    input_dir = '/'.join(input_conll_file.split('/')[:-1])

    print(f'Copying dev and test files from {input_dir} to {output_dir}')

    shutil.copyfile(f'{input_dir}/{test_file}', f'{output_dir}/{test_file}')
    shutil.copyfile(f'{input_dir}/{dev_file}', f'{output_dir}/{dev_file}')


        
    frame_lus, frame_examples, lu_examples = get_example_dicts(examples)
   
    print('\n--inputfile stats: \n')
    print('# of input FSPs: ', len(examples))
    print('# of input frames: ', len(frame_lus.keys()))
    print('# of input lu: ', len(lu_examples.keys()))
    
    frame_lus, frame_examples, lu_examples = get_example_dicts(sub_examples)
    
    print('\n--outputfile stats: \n')
    print('# of sampled FSPs: ', len(sub_examples))
    print('# of sampled frames: ', len(frame_lus.keys()))
    print('# of sampled lu: ', len(lu_examples.keys()))

    
    
def random_sample_examples(input_conll_file, output_conll_file, sample_size=0.1, seed=None, syn_type = None):
    
    random.seed(seed)
    examples, missingargs, totalexamples = read_conll(input_conll_file, syn_type=syn_type)
    E = len(examples)
    sample_size = round(sample_size*E)
    sub_examples = random.sample(examples, sample_size)
    
    sub_examples = sort_sentencewise(sub_examples)   
        
    output_dir = '/'.join(output_conll_file.split('/')[:-1])
    if not os.path.exists(output_dir): 
        print(f'creating output_dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_conll_file, "w", encoding="utf-8") as cf:
        for example in sub_examples:
            cf.write(example_to_str(example))
            cf.write('\n')
        cf.close()
        
    input_dir = '/'.join(input_conll_file.split('/')[:-1])

    print(f'Copying dev and test files from {input_dir} to {output_dir}')

    shutil.copyfile(f'{input_dir}/{test_file}', f'{output_dir}/{test_file}')
    shutil.copyfile(f'{input_dir}/{dev_file}', f'{output_dir}/{dev_file}')


        
    frame_lus, frame_examples, lu_examples = get_example_dicts(examples)
   
    print('\n--inputfile stats: \n')
    print('# of input FSPs: ', len(examples))
    print('# of input frames: ', len(frame_lus.keys()))
    print('# of input lu: ', len(lu_examples.keys()))
    
    frame_lus, frame_examples, lu_examples = get_example_dicts(sub_examples)
    
    print('\n--outputfile stats: \n')
    print('# of sampled FSPs: ', len(sub_examples))
    print('# of sampled frames: ', len(frame_lus.keys()))
    print('# of sampled lu: ', len(lu_examples.keys()))
    
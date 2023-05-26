import json
import fire
import os
import sys    
import copy
import shutil
import pandas as pd
from ordered_set import OrderedSet
import pickle 
from sesame.dataio import read_conll
from sesame.conll09 import CoNLL09Example, CoNLL09Element
from sesame.sentence import Sentence

from .augment_conll import augment_conllExamples, generate_newExamples, to_dict
from .conll_helper import get_frameName
from .augment_conll import PROC_FUNCS_OPTIONS, matchgold_predictions

import src
from src.run_predict import load_predictor
# ----------------------------------------------------------------------

DATA_DIR = 'data/open_sesame_v1_data/fn1.7'

DEV_FILE = 'fn1.7.dev.syntaxnet.conll'
TEST_FILE = 'fn1.7.test.syntaxnet.conll'
TRAIN_FILE = 'fn1.7.fulltext.train.syntaxnet.conll'
# -----------------------------------
STYLE = '''
    table, th, td {
        border: 1px solid black;
    }
    th {
        background-color:#8DCDEC ;
        text-align:left;
    }
    td {
        height: 50px;
        vertical-align: bottom;
    }
    caption {
        caption-side: top;
    }
    span.lu {
        color:#FF5733;
    }
    span.role {
        color:#0534D7;
    }
    span.noun {
        color:#07AAC1
    }
    '''
def mask_sentence(seed_example, new_examples, predictions_dict):
    frame = get_frameName(seed_example)
    html_sents = []

    html_tokens = []
    elements = seed_example._elements
    for i, e in enumerate(elements):
        if e.id in predictions_dict.keys():
            if predictions_dict[e.id]['word_type']=='lu_v' or predictions_dict[e.id]['word_type']=='lu_n':
                html_tokens.append(f'{{\\color{{myred}} {e._form} }}\\textsubscript{{{frame}}}')
            if predictions_dict[e.id]['word_type']=='role':
                html_tokens.append(f'{{\\color{{myblue}} {e._form}}}')
            if predictions_dict[e.id]['word_type']=='noun' or predictions_dict[e.id]['word_type']=='ibnoun':
                html_tokens.append(f'{{\\color{{mydeepblue}} {e._form}}}')                       
        else:
            html_tokens.append(f'{e._form}')  

    html_sents.append(' '.join(html_tokens))
    
    
    for example in new_examples:
        elements = example._elements
        html_tokens = []
        for i, e in enumerate(elements):
            if e.id in predictions_dict.keys():
                if predictions_dict[e.id]['word_type']=='lu_v' or predictions_dict[e.id]['word_type']=='lu_n':
                    html_tokens.append(f'{{\\color{{myred}} {e._form} }}\\textsubscript{{{frame}}}')
                if predictions_dict[e.id]['word_type']=='role':
                    html_tokens.append(f'{{\\color{{myblue}} {e._form}}}')
                if predictions_dict[e.id]['word_type']=='noun' or predictions_dict[e.id]['word_type']=='ibnoun':
                    html_tokens.append(f'{{\\color{{mydeepblue}} {e._form}}}')                       
            else:
                html_tokens.append(f'{e._form}')  
        
        html_sents.append(' '.join(html_tokens))
    return html_sents


def html_sentence(seed_example, new_examples, predictions_dict):
    frame = get_frameName(seed_example)
    html_sents = []

    elements = seed_example._elements
    html_tokens = ['<p>']
    for i, e in enumerate(elements):
        if e.id in predictions_dict.keys():
            if predictions_dict[e.id]['word_type']=='lu_v' or predictions_dict[e.id]['word_type']=='lu_n':
                html_tokens.append(f'<span class="lu">{e._form}</span>')
                html_tokens.append(f'<b><sub>{frame}</b></sub>')
            if predictions_dict[e.id]['word_type']=='role':
                html_tokens.append(f'<span class="role">{e._form}</span>')
            if predictions_dict[e.id]['word_type']=='noun' or predictions_dict[e.id]['word_type']=='ibnoun':
                html_tokens.append(f'<span class="noun">{e._form}</span>')                       
        else:
            html_tokens.append(f'{e._form}')  

    html_tokens.append('</p>')
        
    html_sents.append(' '.join(html_tokens))
    
    
    for example in new_examples:
        elements = example._elements
        html_tokens = ['<p>']
        for i, e in enumerate(elements):
            if e.id in predictions_dict.keys():
                if predictions_dict[e.id]['word_type']=='lu_v' or predictions_dict[e.id]['word_type']=='lu_n':
                    html_tokens.append(f'<span class="lu">{e._form}</span>')
                if predictions_dict[e.id]['word_type']=='role':
                    html_tokens.append(f'<span class="role">{e._form}</span>')
                if predictions_dict[e.id]['word_type']=='noun' or predictions_dict[e.id]['word_type']=='ibnoun':
                    html_tokens.append(f'<span class="noun">{e._form}</span>')                       
            else:
                html_tokens.append(f'{e._form}')  

        html_tokens.append('</p>')
        
        html_sents.append(' '.join(html_tokens))
    return html_sents
 

NOTATIONS = {"lu":"lexical units were expanded",
            "roles":"roles expanded",
            "nouns-xx":"xx percentage of nouns have been expanded",
            "ooa":"outside-of-annotations, expanded nouns were not part of any multi token role",
            "lemma":"final substitues were lemmatized",
            "gold":"substitutes were filtered for gold answers"
                    }

def write_html_table(df, output_file, caption='Examples of Expansions', notations={}, N=2):
    with open(output_file, 'w') as fp:
        fp.write('<html>')
        fp.write('<body>')
        fp.write('<style>')
        fp.write(STYLE)
        fp.write('</style>')
        for k,v in notations.items():
            fp.write('<h3><b>Notations:</b></h3>')
            fp.write(f'''<ol>
            <li><b>{key}:</b> {value} </li>
            </ol>''')
            
        fp.write('<table>')
        fp.write(f'''
        <caption>
        {caption} 
        <br>Colors:
        <span class="lu">lexical_unit</span>, 
        <span class="role">role</span>,
        <span class="noun">noun</span> 
        </caption>
        ''')
        fp.write('<tr>') 
        for c in df.columns:
            fp.write(f'<th>{c}</th>')

        fp.write('</tr>')   

        for i in df.index:
            row = df.iloc[i]
            if i%(N+1) == 0:
                fp.write('<tr style="background-color:#DADDDE;">')
            else:
                fp.write('<tr>')

            for j, c in enumerate(row):
                fp.write(f'<td>{c}</td>')
            fp.write('</tr>')

        fp.write('</table>')
        fp.write('</body>')
        fp.write('</html>')
    
        

    
def table(base_exp, expanded_exps, output_file, data_dir=DATA_DIR, E=2, N=2,
         caption='Examples of Expansions',
         notations={}):
    input_exp = base_exp
    data_dir = DATA_DIR
    input_file = os.path.join(data_dir, input_exp, TRAIN_FILE)
    print('input_exp:',input_exp)
    examples, __, __ = read_conll(input_file)

    results = []
    exp_names = []
    for exp_name, (preds_model, (proc_funcs, match_lugold, match_rolegold), final_preds, exp_path) in expanded_exps.items():
        print(exp_name, preds_model, (proc_funcs, match_lugold, match_rolegold), final_preds, exp_path)
        
        output_exp = os.path.join(data_dir, exp_path)
        
        df = pd.read_pickle(f'{output_exp}/data.pkl')
        predictions = pd.read_pickle(os.path.join(output_exp, preds_model, final_preds))
        exp_names.append(exp_name)
        exp_res = {'name':exp_name,
                       'expansions':{}
                  }
        new_examples, predictions_dicts = generate_newExamples(examples, 
                                                               df, predictions,
                                                               proc_funcs=PROC_FUNCS_OPTIONS[proc_funcs],
                                                                N=2,
                                                                E=E)
        for i, ex in enumerate(examples[:E]):
 
            html_sents = html_sentence(ex, new_examples[i], predictions_dicts[i])
            mask_sents = mask_sentence(ex, new_examples[i], predictions_dicts[i])
            
            # if number of expanded examples are less than N
            for n in range(N + 1 - len(html_sents)):# < N+1: #+1 for original example
                html_sents.append('<p></p>')
                mask_sents.append('')
                
            exp_res['expansions'][i] = (html_sents, mask_sents)
            
        results.append(exp_res)
    
    df1 = pd.DataFrame(columns=exp_names)
    df2 = pd.DataFrame(columns=exp_names)
    for n, res in enumerate(results):
        name = res['name']
        expansions= res['expansions']
        sents_col1 = []
        sents_col2 = []
        for i, (sents1, sents2) in expansions.items():
            sents_col1.extend(sents1)
            sents_col2.extend(sents2)
        df1[name] = sents_col1
        df2[name] = sents_col2
        
    if output_file:
        write_html_table(df1, f'{output_file}.html' ,N=N, caption=caption,
         notations=notations)    
        df2.to_csv(f'{output_file}.csv', index=False)     
        
        
def main(configs, output_file, cuda_device, E=2, exp_names=None):
    
    if type(configs) is str:
        with open(configs, 'r') as f:
            exp_configs = json.load(f)
    
    print('Cuda device:', cuda_device)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    
    if type(exp_names) is str:
        exp_names = exp_names.split(',')

    if exp_names is None: exp_names = [exp['name'] for exp in exp_configs]

    experiments = []
    loaded_models = {'bert-large-cased':load_predictor('bert-large-cased')
                }
    for exp in exp_configs:
           if exp['name'] in exp_names:
                experiments.append(exp)
    
    print('exp_names:', exp_names)
    print('# of experiments:', len(experiments))
    input_exp = experiments[0]['args']['input_exp']
    data_dir = DATA_DIR
    if 'data_dir' in experiments[0]['args'].keys():
        data_dir = experiments[0]['args']['data_dir'] 
    input_file = f'{data_dir}/{input_exp}/{TRAIN_FILE}'
    print('input_exp:',input_exp)
    examples, __, __ = read_conll(input_file)
    examples = examples[:E] if E!=-1 else examples
    results = []
    
    use_tokenizer=False

    for exp in experiments:
        exp_res = {}
               
        name = exp['name']
        args = exp['args']
        
        print('-'*25)
        print('exp_name:', name)
        print('-'*25)
        model_type = args['preds_model']
        if model_type in loaded_models:
            predictor = loaded_models[model_type]

        else:  
            predictor = load_predictor(model_type)        
        
        if 'N' in args.keys():
            N = int(args['N'])
        else:
            N = 2
        if 'substitute_lu' in args.keys():
            substitute_lu = bool(args['substitute_lu'])
        else:
            substitute_lu = False
            
        if 'substitute_role' in args.keys():
            substitute_role = bool(args['substitute_role'])
        else:
            substitute_role = False
        
        if 'role_tokens' in args.keys():
            role_tokens = bool(args['role_tokens'])
        else:
            role_tokens = [1]    
            
        if 'noun_max' in args.keys():
            noun_max = float(args['noun_max'])
        else:
            noun_max = 0
            
        if 'ibn' in args.keys():
            ibn = bool(args['ibn'])
        else:
            ibn = False
            
        if 'proc_funcs' in args.keys():
            proc_funcs = PROC_FUNCS_OPTIONS[args['proc_funcs']]
        else:
            proc_funcs = PROC_FUNCS_OPTIONS['lemma']
            
                        
        if 'match_lugold' in args.keys():
            match_lugold = bool(args['match_lugold'])
        else:
            match_lugold = True
            
        if 'match_rolegold' in args.keys():
            match_rolegold = bool(args['match_rolegold'])
        else:
            match_rolegold = False
        
        print('exp to expand:', input_exp)
        print('N:', N)
        print('substitute_role:', substitute_role)
        print('max noun percentage:', noun_max)
        print('in_boundary noun substitution:', ibn)
        print('model_type:', args['model_type'])
        print('proc_funcs:', proc_funcs)
        print('match_lugold:', match_lugold)
        print('match_rolegold:', match_rolegold)
       
        exp_res = {'name':exp['name'],
                       'expansions':{}
                  }
        new_examples, predictions_dicts = augment_conllExamples(examples, predictor, use_tokenizer, 
                                                                N=N,
                                                                substitute_lu=substitute_lu,
                                                                substitute_role=substitute_role, role_tokens=role_tokens,
                                                                noun_max=noun_max, ibn=ibn,
                                                                proc_funcs=proc_funcs,
                                                                match_lugold=match_lugold,
                                                                match_rolegold=match_rolegold)
        for i, ex in enumerate(examples):
 
            html_sents = html_sentence(ex, new_examples[i], predictions_dicts[i])
            mask_sents = mask_sentence(ex, new_examples[i], predictions_dicts[i])
            
            # if number of expanded examples are less than N
            for n in range(N + 1 - len(html_sents)):# < N+1: #+1 for original example
                html_sents.append('<p></p>')
                mask_sents.append('')
                
            exp_res['expansions'][i] = (html_sents, mask_sents)
            
        results.append(exp_res)    
                  
        
    df1 = pd.DataFrame(columns=exp_names)
    df2 = pd.DataFrame(columns=exp_names)
    for n, res in enumerate(results):
        name = res['name']
        expansions= res['expansions']
        sents_col1 = []
        sents_col2 = []
        for i, (sents1, sents2) in expansions.items():
            sents_col1.extend(sents1)
            sents_col2.extend(sents2)
        df1[name] = sents_col1
        df2[name] = sents_col2
        
    if output_file:
        write_html_table(df1, f'{output_file}.html' ,N=N)    
        df2.to_csv(f'{output_file}.csv', index=False)
        
# ===========================================
import fire
if __name__ == '__main__':
    fire.Fire({
      'table': table,
      'main': main,
  })
    
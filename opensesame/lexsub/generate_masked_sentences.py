import os
import pandas as pd 
import pickle
from sesame.dataio import read_conll
from lexsub.augment_conll import mask_sentences, PROC_FUNCS_OPTIONS
# -----------------------------------

DATA_DIR = 'data/open_sesame_v1_data/fn1.7'

DEV_FILE = 'fn1.7.dev.syntaxnet.conll'
TEST_FILE = 'fn1.7.test.syntaxnet.conll'
TRAIN_FILE = 'fn1.7.fulltext.train.syntaxnet.conll'

# ================================


def main(input_exp, output_exp, data_dir=DATA_DIR,
         substitute_lu=True, 
         substitute_role=False, role_tokens="1", role_postags=None,
         noun_max=0, ibn=False, 
         verbose=False):   
    
    if type(role_postags) is str:
            role_postags = role_postags.split(',')
    if type(role_tokens) is str:
            role_tokens = role_tokens.split(',')
    role_tokens = [int(r) for r in role_tokens]
            
    if verbose: 
        print('exp to expand:', input_exp)
        print('substitute_lu:', substitute_lu)
        print('substitute_role:', substitute_role)
        print('role_tokens:', role_tokens)
        print('role_postags:', role_postags)

        print('max noun percentage:', noun_max)
        print('in_boundary noun substitution:', ibn)
        print('output_exp:', output_exp)

        
    input_exp = os.path.join(data_dir, input_exp)
    input_file = os.path.join(input_exp, TRAIN_FILE)

    examples, __, __ = read_conll(input_file)
    df, config_dicts = mask_sentences(examples,
                                    substitute_lu=substitute_lu,
                                    substitute_role=substitute_role, role_tokens=role_tokens, role_postags = role_postags,
                                    noun_max=noun_max, 
                                    ibn=ibn, 
                                    verbose=verbose)
    
    
    output_exp = os.path.join(data_dir, output_exp)
    if not os.path.exists(output_exp): 
        print(f'creating output_dir: {output_exp}')
        os.makedirs(output_exp, exist_ok=True)

#     print('writing configurations dict...')
#     with open(f'{output_exp}/config_dicts.pkl', 'wb') as fp:
#         pickle.dump(config_dicts, fp)

    print('writing dataframe...')
    df.to_pickle(f'{output_exp}/data.pkl')
    print('done...')
# ===========================================
import fire
if __name__ == '__main__':
    fire.Fire(main)
    
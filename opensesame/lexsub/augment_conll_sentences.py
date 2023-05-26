import os
import fire
import pandas as pd 
import pickle
from sesame.dataio import read_conll
from lexsub.augment_conll import mask_sentences, PROC_FUNCS_OPTIONS
from lexsub.augment_conll import postprocess_predictions, generate_newExamples, read_goldclusters, fill_goldclusters
from lexsub.augment_conll import write_conll, copy_dev_test_files, join_examples
# -----------------------------------

DATA_DIR = 'data/open_sesame_v1_data/fn1.7'

DEV_FILE = 'fn1.7.dev.syntaxnet.conll'
TEST_FILE = 'fn1.7.test.syntaxnet.conll'
TRAIN_FILE = 'fn1.7.fulltext.train.syntaxnet.conll'

# ================================
    
def main(input_exp, output_exp, preds_model, data_dir=DATA_DIR,
         proc_funcs='lemma', parser='nltk', match_lugold=True, match_rolegold=True, 
         N=2, postprocess=True, final_preds_path=None, pipeline='', 
         gold_clusters_verbs_path='../workdir/data/swv_gold_dataset.pkl',
         gold_clusters_nouns_path='../workdir/data/swn_gold_dataset.pkl',
         gold_clusters_roles_path='../workdir/data/swr_gold_dataset.pkl',
         verbose=False):
    
    if verbose: 
        print('exp to expand:', input_exp)
        print('preds_model:', preds_model)
        print('proc_funcs:', PROC_FUNCS_OPTIONS[proc_funcs])
        print('match_lugold:', match_lugold)
        print('match_rolegold:', match_rolegold)
        
    input_exp = os.path.join(data_dir, input_exp)
    input_file = os.path.join(input_exp, TRAIN_FILE)
    
    output_exp = os.path.join(data_dir, output_exp)
    
    save_dir_path = os.path.join(output_exp, preds_model, pipeline)
    
    if not os.path.exists(save_dir_path): 
        print(f'creating output_dir: {save_dir_path}')
        os.mkdir(save_dir_path)

    if not final_preds_path:
        final_preds_path = f'{save_dir_path}/final_predictions.pkl'
    else:
        final_preds_path = f'{data_dir}/{final_preds_path}'
    
    examples, __, __ = read_conll(input_file)
    df = pd.read_pickle(f'{output_exp}/data.pkl')
    
    print('~'*50, '\nfilling gold_cluster column...\n', '~'*50)
    GOLD_CLUSTERS = read_goldclusters(gold_clusters_verbs_path, gold_clusters_nouns_path, gold_clusters_roles_path)
    df['gold_cluster'] = fill_goldclusters(df['word_type'].tolist(), df['identifier'].tolist(), parser, 
                                           GOLD_CLUSTERS)
    
#     config_dicts = pd.read_pickle(f'{output_exp}/config_dicts.pkl')
    if postprocess:
        predictions = pd.read_pickle(f'{output_exp}/{preds_model}/predictions.pkl')

        predictions = postprocess_predictions(df.copy(), predictions,
                                              proc_funcs=PROC_FUNCS_OPTIONS[proc_funcs],
                                              parser=parser,    
                                              verbose=verbose)
        
        print('~'*50, '\nsaving final predictions...\n', '~'*50)
                
        with open(final_preds_path, 'wb') as fp:
            pickle.dump(predictions, fp)        

    
    else:
        predictions = pd.read_pickle(final_preds_path)


    augmented_examples, config_dicts =  generate_newExamples(examples, df, predictions,
                                                             match_lugold=match_lugold, match_rolegold=match_rolegold,
                                                             proc_funcs=PROC_FUNCS_OPTIONS[proc_funcs], parser=parser,
                                                             N=N,
                                                             verbose=verbose)

    all_examples =  join_examples(examples, augmented_examples)

#     return augmented_examples
    save_dir_path = os.path.join(output_exp, preds_model, f'{pipeline}')
    output_file = f'{save_dir_path}/{TRAIN_FILE}'
    print('writing train file...')
    write_conll(all_examples, output_file)
    print('writing dev and test file...')
    copy_dev_test_files(input_exp, save_dir_path)
    print('done...')
            
# ===========================================
if __name__ == '__main__':
    fire.Fire(main)
    
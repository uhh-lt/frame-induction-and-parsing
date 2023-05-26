import fire
from .io_util import read_file, write_file
from .util import preprocess_fn_verbs

from .mask_words import mask_lu_T, mask_lu_Tand, mask_lu_andT, mask_lu_Tandthen, mask_lu_andthenT, mask_lu_TandT, mask_lu_TandthenT
from .mask_words import mask_role_T, mask_role_Tand, mask_role_andT, mask_role_Tandthen, mask_role_andthenT, mask_role_TandT, mask_role_TandthenT

from .create_datasets import extract_lu_data, extract_verb_clusters, extract_noun_clusters, lu_dataset, get_lu_clusters
from .create_datasets import extract_roles_data, extract_role_clusters, roles_dataset, get_role_clusters, lemmatize_roleclusters
# ----------------------------------------------------------
# PATTERNS = ['T', 'Tand-','-andT','Tandthen-','-andthenT']
PATTERNS = ['T', 'Tand-','Tandthen-', 'TandT', 'TandthenT']

lu_patterns = {
    'T': mask_lu_T,
    'Tand-': mask_lu_Tand,
    '-andT': mask_lu_andT,
    'Tandthen-': mask_lu_Tandthen,
    '-andthenT': mask_lu_andthenT,
    'TandT': mask_lu_TandT,
    'TandthenT': mask_lu_TandthenT
}  
role_patterns = {
    'T': mask_role_T,
    'Tand-': mask_role_Tand,
    '-andT': mask_role_andT,
    'Tandthen-': mask_role_Tandthen,
    '-andthenT': mask_role_andthenT,
    'TandT': mask_role_TandT,
    'TandthenT': mask_role_TandthenT
}  
# ----------------------------------------------------------
input_dir = '../workdir/framenet'
output_dir = '../workdir/data'


from .lemmatize_util import lemma
try:
    lemma('Progress', lemmatizer='pattern')
except:
    pass

    
def gold_lu_dataset(input_file, cluster_file, output_dir, n_tokens=[1], pos='v', name_prefix='swv'):
    
    gold = lu_dataset(input_file=input_file, n_tokens=n_tokens, contigous=True)
    print('*'*15, 'creating gold dataset of {} word lus'.format(n_tokens)) 
    if n_tokens!=[1]:
        n_tokens = [i+1 for i in range(n_tokens[-1])]
        print('for {} word lus'.format(n_tokens))
    gold['gold_cluster'] = get_lu_clusters(gold['frameName'], cluster_file = cluster_file, n_tokens=n_tokens)
    if pos=='v':
        gold['gold_cluster_processed'] = gold['gold_cluster'].apply(preprocess_fn_verbs)
    else:
        gold['gold_cluster_processed'] = gold['gold_cluster']

    write_file(gold, '{}/{}_gold_dataset.pkl'.format(output_dir, name_prefix))

    
def gold_roles_dataset(input_file, cluster_files, output_dir, n_tokens=[1], name_prefix='swr'):
    
    gold = roles_dataset(input_file=input_file, n_tokens=n_tokens)
    print('*'*15, 'creating gold dataset of {} word roles'.format(n_tokens))  
    if n_tokens!=[1]:
        n_tokens = [i+1 for i in range(n_tokens[-1])]
        print('for {} word roles'.format(n_tokens))
    keys = list(cluster_files.keys())
    for key in keys:
        print(key)
        gold[key] = get_role_clusters(gold['feID'], cluster_file = cluster_files[key], n_tokens=n_tokens)

    write_file(gold, '{}/{}_gold_dataset.pkl'.format(output_dir, name_prefix))


def mask_lu_datasets(input_file, output_dir, masking_patterns=PATTERNS, n_tokens=[1], name_prefix='swv'):

    lus = lu_dataset(input_file=input_file, n_tokens=n_tokens)
    print('*'*15, 'masking lu')    
    for pattern in masking_patterns:
        lus['masked_sent'] = lus.apply(lu_patterns[pattern], axis=1)
        write_file(lus, '{}/{}_{}.pkl'.format(output_dir, name_prefix, pattern))
    

def mask_roles_datasets(input_file, output_dir, masking_patterns=PATTERNS, n_tokens=[1], name_prefix='swr'):

    roles = roles_dataset(input_file=input_file, n_tokens=n_tokens)
    print('*'*15, 'masking of {} word roles'.format(n_tokens))    
    for pattern in masking_patterns:
        roles['masked_sent'] = roles.apply(role_patterns[pattern], axis=1)
        write_file(roles, '{}/{}_{}.pkl'.format(output_dir, name_prefix, pattern))

# ----------------------------------------------
def create_essential_verbs_datasets(input_dir, output_dir):
    
    MAIN_FILE = '{}/all_fn_data.csv.gz'.format(input_dir)
    
    verbs_file='{}/final_verbs_data.pkl'.format(output_dir)   
    verbs_clusters_file='{}/fn_verbs_clusters.pkl'.format(output_dir)   
 
    print('*'*15, 'extracting verbs dataset')    
    extract_lu_data(input_file = MAIN_FILE, output_file=verbs_file, pos='v')
    print('*'*15, 'extracting verbs clusters')    
    extract_verb_clusters(input_dir = input_dir, output_file=verbs_clusters_file)

def create_essential_nouns_datasets(input_dir, output_dir):
    
    MAIN_FILE = '{}/all_fn_data.csv.gz'.format(input_dir)
    
    lu_file='{}/final_nouns_data.pkl'.format(output_dir)   
    lu_clusters_file='{}/fn_nouns_clusters.pkl'.format(output_dir)   
 
    print('*'*15, 'extracting noun dataset')    
    extract_lu_data(input_file = MAIN_FILE, output_file=lu_file, pos='n')
    print('*'*15, 'extracting noun clusters')    
    extract_noun_clusters(input_dir = input_dir, output_file=lu_clusters_file)
    

def create_essential_roles_datasets(input_dir, output_dir):
    
    MAIN_FILE = '{}/all_fn_data.csv.gz'.format(input_dir)
    
    roles_file='{}/final_roles_data.pkl'.format(output_dir)  
    roles_clusters_file='{}/fn_roles_clusters.pkl'.format(output_dir)   
    
  
    print('*'*15, 'extracting roles dataset')    
    extract_roles_data(input_file = MAIN_FILE, output_file=roles_file)
    
    print('*'*15, 'extracting role clusters')    
    extract_role_clusters(input_file = roles_file, output_file=roles_clusters_file)


def create_source_datasets(input_dir=input_dir, output_dir=output_dir, 
                              data_types=['verbs', 'nouns', 'roles']):
    """
    required_files: all_fn_data, lus-wo-ch-verbs, lus-with-ch-r-verbs 
    these files are present in input_dir
    all new files will be saved to output_dir
    """
    
    MAIN_FILE = '{}/all_fn_data.csv.gz'.format(input_dir)

    if 'verbs' in data_types:
        #============================================================== Verbs dataset
        verbs_file='{}/final_verbs_data.pkl'.format(output_dir) 
        verbs_clusters_file='{}/fn_verbs_clusters.pkl'.format(output_dir)   


        create_essential_verbs_datasets(input_dir, output_dir)

    if 'nouns' in data_types:
    #============================================================== Nouns dataset
        nouns_file='{}/final_nouns_data.pkl'.format(output_dir) 
        nouns_clusters_file='{}/fn_nouns_clusters.pkl'.format(output_dir)   


        create_essential_nouns_datasets(input_dir, output_dir)

    if 'roles' in data_types:
        # ============================================================== Roles datasets
        roles_file='{}/final_roles_data.pkl'.format(output_dir)  
        roles_clusters_file='{}/fn_roles_clusters.pkl'.format(output_dir)   

        create_essential_roles_datasets(input_dir, output_dir)

#     #     -------------------------------
        print('*'*15, 'lemmatizing roles clusters') 
        roles_clusters_patternlemmatized='{}/fn_roles_clusters_patternlemmatized.pkl'.format(output_dir)
        roles_clusters_nltklemmatized='{}/fn_roles_clusters_nltklemmatized.pkl'.format(output_dir)   

        lemmatize_roleclusters(input_file = roles_clusters_file, output_file=roles_clusters_patternlemmatized, parser='pattern')
        lemmatize_roleclusters(input_file = roles_clusters_file, output_file=roles_clusters_nltklemmatized, parser='nltk')

   # ---------------------------------------------------
def create_final_datasets(input_dir=input_dir, output_dir=output_dir, 
                              data_types=['verbs', 'nouns', 'roles']):
    """
    required_files: final_verbs_data, final_nouns_data, final_roles_data
    these files are present in input_dir
    all new files will be saved to output_dir
    """
    
    if 'verbs' in data_types:
        #============================================================== Verbs dataset
        verbs_file='{}/final_verbs_data.pkl'.format(input_dir) 
        verbs_clusters_file='{}/fn_verbs_clusters.pkl'.format(input_dir)   

#         # ------------------------------- Gold dataset
        gold_lu_dataset(input_file=verbs_file, cluster_file = verbs_clusters_file, output_dir=output_dir, n_tokens=[1], name_prefix = 'swv')

#         gold_lu_dataset(input_file=verbs_file, cluster_file = verbs_clusters_file, output_dir=output_dir, n_tokens=[2, 3], name_prefix = 'mwv')
#         # ------------------------------- Masking  
        mask_lu_datasets(input_file=verbs_file, output_dir=output_dir, n_tokens=[1], name_prefix = 'swv')
#         mask_lu_datasets(input_file=verbs_file, output_dir=output_dir, n_tokens=[2, 3], name_prefix = 'mwv')
    
    if 'nouns' in data_types:
        #============================================================== Nouns dataset
        nouns_file='{}/final_nouns_data.pkl'.format(input_dir) 
        nouns_clusters_file='{}/fn_nouns_clusters.pkl'.format(input_dir)   

        # ------------------------------- Gold dataset
        gold_lu_dataset(input_file=nouns_file, cluster_file = nouns_clusters_file, output_dir=output_dir, n_tokens=[1], pos='n', name_prefix = 'swn')

#         gold_lu_dataset(input_file=nouns_file, cluster_file = nouns_clusters_file, output_dir=output_dir, n_tokens=[2, 3], pos='n', name_prefix = 'mwn')
        # ------------------------------- Masking  
        mask_lu_datasets(input_file=nouns_file, output_dir=output_dir, n_tokens=[1], name_prefix = 'swn')
#         mask_lu_datasets(input_file=nouns_file, output_dir=output_dir, n_tokens=[2, 3], name_prefix = 'mwn')

    if 'roles' in data_types:
        # ============================================================== Roles datasets
        roles_file='{}/final_roles_data.pkl'.format(input_dir)  
        roles_clusters_file='{}/fn_roles_clusters.pkl'.format(input_dir)   

        roles_clusters_patternlemmatized='{}/fn_roles_clusters_patternlemmatized.pkl'.format(input_dir)
        roles_clusters_nltklemmatized='{}/fn_roles_clusters_nltklemmatized.pkl'.format(input_dir)   

    #     -------------------------------

        all_role_cluster_files = {'gold_cluster': roles_clusters_file,
                                  'gold_cluster_patternlemmatized': roles_clusters_patternlemmatized,
                                  'gold_cluster_nltklemmatized': roles_clusters_nltklemmatized
                                 }

    # ------------------------------- Gold dataset
        gold_roles_dataset(input_file=roles_file, cluster_files = all_role_cluster_files, output_dir=output_dir, n_tokens=[1], name_prefix='swr')

#         gold_roles_dataset(input_file=roles_file, cluster_files = all_role_cluster_files, output_dir=output_dir, n_tokens=[2, 3], name_prefix='mwr')

        # ------------------------------- Masking
        mask_roles_datasets(input_file=roles_file, output_dir=output_dir, n_tokens=[1], name_prefix='swr')

#         mask_roles_datasets(input_file=roles_file, output_dir=output_dir, n_tokens=[2, 3], name_prefix='mwr')
    
# ---------------------------------------------------


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='generate some essential datasets including verbs, roles, their clustering and masked datasets etc frames, assuming following three files are in input_dir:\n 1) all_fn_data\n 2) lus-wo-ch-verbs\n 3) lus-with-ch-r-verbs \n all output files will be written to output_dir')
    parser.add_argument('-i', '--input_dir', help='Directory with the input files.')
    parser.add_argument('-o', '--output_dir', help='Output directory.')
    parser.add_argument('--data_types', default='verbs,nouns,roles')
    args = parser.parse_args()
    print("Input: ", args.input_dir)
    print("Output: ", args.output_dir)
    
    create_source_datasets(args.input_dir, args.output_dir, args.data_types.split(','))
    create_final_datasets(args.input_dir, args.output_dir, args.data_types.split(','))


if __name__ == '__main__':
    fire.Fire()

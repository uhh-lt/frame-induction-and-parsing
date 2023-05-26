import pandas as pd
from  ordered_set import OrderedSet
import os
from pathlib import Path

from .util import luIndex_toList, strList_toList

from .io_util import read_csv, to_csv, read_file, write_file
from .mask_words import mask_lu_T, mask_lu_Tand, mask_lu_andT, mask_lu_Tandthen, mask_lu_andthenT
from .mask_words import mask_role_T, mask_role_Tand, mask_role_andT, mask_role_Tandthen, mask_role_andthenT
from .mask_words import mask_lu_withPatterns, mask_lu_withPatterns

# from .lemmatize_util import spacy_poslemma, spacy_lemma
from .lemmatize_util import nltk_poslemma

from .lemmatize_util import pattern_lemma
# --------------------------------------------------
lemmatizers = {'pattern': pattern_lemma,
               'nltk': nltk_poslemma,
#                'spacy': spacy_lemma,
#                'spacypos': spacy_poslemma,
              }
# ----------------------
DATA_FILES = '../workdir/framenet'
MAIN_FILE = '{}/all_fn_data.csv.gz'.format(DATA_FILES)
# ----------------------------------------------------------
def extract_lu_data(input_file = MAIN_FILE,
                    output_file='{}/final_verbs_data.pkl'.format(DATA_FILES),
                    pos='v'):
    #  Lu dataset
    df = read_file(input_file)
    print(output_file)
    dfv = df[['frameID', 'frameName', 'sentence', 'luName', 'luID', 'luText', 'luPOS', 'luIndex']]
    print('# of records extracted in total:', len(dfv))
    dfv = dfv.loc[dfv['luPOS'] == pos]
    dfv = dfv.drop(columns =['luPOS'])
    dfv = dfv.dropna(axis=0)
    print('after dropping null:', len(dfv))

    dfv = dfv.drop_duplicates(['frameID', 'frameName', 'sentence', 'luName', 'luID', 'luText', 'luIndex']).reset_index(drop=True)
    print('after dropping duplicates:', len(dfv))
    
    dfv['luIndex'] = dfv['luIndex'].apply(luIndex_toList)
    

    if output_file is not None:
        write_file(dfv, output_file)    

    return dfv

def extract_roles_data(input_file = MAIN_FILE,
                       output_file='{}/final_roles_data.pkl'.format(DATA_FILES)):

    df = read_csv(input_file)


    #  ROLES dataset
    dfr = df[['frameID', 'frameName', 'sentence', 'feName',  'feID', 'feText', 'feIndex']]
    print('# of records extracted in total:', len(dfr))

    dfr = dfr.dropna(axis=0)
    print('after dropping null:', len(dfr))

    dfr = dfr.drop_duplicates(['frameID', 'frameName', 'sentence', 'feName', 'feID', 'feText', 'feIndex']).reset_index(drop=True)
    print('after dropping duplicates:', len(dfr))
    
    dfr['feID'] = dfr['feID'].astype(int)
    dfr['feIndex'] = dfr['feIndex'].apply(luIndex_toList)

    if output_file is not None:
        write_file(dfr, output_file)  
    
    return dfr



def extract_verb_clusters(input_dir=DATA_FILES,
                         output_file='{}/fn_verbs_clusters.pkl'.format(DATA_FILES)):
    
    """This method assumes the input_dir contains two files, which are created in result of executing extract_framenet_verb_clusters"""
    
    df = pd.DataFrame(columns=['frameName', 'cluster'])#, 'cluster_withchildframes'])

    file = '{}/lus-wo-ch-verbs.csv'.format(input_dir)
    df1 = pd.read_csv(file, sep = '\t')
    df1 = df1.loc[df1['size']>0]
   
    df['frameName'] = df1['cid']

    df['cluster'] = df1['cluster'].apply(lambda x: x.split(','))
    df['cluster'] = df['cluster'].apply(lambda lus: [lu.strip() for lu in lus])
    df['cluster'] = df['cluster'].apply(lambda lus: [lu.split('.v')[0] for lu in lus])

#     file = '{}/lus-with-ch-r-verbs.csv'.format(input_dir)
#     df2 = pd.read_csv(file, sep = '\t')
#     df2 = df2.loc[df2['size']>0]


#     df['cluster_withchildframes'] = df2['cluster'].apply(lambda x: x.split(','))
#     df['cluster_withchildframes'] = df['cluster_withchildframes'].apply(lambda lus: [lu.strip() for lu in lus])
#     df['cluster_withchildframes'] = df['cluster_withchildframes'].apply(lambda lus: [lu.split('.v')[0] for lu in lus])

    if output_file is not None:
        write_file(df, output_file)  
   
    
    return df
def extract_noun_clusters(input_dir=DATA_FILES,
                         output_file='{}/fn_noun_clusters.pkl'.format(DATA_FILES)):
    
    """This method assumes the input_dir contains two files, which are created in result of executing extract_framenet_verb_clusters"""
    
    df = pd.DataFrame(columns=['frameName', 'cluster'])#, 'cluster_withchildframes'])

    file = '{}/lus-wo-ch.csv'.format(input_dir)
    df1 = pd.read_csv(file, sep = '\t')
    df1 = df1.loc[df1['size']>0]
    
    df['frameName'] = df1['cid']

    df['cluster'] = df1['cluster'].apply(lambda x: x.split(','))
    df['cluster'] = df['cluster'].apply(lambda lus: [lu.strip() for lu in lus])
    df['cluster'] = df['cluster'].apply(lambda lus: [lu for lu in lus if lu.endswith('.n')])
    df['cluster'] = df['cluster'].apply(lambda lus: [lu.split('.n')[0] for lu in lus])
    df = df.loc[df['cluster'].apply(lambda lus: lus != [])]

#     file = '{}/lus-with-ch-r.csv'.format(input_dir)
#     df2 = pd.read_csv(file, sep = '\t')
#     df2 = df2.loc[df2['size']>0]


#     df2['cluster_withchildframes'] = df2['cluster'].apply(lambda x: x.split(','))
#     df2['cluster_withchildframes'] = df2['cluster_withchildframes'].apply(lambda lus: [lu.strip() for lu in lus])
#     df2['cluster_withchildframes'] = df2['cluster_withchildframes'].apply(lambda lus: [lu for lu in lus if lu.endswith('.n')])
#     df2['cluster_withchildframes'] = df2['cluster_withchildframes'].apply(lambda lus: [lu.split('.n')[0] for lu in lus])
#     df2 = df2.loc[df2['cluster_withchildframes'].apply(lambda lus: lus != [])]
    if output_file is not None:
        write_file(df, output_file)  
   
    
    return df


def extract_role_clusters(input_file = '{}/final_roles_data.pkl'.format(DATA_FILES),
                       output_file='{}/fn_roles_clusters.pkl'.format(DATA_FILES)):


    dfr = read_file(input_file)
    #  gold dataset of ROLES
    dfn = dfr[['frameID', 'frameName', 'feName', 'feID', 'feText']]
    dfn = dfn.drop_duplicates(['frameID', 'frameName', 'feName', 'feID', 'feText'])
    print('# of records in gold dataset of roles:', len(dfn))
    
    to_csv(dfn, '{}/fn_roles_gold.csv'.format(Path(output_file).parent))
    
    roles_dict = goldroles_to_dict(dfn)

    dfg = dfn[['frameName', 'feName', 'feID']].copy()
    dfg = dfg.drop_duplicates().reset_index(drop=True)
    
    print('# of roles clusters', len(dfg))

    dfg['cluster'] = dfg['feID'].apply(lambda x: roles_dict[x])
    
    if output_file is not None:
        write_file(dfg, output_file)  

    return dfg
        
    
        
def goldroles_to_dict(df_roles, n_tokens=None):
    """if n_tokens=None, consider all roles"""

    roles_dict = {}
    for feid in df_roles['feID'].unique():
        roles = df_roles[df_roles['feID']==feid]['feText']
        roles = roles.tolist()
        if n_tokens is not None:
            roles = [r for r in roles if len(r.split(' ')) in n_tokens]
        roles_dict[feid] = roles
        
    return roles_dict 


def lemmatize_roleclusters(input_file='{}/fn_roles_clusters.pkl'.format(DATA_FILES), 
                         output_file='{}/fn_roles_clusters_patternlemmatized.pkl'.format(DATA_FILES), parser='pattern'): 
    
    lemmatizer_func = lemmatizers[parser]
    role_clusters = read_file(input_file)
#     pattern parser do the lowercase itself, nltk also dependent on lowercase for correct answer
    role_clusters['cluster'] = role_clusters['cluster'].apply(lambda roles: [lemmatizer_func(role.lower()) for role in roles])
    role_clusters['cluster'] = role_clusters['cluster'].apply(lambda roles: list(OrderedSet(roles)))
      
    write_file(role_clusters, output_file)
    return role_clusters


# -----------------------------------------------------------------------
def get_lu_clusters(frames, cluster_file = '{}/fn_verbs_clusters.pkl'.format(DATA_FILES), childframes=False, n_tokens=None): 
    """n_tokens functionality is not implemented yet"""
    
    lu_clusters = read_file(cluster_file)
    
    clusters = []
    if childframes:
        for frame in frames:
            clusters.append(lu_clusters.loc[lu_clusters['frameName']==frame, 'cluster_withchildframes'].values[0])
    else:
        for frame in frames:
            clusters.append(lu_clusters.loc[lu_clusters['frameName']==frame, 'cluster'].values[0])
    
    if n_tokens is not None:
        clusters = [[lu for lu in cluster if len(lu.split(' ')) in n_tokens] for cluster in clusters]

    
    return clusters

def get_role_clusters(feIDs, cluster_file='{}/fn_roles_clusters.pkl'.format(DATA_FILES), n_tokens=None): 
    
    role_clusters = read_file(cluster_file)
    if n_tokens is not None:
        role_clusters['cluster'] = role_clusters['cluster'].apply(lambda roles: [role for role in roles if len(role.split(' ')) in n_tokens])
    clusters = []
    for feID in feIDs:
            clusters.append(role_clusters.loc[role_clusters['feID']==feID, 'cluster'].values[0])
    
    return clusters

# ======================================================================= 
def cardinality(text, *args):
    
    if type(text) is list:
        return len(text)
    else:
        return len(text.split(' '))

def contigous_indices(indices):
    e = indices[0][1]
    for s1, e1 in indices[1:]:
        if e+2 == s1: 
            e = e1
            continue
        else:
            return False
    return True


def contigous_tokens(row):
    if row['sentence'].find(row['luText']) > -1: return True
    return False



# -----------------------------
def lu_dataset(input_file = '{}/final_verbs_data.pkl'.format(DATA_FILES), 
                 output_file=None,
                 n_tokens = [1],
                 contigous=True):
    """
    n_tokens: a list of integers with required tokens in verb lu, or m, to slect all verbs with tokens > 1
    contigous = if n_tokens > 1 then select the records with contigous tokens    
    """
    df = read_file(input_file)
    df = df[['frameID','frameName', 'luName', 'luID', 'luText', 'luIndex', 'sentence']]
    print("# of input records: ", len(df))
        
    if(type(df['luIndex'].iloc[0]) is not list):
        df['luIndex'] = df['luIndex'].apply(luIndex_toList)        
    
     
    if n_tokens == 'm':
        df = df.loc[(df['luName'].apply(cardinality)> 1) 
                    & (df['luName'].apply(cardinality) == df['luText'].apply(cardinality))
                    & (df['luName'].apply(cardinality)==df['luIndex'].apply(cardinality))].copy().reset_index(drop=True)
    
    else: 
        dfv = pd.DataFrame(columns=df.columns)
        for n in n_tokens:
            df1 = df.loc[(df['luName'].apply(cardinality) == n) 
                    & (df['luText'].apply(cardinality)  == n) 
                    & (df['luIndex'].apply(cardinality) ==n)].copy().reset_index(drop=True)
            
            dfv = dfv.append(df1, ignore_index=True)
            
        df = dfv.reset_index(drop=True)

    # ---------------------- 
    print('# of records with {} word lus:{}'.format(n_tokens, len(df)))
    if contigous:
        df = df.loc[df['luIndex'].apply(contigous_indices)]
        print('# of records with contigous text of lus:{}'.format(len(df)))


    # ----------------------
    if output_file != None:
        write_file(df, output_file)  
       
    return df

# # ---------------- produce verb dataset with best configurations [T]
# def swl_dataset_best(input_file = '{}/final_verbs_data.pkl'.format(DATA_FILES),
#                      output_file='{}/swv_T.pkl'.format(DATA_FILES)):
        
#     df = lu_dataset(input_file=input_file, n_tokens=1)
#     df['masked_sent'] = df.apply(mask_lu_T, axis=1)
    
#     # ---------------------- get verb clusters from framenet
#     print('# of records with {} tokens lus:{}'.format(1, len(df)))

#     # ----------------------
#     if output_file != None:
#         write_file(df, output_file)  

        
#     return df
# ---------------------------------------------------------- Role datasets

def roles_dataset(input_file = '{}/final_roles_data.pkl'.format(DATA_FILES),
                  output_file = None, 
                  n_tokens=[1]):
    
    """
    n_tokens: can be  an integr or a list of integers, or m, to slect all verbs with tokens > 1
    
    roles are normally always contigous
    """

    df = read_file(input_file)
    print("# of input records: ", len(df))

    df = df[['frameID','frameName', 'feName', 'feID', 'feText','feIndex', 'sentence']]
    if(type(df['feIndex'].iloc[0]) is not list):
        df['feIndex'] = df['feIndex'].apply(luIndex_toList)
    
        
    if n_tokens == 'm':
        df = df.loc[df['feText'].apply(cardinality)> 1].copy().reset_index(drop=True)
    
    else:
        dfr = pd.DataFrame(columns=df.columns)
        for n in n_tokens:
            df1 = df.loc[df['feText'].apply(cardinality) == n].copy().reset_index(drop=True)
            
            dfr = dfr.append(df1, ignore_index=True)
            
        df = dfr.reset_index(drop=True)

    # ----------------------
    print('# of records with {} words role:{}'.format(n_tokens, len(df)))
    # ----------------------
    if output_file != None:
        write_file(df, output_file)  

    return df

# --------------------------------

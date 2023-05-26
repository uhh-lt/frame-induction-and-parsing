from scipy.stats import ttest_ind_from_stats, ttest_ind
import pandas as pd
from glob import glob

def parse_results(result_file, parser='bert'):
    
    with open(result_file, 'r') as fp:
        lines = fp.readlines()
    if 'test' in result_file: 
        
        f1 = lines[0].split(':')[1].replace('\n', '')
        p = lines[1].split(':')[1].replace('\n', '')
        r = lines[2].split(':')[1].replace('\n', '')
        if 'argid' in result_file or parser == 'bert': 
            f1, p, r = [m.split(':')[1] for m in lines[0].replace('\n', '').split(',')]
            
    else:
        f1 = lines[0].replace('\n', '')
        p = lines[1].replace('\n', '')
        r = lines[2].replace('\n', '')
        
    return float(f1)*100, float(p)*100, float(r)*100
        

def ttest(sample1, sample2, equal_var = True): # if equal_var = False, it becomes Welch’s t-test
    t, p = ttest_ind(sample1, sample2, equal_var = equal_var)    
    if p < 0.01:
        return 'p < 0.01'
    else:
        return 'p > 0.01'
    return p
   

def ttest_df(row, seed_df, df, eval_measure='f1', equal_var = True, verbose=False):
    if verbose:
        print(df.loc[(df['preds_model']==row['preds_model'].iloc[0])  & (df['task']==row['task'].iloc[0]) & (df['exp']==row['exp'].iloc[0])])    
    
    return ttest(seed_df.loc[seed_df['task']==row['task'].iloc[0]][eval_measure], 
                df.loc[(df['preds_model']==row['preds_model'].iloc[0])  & (df['task']==row['task'].iloc[0]) & (df['exp']==row['exp'].iloc[0])][eval_measure],
                equal_var = equal_var)


def get_results_opensesame(all_exps, output_model_dir,
               task_name="argid", verbose=False):
    if verbose: print(f"Getting results for: {task_name}\n")
    df = pd.DataFrame(columns=['exp_path', 'task', 'run#', 'f1', 'pre', 'rec'])

    for exp in all_exps:        
#         print(exp)
        for model in glob(f'{output_model_dir}/{exp}/{task_name}*'):
#             print(model)
            
            run_num = model.split('/')[-1].split('-')[-1]
            if not run_num.isdigit(): continue
            else: run_num = int(run_num)    
            if run_num == 0: continue
            if not task_name in model: continue
            try:
                f1, p, r = parse_results(f'{model}/test-f1.txt')

#                 task_name = model.split('/')[-1].split('_')[0]
#                 print(task_name)
                df.loc[len(df)] = [exp, task_name, int(run_num), f1, p, r]
            except Exception as e:
    #             print(model)
                print(e)
                continue
    return df


def get_results_bertSRL(exps, output_model_dir, verbose=False):
    df = pd.DataFrame(columns=['exp_path', 'task', 'run#', 'f1', 'pre', 'rec'])
    for exp in exps:
        for model in glob(f'{output_model_dir}/{exp}/*'):
            try:
                run_num = int(model.split('/')[-1].split('-')[-1])
            except:continue    
            try:
                f1, p, r = parse_results(f'{model}/test-f1.txt')

                task_name = 'argid' 
                df.loc[len(df)] = [exp, task_name, run_num, f1, p, r]
            except Exception as e:
                print(e)
                continue
    return df



def prettify(df):
    
    df['preds_model'] = df['exp_path'].apply(lambda x: x.split('/')[2] if 'expanded' in x else 'base')
    df['pipeline'] = df['exp_path'].apply(lambda x: x.split('/')[3] if 'expanded' in x else '')

    df['dataset'] = df['exp_path'].apply(lambda x: x.split('/')[1])

    df['dataset'] = df['dataset'].apply(lambda x: x.replace('01ExPerSent_verbs_rand01_expanded_','augmented_'))
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('01ExPerSent_nouns_rand01_expanded_','augmented_'))
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('_', '-'))
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('lu', 'lexical unit'))

    df = df.round(2)
    return df


def prettify2(df):

    df['preds_model'] = df['exp_path'].apply(lambda x: x.split('/')[2] if 'expanded' in x else 'base')
    df['dataset'] = df['exp_path'].apply(lambda x: x.split('/')[1])
    df['pipeline'] = df['exp_path'].apply(lambda x: x.split('/')[3] if 'expanded' in x else '')
    df['sample_size'] = df['dataset'].apply(lambda x: int(x[:3]))

    df['dataset'] = df['dataset'].apply(lambda x: x.split('pc_')[-1])
    
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('verbs_expanded_','augmented_'))
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('nouns_expanded_','augmented_'))

    df['dataset'] = df['dataset'].apply(lambda x: "nPercentData" if x== 'verbs' or x == 'nouns' else x)


    
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('_', '-'))
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('lu', 'lexical unit'))
    
    df = df.round(2)
    
    return df

## calculate mean, std, p-value for Experiment-1: 01AnnotationPerSentence
def compute_statistical_measures(df,
                                 eval_measure="f1",
                                 verbose=False):
    
    group_columns = ['preds_model', 'dataset', 'task']

    dfg=df.groupby(group_columns, as_index=False).agg(['mean', 'std']).reset_index()
    dfg = dfg.round(2)
    ps = []

    for m, ds in zip(dfg['preds_model'], dfg['dataset']):

        sample1 = df.loc[(df['preds_model']=='base')][eval_measure]
        sample2 = df.loc[(df['preds_model']==m) & (df['dataset']==ds)][eval_measure]
        if verbose:
            if m!="base":
                print(n, m, ds)
                print("sample1:", list(sample1))
                print("sample2:", list(sample2))
                print(ttest(sample1, sample2))
        ps.append(ttest(sample1, sample2))

    dfg['p_value'] = ps
    dfg
    return dfg



## calculate mean, std, p-value for Experiment-2: learning curve
def compute_statistical_measures_lr(df,
                                  eval_measure="f1",
                                  verbose=False):
    
    group_columns = ['preds_model', 'dataset', 'task', 'sample_size']

    dfg=df.groupby(group_columns, as_index=False).agg(['mean', 'std']).reset_index()
    dfg = dfg.round(2)
    ps = []

    for m, ds, n in zip(dfg['preds_model'], dfg['dataset'], dfg['sample_size']):

        sample1 = df.loc[(df['preds_model']=='base') & (df['sample_size']==n)][eval_measure]
        sample2 = df.loc[(df['preds_model']==m) & (df['dataset']==ds) & (df['sample_size']==n)][eval_measure]
        if verbose:
            if m!="base":
                print(n, m, ds)
                print("sample1:", list(sample1))
                print("sample2:", list(sample2))
                print(ttest(sample1, sample2))
        ps.append(ttest(sample1, sample2))

    dfg['p_value'] = ps
    dfg
    return dfg


# seed_df = df.loc[df['preds_model']=='base'].copy()
# # dfg['p_value'] = dfg.apply(lambda row: ttest(seed_df.loc[seed_df['task']==row['task'].iloc[0]]['f1'], 
# #                                              df.loc[(df['preds_model']==row['preds_model'].iloc[0])  & (df['task']==row['task'].iloc[0]) & (df['exp']==row['exp'].iloc[0])]['f1']), 
# #                            axis=1)
# dfg['p_value'] = dfg.apply(lambda row: ttest_df(row, seed_df, df, verbose=False),
#                                 axis=1)

        
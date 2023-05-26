import os
import pandas as pd
from ordered_set import OrderedSet 
import time
from .io_util import read_file, write_file
from .lemmatize_util import lemma
from .util import strList_toList
from .util import remove_quotes, not_quoted
from .create_datasets import extract_lu_data,extract_roles_data
# ------------------------------------------- get single word list for DT extraction
def lus_forDT(df):
    
    words = []
    
    lus = df['luName'].unique().tolist()
    lus.extend(df['luText'].unique().tolist())
    # single token
    lus = [lu for lu in lus if not ' ' in lu]

    words.extend(lus)
    words.extend([w.lower() for w in words])
    words = list(set(words))

    lemmas1 = [lemma(w, lemmatizer='pattern') for w in words]
#     lemmas2 = [lemma(w, lemmatizer='spacy') for w in words]

    words.extend(lemmas1)
    words.extend(lemmas2)


    words = list(set(words))
    words.sort()
    
    print('single word lus: ', len(words))
    
    return words
    

def roles_forDT(df):
     
    words = []
    
    df = df.loc[df['feText'].apply(lambda x: not ' ' in x)]
    roles = df['feText'].unique().tolist()
    
    words.extend(roles)
    words.extend([w.lower() for w in words])
    words = list(set(words))

    lemmas1 = [lemma(w, lemmatizer='pattern') for w in words]
#     lemmas2 = [lemma(w, lemmatizer='spacy') for w in words]

    words.extend(lemmas1)
    words.extend(lemmas2)


    words = list(set(words))
    words.sort()
        
    print('single word roles: ', len(words))
    
    return words


# ------------------------------------------------
def divide_into_chunks(L, N):
    for i in range(0, len(L), N):
        yield L[i:i+N]

    
def _extract_dt(chunk, words):
     
    rows = []
    chunk_size = len(chunk)    
    
    for i in range(chunk_size):
        row =  chunk[i].split('\t')
        word = row[0]
#         word = remove_quotes(row[0])
#         word_0 = lemma(word, 'spacy')
        
        if word in words:
            rows.append([row[0], row[1], float(row[2])])

    return rows
    
    
def extract_dt(file, wordlist):
    """
    extract all records from dt for wordlist 
    """
     
    words = OrderedSet(wordlist)
    
    print("reading :", file)

    fp= open(file, 'r')
    lines = fp.readlines()
    fp.close()
    print("# of lines:", len(lines))
    
    chunks = list(divide_into_chunks(lines, 50000000))
    N = len(chunks)
    new_rows = []
    
    for n in range(0, N):
                   
        print('chunk#:------', n)
        print('# of lines:', len(chunks[n]))

        start_time = time.time()
        
        rows = _extract_dt(chunks[n], words)
        
        elapsed_time = time.time() - start_time
        print(time.strftime("Total time:------ %H:%M:%S", time.gmtime(elapsed_time)))
        
        new_rows.extend(rows)
     
    dt = pd.DataFrame(columns = ['w1', 'w2', 'score'], data=new_rows)
    return dt



# ------------------------------------------------
def extract_swDTs(input_file, output_dir, dt_files, wordlist_file=None):
    
    dfv = extract_lu_data(input_file = input_file, pos='v', output_file=None)
    dfn = extract_lu_data(input_file = input_file, pos='n', output_file=None)
    dfr = extract_roles_data(input_file = input_file, output_file=None)
    
    words1 = lus_forDT(dfv)
    words2 = lus_forDT(dfn)
    words1.extend(words2)
    roles = roles_forDT(dfr)
    words1.extend(roles)
    
    words = list(set(words1))
    words.sort()
    print('# of single word lus and roles:', len(words))
    
    if wordlist_file:
        print('writing wordlist to:', wordlist_file)
        with open(wordlist_file, 'w') as fp:
            for w in words:
                    fp.write(w)
                    fp.write('\n')
                    
    dt_names = list(dt_files.keys())
    for dt_name in dt_names:
        dt_file = dt_files[dt_name]
        dt = extract_dt(dt_file, words)
        
        print('extracted rows:',len(dt))
        output_file = '{}/{}.csv.gz'.format(output_dir, dt_name)
        
        print("writing extracted dt to: ", output_file)
        write_file(dt, output_file)
 

# ------------------------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='extract relevant records drom distributional thesaurus assuming following files are present in input_dir: 1)all_fn_data.csv.gz\n. dt_dir must contain DT files. All output files will be written to output_dir')
    parser.add_argument('-i', '--input_dir', help='Directory with the input files.')
    parser.add_argument('-d', '--dt_dir', help='Directory with the DT files., can ne None, in such case input_dir will be use')

    parser.add_argument('-f', '--dt_files', help='Comma separated list (key:value) of DT files in a csv format. key will be used as output file name and value presents the dt file inside dt_dir', 
                        default='dt_59g:dt-59g-deps-wpf1k-fpw1k.csv,dt_wiki:dt-wiki-deps-jst-wpf1k-fpw1k.csv')
    parser.add_argument('-o', '--output_dir', help='Output directory.')


    args = parser.parse_args()
    print("Input: ", args.input_dir)
    print("Output: ", args.output_dir)
    print("DT_DIR: ", args.dt_dir)
    print("DT_Files: ", args.dt_files)
    
    
    MAIN_FILE = '{}/all_fn_data.csv.gz'.format(args.input_dir)
    DT_FILES = {}
    
    for e in args.dt_files.split(','):
        key,value = e.split(':')
        DT_FILES[key] = f'{args.dt_dir}/{value}'
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
#     DT_FILES ={
#         'dt_59g': '{}/dt-59g-deps-wpf1k-fpw1k.csv'.format(args.dt_dir),
#         'dt_wiki': '{}/dt-wiki-deps-jst-wpf1k-fpw1k.csv'.format(args.dt_dir),
#     } 
   

    extract_swDTs(MAIN_FILE, args.output_dir, DT_FILES)#, args.save_wordlist_file)
    

if __name__ == '__main__':
    main()
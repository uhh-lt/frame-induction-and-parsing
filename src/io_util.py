from pathlib import Path
import pandas as pd
import pickle


# ------------------------------------------------
def read_csv(input_file):

    if input_file.endswith('csv.gz'):
        df = pd.read_csv(input_file, compression='gzip')

    else:
        df = pd.read_csv(input_file)

    return df


def to_csv(df, output_file):
   
    if output_file.endswith('csv.gz'):
        df.to_csv(output_file, index=False, compression='gzip')

    else:
        df.to_csv(output_file, index=False)
    

    
def read_pickle(input_file):
    
    pickle_obj = open(input_file,'rb')
    return pickle.load(pickle_obj)



def write_pickle(obj, output_file):
    
    pickle_obj = open(output_file,'wb')
    pickle.dump(obj, pickle_obj)
    pickle_obj.close()

    

def read_file(file):  
    
    suffix = Path(file).suffix
    
    if suffix in '.pickle .pckl .pkl' : return read_pickle(file)
    if suffix == '.csv' or suffix == '.gz':   return read_csv(file)
    
    
    
def write_file(data, file):
    
    suffix = Path(file).suffix
    if suffix in '.pickle .pckl .pkl': write_pickle(data, file)
    if suffix == '.csv' :   to_csv(data, file)
    if suffix == '.gz' :    to_csv(data, file)


# ------------------------------------------------

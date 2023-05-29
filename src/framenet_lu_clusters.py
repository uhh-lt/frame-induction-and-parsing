# Use framenet to generate the lexical unit clusters of frames
# https://github.com/icsi-berkeley/ecg_framenet/
from __future__ import print_function
import os
import sys
sys.path.append('ecg_framenet')
sys.path.append('ecg_framenet/src')

print(sys.path)
import fire
import argparse
import codecs 
from collections import defaultdict
import csv
from ecg_framenet import src
from ecg_framenet.src.builder import *
from ecg_utilities import ECGUtilities as utils
from hypothesize_constructions import *
from scripts import *
import pandas as pd


verbose = False
FN_DIR = '../parser_workdir/data/open_sesame_v1_data/fndata-1.7'
DATA_FILES = '../workdir/framenet_data/'

def load_framenet(data_path):
    frame_path = data_path + "frame/"
    relation_path = data_path + "frRelation.xml"
    lu_path = data_path + "lu/"
    fnb = FramenetBuilder(frame_path, relation_path, lu_path)
    fn = fnb.read() 
    fn.build_relations()
    fn.build_typesystem()
    return fn, fnb


def add_children(fn, lus, f_target, f):
    if verbose: print(f.name, ">>>", end="")
    if len(f.children) > 0:
        for c in f.children:
            fc = fn.get_frame(c)
            lus[f_target.name] = lus[f_target.name].union(set(fc.lexicalUnits)) 
            add_children(fn, lus, f_target, fc)
    
    
def get_verbs(lu_set):
    return set([lu for lu in lu_set if ".v" in str(lu)])

        
    
def get_framenet_clusters(framenet_dir, use_children=False, verbs_only=False):
    fn, fnb = load_framenet(framenet_dir)

    lus = defaultdict(lambda: set())
    lus_count = 0
    
    for i, f in enumerate(fn.frames): 
        lus[f.name] = lus[f.name].union(set(f.lexicalUnits))
        
        if use_children:
            if verbose: print("\n")
            add_children(fn, lus, f, f)
            
        if verbs_only:
            lus[f.name] = get_verbs(lus[f.name])
        
        lus_count += len(lus[f.name])
        
    print(len(lus), lus_count)
    return lus


def save_framenet_clusters(framenet_clusters, output_fpath, frames=None):
    
    """ if frames=None, write clusters of all frames """
   
    with codecs.open(output_fpath, "w", "utf-8") as out: 
        out.write("cid\tsize\tcluster\n")
        if frames is not None:
            for f in framenet_clusters:
                if f in frames:
                    fc = ",".join([str(x)for x in framenet_clusters[f]])
                    out.write("{}\t{}\t{}\n".format(f, len(framenet_clusters[f]), fc))
        else:
            for f in framenet_clusters:
                fc = ",".join([str(x)for x in framenet_clusters[f]])
                out.write("{}\t{}\t{}\n".format(f, len(framenet_clusters[f]), fc))

    print("Output:", output_fpath)
            



def extract_verb_clusters(framenet_dir=FN_DIR, output_dir=DATA_FILES, frames=None): 
    
    save_framenet_clusters(
        get_framenet_clusters(framenet_dir, use_children=False, verbs_only=True),
        join(output_dir, "lus-wo-ch-verbs.csv"), frames)

    save_framenet_clusters(
        get_framenet_clusters(framenet_dir, use_children=True, verbs_only=True),
        join(output_dir, "lus-with-ch-r-verbs.csv"), frames)
 

def extract_all_clusters(framenet_dir=FN_DIR, output_dir=DATA_FILES, frames=None): 
    
    save_framenet_clusters(
        get_framenet_clusters(framenet_dir, use_children=False, verbs_only=False),
        join(output_dir, "lus-wo-ch.csv"), frames)

    save_framenet_clusters(
        get_framenet_clusters(framenet_dir, use_children=True, verbs_only=False),
        join(output_dir, "lus-with-ch-r.csv"), frames)
    
#Note: execute this script in same parent directory where ecg_framenet is located and change the input and output paths accordingly, placing ecg_framenet under main project directory does not work, but from home directory it works

# !python -m framenet_lu_clusters extract_verb_clusters --framenet_dir parser_workdir/data/fndata-1.7 --output_dir workdir/framenet_data
# !python -m framenet_lu_clusters extract_all_clusters --framenet_dir parser_workdir/data/fndata-1.7 --output_dir workdir/framenet_data

# ----------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire()



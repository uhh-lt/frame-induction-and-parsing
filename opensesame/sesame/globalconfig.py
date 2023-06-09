# coding=utf-8
# Copyright 2018 Swabha Swayamdipta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import sys
import os

config_json = open("configurations/global_config.json", "r")
configuration = json.load(config_json)
for key in sorted(configuration):
    sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

VERSION = str(configuration["version"])
DATA_DIR = configuration["data_directory"]
EMBEDDINGS_FILE = configuration["embeddings_file"]
DEBUG_MODE = configuration["debug_mode"]

ARGID_LR = configuration["argid_lr"]
FRAMEID_LR = configuration["frameid_lr"]
TARGETID_LR = configuration["targetid_lr"]

PARSER_DATA_DIR = DATA_DIR + "open_sesame_v1_data/fn" + VERSION
PARSER_OUTPUT_DIR = f'logs/fn{VERSION}'
TRAIN_FTE = PARSER_DATA_DIR + "fn" + VERSION +  ".fulltext.train.syntaxnet.conll"
TRAIN_EXEMPLAR = PARSER_DATA_DIR + "fn" + VERSION +  ".exemplar.train.syntaxnet.conll"
DEV_CONLL = PARSER_DATA_DIR + "fn" + VERSION +  ".dev.syntaxnet.conll"
TEST_CONLL = PARSER_DATA_DIR + "fn" + VERSION +  ".test.syntaxnet.conll"

TENSORBOARD_DIR = f'{configuration["tensorboard_dir"]}/fn{VERSION}'
# The following variables are held constant throughout the repository. Change at your own peril!

FN_DATA_DIR = DATA_DIR + "fndata-" + VERSION + "/"
LU_INDEX = FN_DATA_DIR + "luIndex.xml"
LU_DIR = FN_DATA_DIR + "lu/"
FULLTEXT_DIR = FN_DATA_DIR + "fulltext/"
FRAME_DIR = FN_DATA_DIR + "frame/"
FRAME_REL_FILE = FN_DATA_DIR + "frRelation.xml"

TEST_FILES = [
        "ANC__110CYL067.xml",
        "ANC__110CYL069.xml",
        "ANC__112C-L013.xml",
        "ANC__IntroHongKong.xml",
        "ANC__StephanopoulosCrimes.xml",
        "ANC__WhereToHongKong.xml",
        "KBEval__atm.xml",
        "KBEval__Brandeis.xml",
        "KBEval__cycorp.xml",
        "KBEval__parc.xml",
        "KBEval__Stanford.xml",
        "KBEval__utd-icsi.xml",
        "LUCorpus-v0.3__20000410_nyt-NEW.xml",
        "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
        "LUCorpus-v0.3__enron-thread-159550.xml",
        "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
        "LUCorpus-v0.3__SNO-525.xml",
        "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
        "Miscellaneous__Hound-Ch14.xml",
        "Miscellaneous__SadatAssassination.xml",
        "NTI__NorthKorea_Introduction.xml",
        "NTI__Syria_NuclearOverview.xml",
        "PropBank__AetnaLifeAndCasualty.xml",
        ]

DEV_FILES = [
        "ANC__110CYL072.xml",
        "KBEval__MIT.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
        "Miscellaneous__Hijack.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]

PTB_DATA_DIR = DATA_DIR + "ptb/"

TRAIN_FTE_CONSTITS = "fn" + VERSION + ".fulltext.train.rnng.brackets"
DEV_CONSTITS = "fn" + VERSION + ".dev.rnng.brackets"
TEST_CONSTITS = "fn" + VERSION + ".test.rnng.brackets"

CONSTIT_MAP = {}
# Label settings
UNK = "UNK"
EMPTY_LABEL = "_"
EMPTY_FE = "O"

# BIOS scheme settings
BEGINNING = 0
INSIDE = 1
OUTSIDE = 2
SINGULAR = 3

BIO_INDEX_DICT = {
        "B": BEGINNING,
        "I": INSIDE,
        EMPTY_FE: OUTSIDE,
        "S": SINGULAR
}

INDEX_BIO_DICT = {index: tag for tag, index in BIO_INDEX_DICT.items()}

# !!!change ***********
def load_experiment_configs(exp_name, data_dir=PARSER_DATA_DIR, output_dir=PARSER_OUTPUT_DIR, tensorboard_dir=TENSORBOARD_DIR, version=VERSION):
    
    
    global VERSION

    global PARSER_OUTPUT_DIR
    global PARSER_DATA_DIR
    global TENSORBOARD_DIR
    
    global FN_DATA_DIR
    global LU_INDEX
    global LU_DIR
    global FULLTEXT_DIR
    global FRAME_DIR
    global FRAME_REL_FILE
     
    global TRAIN_FTE
    global TRAIN_EXEMPLAR
    global DEV_CONLL
    global TEST_CONLL
    global CONSTIT_MAP
    
    global TRAIN_FTE_CONSTITS 
    global DEV_CONSTITS
    global TEST_CONSTITS

    VERSION = version
    
    FN_DATA_DIR = DATA_DIR + "fndata-" + VERSION + "/"
    LU_INDEX = FN_DATA_DIR + "luIndex.xml"
    LU_DIR = FN_DATA_DIR + "lu/"
    FULLTEXT_DIR = FN_DATA_DIR + "fulltext/"
    FRAME_DIR = FN_DATA_DIR + "frame/"
    FRAME_REL_FILE = FN_DATA_DIR + "frRelation.xml"

    TRAIN_FTE_CONSTITS = "fn" + VERSION + ".fulltext.train.rnng.brackets"
    DEV_CONSTITS = "fn" + VERSION + ".dev.rnng.brackets"
    TEST_CONSTITS = "fn" + VERSION + ".test.rnng.brackets"
   
    if data_dir.startswith('~'): data_dir = os.path.expanduser(data_dir)
    if data_dir.startswith('.'): data_dir = os.path.abspath(data_dir)
    if output_dir.startswith('~'): output_dir = os.path.expanduser(output_dir)
    if output_dir.startswith('.'): output_dir = os.path.abspath(output_dir)
    
    PARSER_DATA_DIR = os.path.join(data_dir, exp_name) + '/'
    PARSER_OUTPUT_DIR = os.path.join(output_dir, exp_name) + '/'
    
    if tensorboard_dir:
        if tensorboard_dir.startswith('~'): tensorboard_dir = os.path.expanduser(tensorboard_dir)
        if tensorboard_dir.startswith('.'): tensorboard_dir = os.path.abspath(tensorboard_dir)
        TENSORBOARD_DIR = os.path.join(tensorboard_dir, exp_name) + '/'
 
    
    TRAIN_FTE = PARSER_DATA_DIR + "fn" + VERSION +  ".fulltext.train.syntaxnet.conll"
    TRAIN_EXEMPLAR = PARSER_DATA_DIR + "fn" + VERSION +  ".exemplar.train.syntaxnet.conll"
    DEV_CONLL = PARSER_DATA_DIR + "fn" + VERSION +  ".dev.syntaxnet.conll"
    TEST_CONLL = PARSER_DATA_DIR + "fn" + VERSION +  ".test.syntaxnet.conll"
    
    CONSTIT_MAP = {
        TRAIN_FTE : TRAIN_FTE_CONSTITS,
        DEV_CONLL : DEV_CONSTITS,
        TEST_CONLL : TEST_CONSTITS
        }
    
    configuration = {
                    "PARSER_DATA_DIR": f"{PARSER_DATA_DIR}", 
                    "PARSER_OUTPUT_DIR" : f"{PARSER_OUTPUT_DIR}",
                    }
    
    for key in sorted(configuration):
        sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))
# !!!change ***********

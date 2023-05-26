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

from __future__ import division

import codecs
import json
import math
import numpy as np
import os
import random
import sys
import time
import tqdm
import pandas as pd
from tensorboardX import SummaryWriter
from optparse import OptionParser
import logging
from .logger import create_logger

from .globalconfig import VERSION, PARSER_DATA_DIR, PARSER_OUTPUT_DIR, TARGETID_LR

optpr = OptionParser()
optpr.add_option("--mode", dest="mode", type="choice",
                 choices=["train", "test", "refresh", "predict"], default="train")
optpr.add_option("-n", "--model_name", help="Name of model directory to save model to.")
optpr.add_option("--raw_input", type="str", metavar="FILE")
optpr.add_option("--config", type="str", metavar="FILE")
# !!!change ***********
optpr.add_option("--version", type="str", default=VERSION)
optpr.add_option("--data_dir", type="str", default=PARSER_DATA_DIR)
optpr.add_option("--output_dir", type="str", default=PARSER_OUTPUT_DIR)
optpr.add_option("--tensorboard_dir", type="str", default=None)
optpr.add_option("--exp_name", type="str")
optpr.add_option("--fixseed", action="store_true", default=False)
optpr.add_option("--num_steps", type=int, default=-1)
# !!!change ***********

(options, args) = optpr.parse_args()

# !!!change ***********
from sesame.globalconfig import load_experiment_configs
load_experiment_configs(options.exp_name, data_dir=options.data_dir, output_dir=options.output_dir, tensorboard_dir=options.tensorboard_dir, version=options.version)
from .globalconfig import VERSION, PARSER_DATA_DIR, PARSER_OUTPUT_DIR, TENSORBOARD_DIR
# to fix seed for dynet
import dynet_config
if options.fixseed:
    dynet_config.set(random_seed=2149401521)
    random.seed(2149401521)
# dynet_config.set_gpu()

# move all project imports here, after calling load_experiment_configs
from dynet import Model, LSTMBuilder, SimpleSGDTrainer, lookup, concatenate, rectify, renew_cg, dropout, log_softmax, esum, pick
from .conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, LEMDICT, LUDICT, LUPOSDICT
from .dataio import create_target_lu_map, get_wvec_map, read_conll
from .evaluation import calc_f, evaluate_example_targetid
from .frame_semantic_graph import LexicalUnit
from .globalconfig import VERSION, TRAIN_FTE, UNK, DEV_CONLL, TEST_CONLL
from .housekeeping import unk_replace_tokens
from .raw_data import make_data_instance
from .semafor_evaluation import convert_conll_to_frame_elements

# !!!change ***********
log_tensorboard = False
tb_writer1 = None
tb_writer2 = None
if options.tensorboard_dir and options.mode in ['train', 'refresh']:
    tensorboard_dir = os.path.join(TENSORBOARD_DIR, options.model_name)
    try:
        os.system(f"rm -rf {tensorboard_dir}/*")
        os.system(f"rm -rf {tensorboard_dir}/*")
    except:
        pass
    log_tensorboard = True
    tb_writer1 = SummaryWriter(logdir=f"{tensorboard_dir}/dev")
    tb_writer2 = SummaryWriter(logdir=f"{tensorboard_dir}/train")
# model_dir = "logs/{}/".format(options.model_name)
model_dir = os.path.join(PARSER_OUTPUT_DIR, options.model_name) + '/'
model_file_best = "{}best-targetid-{}-model".format(model_dir, VERSION)
model_file_name = "{}targetid-{}-model".format(model_dir, VERSION)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

train_conll = TRAIN_FTE

USE_DROPOUT = True
if options.mode in ["test", "predict"]:
    USE_DROPOUT = False

logger = create_logger("targetid_log", model_dir, level=logging.INFO)

logger.info("_____________________\n")
logger.info("COMMAND: {}\n".format(" ".join(sys.argv)))
if options.mode in ["train", "refresh"]:
    logger.info("VALIDATED MODEL SAVED TO:\t{}\n".format(model_file_best))
else:
    logger.info("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_best))
logger.info("PARSING MODE:\t{}\n".format(options.mode))
logger.info(f"FIXSEED: {options.fixseed}\n")
logger.info("_____________________\n\n")


def combine_examples(corpus_ex):
    """
    Target ID needs to be trained for all targets in the sentence jointly, as opposed to
    frame and arg ID. Returns all target annotations for a given sentence.
    """
    combined_ex = [corpus_ex[0]]
    for ex in corpus_ex[1:]:
        if ex.sent_num == combined_ex[-1].sent_num:
            current_sent = combined_ex.pop()
            target_frame_dict = current_sent.targetframedict.copy()
            target_frame_dict.update(ex.targetframedict)
            current_sent.targetframedict = target_frame_dict
            combined_ex.append(current_sent)
            continue
        combined_ex.append(ex)
    logger.info("Combined {} instances in data into {} instances.\n".format(
        len(corpus_ex), len(combined_ex)))
    return combined_ex

train_examples, _, _ = read_conll(train_conll)
combined_train = combine_examples(train_examples)

# Need to read all LUs before locking the dictionaries.
target_lu_map, lu_names = create_target_lu_map()
post_train_lock_dicts()

# Read pretrained word embeddings.
pretrained_map = get_wvec_map()
PRETRAINED_DIM = len(list(pretrained_map.values())[0])

lock_dicts()
UNKTOKEN = VOCDICT.getid(UNK)

if options.mode in ["train", "refresh"]:
    dev_examples, _, _ = read_conll(DEV_CONLL)
    combined_dev = combine_examples(dev_examples)
    out_conll_file = "{}predicted-{}-targetid-dev.conll".format(model_dir, VERSION)
elif options.mode == "test":
    dev_examples, m, t = read_conll(TEST_CONLL)
    combined_dev = combine_examples(dev_examples)
    out_conll_file = "{}predicted-{}-targetid-test.conll".format(model_dir, VERSION)
elif options.mode == "predict":
    assert options.raw_input is not None
    with open(options.raw_input, "r") as fin:
        instances = [make_data_instance(line, i) for i, line in enumerate(fin)]
    out_conll_file = "{}predicted-targets.conll".format(model_dir)
else:
    raise Exception("Invalid parser mode", options.mode)

# Default configurations.
configuration = {"train": train_conll,
                 "unk_prob": 0.1,
                 "dropout_rate": 0.01,
                 "token_dim": 100,
                 "pos_dim": 100,
                 "lemma_dim": 100,
                 "lstm_input_dim": 100,
                 "lstm_dim": 100,
                 "lstm_depth": 2,
                 "hidden_dim": 100,
                 "use_dropout": USE_DROPOUT,
                 "pretrained_embedding_dim": PRETRAINED_DIM,
                 "num_epochs": 100,
                 "patience": 25,
#                  "eval_after_every_epochs": 100,
# !!!change ***********
#                  "dev_eval_epoch_frequency": 3}
                 "dev_eval_epoch_frequency": 2000,
                 "fixseed": options.fixseed,
                 "epochs_trained":0,
                 "last_updated_epoch":0,
                 "num_steps":-1,
                 "steps_trained":0,
                 "last_updated_step":0,
                 "lr": TARGETID_LR,
                }

if options.mode == "train":
    max_epochs = configuration['num_epochs']
    max_steps = configuration['num_epochs']*len(combined_train)    
    if options.num_steps!=-1:
        max_steps = options.num_steps
        max_epochs = math.ceil(options.num_steps / len(combined_train))

    configuration["num_steps"] = max_steps
    configuration['num_epochs'] = max_epochs

    if configuration["num_steps"] <= 2000:
        configuration["dev_eval_epoch_frequency"] = 200
# !!!change ***********

configuration_file = os.path.join(model_dir, "configuration.json")
if options.mode == "train":
    if options.config:
        config_json = open(options.config, "r")
        configuration = json.load(config_json)
    with open(configuration_file, "w") as fout:
        fout.write(json.dumps(configuration, indent=2))
        fout.close()
else:
    json_file = open(configuration_file, "r")
    configuration = json.load(json_file)

UNK_PROB = configuration["unk_prob"]
DROPOUT_RATE = configuration["dropout_rate"]

TOK_DIM = configuration["token_dim"]
POS_DIM = configuration["pos_dim"]
LEMMA_DIM = configuration["lemma_dim"]
INPUT_DIM = TOK_DIM + POS_DIM + LEMMA_DIM

LSTM_INP_DIM = configuration["lstm_input_dim"]
LSTM_DIM = configuration["lstm_dim"]
LSTM_DEPTH = configuration["lstm_depth"]
HIDDEN_DIM = configuration["hidden_dim"]

NUM_EPOCHS = configuration["num_epochs"]
PATIENCE = configuration["patience"]
# EVAL_EVERY_EPOCH = configuration["eval_after_every_epochs"]
DEV_EVAL_EPOCH = configuration["dev_eval_epoch_frequency"] #* EVAL_EVERY_EPOCH

NUM_STEPS = configuration["num_steps"]
TARGETID_LR = configuration["lr"]

# !!!change ***********
if options.mode == "train":
    with open(configuration_file, "w") as fout:
        fout.write(json.dumps(configuration, indent=2))
        fout.close()

logger.info("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
for key in sorted(configuration):
    logger.info("{}:\t{}\n".format(key.upper(), configuration[key]))

logger.info("\n")

def print_data_status(fsp_dict, vocab_str):
    logger.info("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))

print_data_status(VOCDICT, "Tokens")
print_data_status(POSDICT, "POS tags")
print_data_status(LEMDICT, "Lemmas")
logger.info("\n_____________________\n\n")

logger.info(f"TRAIN_SET size \t{len(train_examples)}")
logger.info(f"TRAIN_SET size targets combined \t{len(combined_train)}")
logger.info(f"DEV_SET size \t{len(dev_examples)}")
logger.info(f"DEV_SET size targets combined \t{len(combined_dev)}")

logger.info("\n_____________________\n\n")


def get_fn_pos_by_rules(pos, token):
    """
    Rules for mapping NLTK part of speech tags into FrameNet tags, based on co-occurrence
    statistics, since there is not a one-to-one mapping.
    """
    if pos[0] == "v" or pos in ["rp", "ex", "md"]:  # Verbs
        rule_pos = "v"
    elif pos[0] == "n" or pos in ["$", ":", "sym", "uh", "wp"]:  # Nouns
        rule_pos = "n"
    elif pos[0] == "j" or pos in ["ls", "pdt", "rbr", "rbs", "prp"]:  # Adjectives
        rule_pos = "a"
    elif pos == "cc":  # Conjunctions
        rule_pos = "c"
    elif pos in ["to", "in"]:  # Prepositions
        rule_pos = "prep"
    elif pos in ["dt", "wdt"]:  # Determinors
        rule_pos = "art"
    elif pos in ["rb", "wrb"]:  # Adverbs
        rule_pos = "adv"
    elif pos == "cd":  # Cardinal Numbers
        rule_pos = "num"
    else:
        logger.info("WARNING: Rule not defined for part-of-speech {} word {} - treating as noun.".format(pos, token))
        return "n"
    return rule_pos


def check_if_potential_target(lemma):
    """
    Simple check to see if this is a potential position to even consider, based on
    the LU index provided under FrameNet. Note that since we use NLTK lemmas,
    this might be lossy.
    """
    nltk_lem_str = LEMDICT.getstr(lemma)
    return nltk_lem_str in target_lu_map or nltk_lem_str.lower() in target_lu_map


def create_lexical_unit(lemma_id, pos_id, token_id):
    """
    Given a lemma ID and a POS ID (both lemma and POS derived from NLTK),
    create a LexicalUnit object.
    If lemma is unknown, then check if token is in the LU vocabulary, and
    use it if present (Hack).
    """
    nltk_lem_str = LEMDICT.getstr(lemma_id)
    if nltk_lem_str not in target_lu_map and nltk_lem_str.lower() in target_lu_map:
        nltk_lem_str = nltk_lem_str.lower()

    # Lemma is not in FrameNet, but it could be a lemmatization error.
    if nltk_lem_str == UNK:
        if VOCDICT.getstr(token_id) in target_lu_map:
            nltk_lem_str = VOCDICT.getstr(token_id)
        elif VOCDICT.getstr(token_id).lower() in target_lu_map:
            nltk_lem_str = VOCDICT.getstr(token_id).lower()
    assert nltk_lem_str in target_lu_map
    assert LUDICT.getid(nltk_lem_str) != LUDICT.getid(UNK)

    nltk_pos_str = POSDICT.getstr(pos_id)
    rule_pos_str = get_fn_pos_by_rules(nltk_pos_str.lower(), nltk_lem_str)
    rule_lupos = nltk_lem_str + "." + rule_pos_str

    # Lemma is not seen with this pos tag.
    if rule_lupos not in lu_names:
        # Hack: replace with anything the lemma is seen with.
        rule_pos_str = list(target_lu_map[nltk_lem_str])[0].split(".")[-1]
    return LexicalUnit(LUDICT.getid(nltk_lem_str), LUPOSDICT.getid(rule_pos_str))


model = Model()
trainer = SimpleSGDTrainer(model, TARGETID_LR)

v_x = model.add_lookup_parameters((VOCDICT.size(), TOK_DIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POS_DIM))
l_x = model.add_lookup_parameters((LEMDICT.size(), LEMMA_DIM))

e_x = model.add_lookup_parameters((VOCDICT.size(), PRETRAINED_DIM))
for wordid in pretrained_map:
    e_x.init_row(wordid, pretrained_map[wordid])
# Embedding for unknown pretrained embedding.
u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')

w_e = model.add_parameters((LSTM_INP_DIM, PRETRAINED_DIM + INPUT_DIM))
b_e = model.add_parameters((LSTM_INP_DIM, 1))

w_i = model.add_parameters((LSTM_INP_DIM, INPUT_DIM))
b_i = model.add_parameters((LSTM_INP_DIM, 1))

builders = [
    LSTMBuilder(LSTM_DEPTH, LSTM_INP_DIM, LSTM_DIM, model),
    LSTMBuilder(LSTM_DEPTH, LSTM_INP_DIM, LSTM_DIM, model),
]

w_z = model.add_parameters((HIDDEN_DIM, 2 * LSTM_DIM))
b_z = model.add_parameters((HIDDEN_DIM, 1))
w_f = model.add_parameters((2, HIDDEN_DIM))  # prediction: is a target or not.
b_f = model.add_parameters((2, 1))


def identify_targets(builders, tokens, postags, lemmas, gold_targets=None):
    """
    Target identification model, using bidirectional LSTMs, with a
    multilinear perceptron layer on top for classification.
    """
    renew_cg()
    train_mode = (gold_targets is not None)

    sentlen = len(tokens)
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]
    lem_x = [l_x[lem] for lem in lemmas]

    emb2_xi = []
    for i in range(sentlen):
        if tokens[i] in pretrained_map:
            # Prevent the pretrained embeddings from being updated.
            emb_without_backprop = lookup(e_x, tokens[i], update=False)
            features_at_i = concatenate([emb_x[i], pos_x[i], lem_x[i], emb_without_backprop])
        else:
            features_at_i = concatenate([emb_x[i], pos_x[i], lem_x[i], u_x])
        emb2_xi.append(w_e * features_at_i + b_e)

    emb2_x = [rectify(emb2_xi[i]) for i in range(sentlen)]

    # Initializing the two LSTMs.
    if USE_DROPOUT and train_mode:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)
    f_init, b_init = [i.initial_state() for i in builders]

    fw_x = f_init.transduce(emb2_x)
    bw_x = b_init.transduce(reversed(emb2_x))

    losses = []
    predicted_targets = {}
    for i in range(sentlen):
        if not check_if_potential_target(lemmas[i]):
            continue
        h_i = concatenate([fw_x[i], bw_x[sentlen - i - 1]])
        score_i = w_f * rectify(w_z * h_i + b_z) + b_f
        if train_mode and USE_DROPOUT:
            score_i = dropout(score_i, DROPOUT_RATE)

        logloss = log_softmax(score_i, [0, 1])
        if not train_mode:
            is_target = np.argmax(logloss.npvalue())
        else:
            is_target = int(i in gold_targets)

        if int(np.argmax(logloss.npvalue())) != 0:
            predicted_targets[i] = (create_lexical_unit(lemmas[i], postags[i], tokens[i]), None)

        losses.append(pick(logloss, is_target))

    objective = -esum(losses) if losses else None
    return objective, predicted_targets


def print_as_conll(gold_examples, predicted_target_dict):
    """
    Creates a CoNLL object with predicted target and lexical unit.
    Spits out one CoNLL for each LU.
    """
    with codecs.open(out_conll_file, "w", "utf-8") as conll_file:
        for gold, pred in zip(gold_examples, predicted_target_dict):
            for target in sorted(pred):
                result = gold.get_predicted_target_conll(target, pred[target][0]) + "\n"
                conll_file.write(result)
        conll_file.close()


def evaluate_model(dev_iterator):
    corpus_result = [0.0, 0.0, 0.0]
    devtagged = devloss = 0.0
    predictions = []
    for didx, devex in enumerate(dev_iterator, 1):
        devludict = devex.get_only_targets()
        dl, predicted = identify_targets(
            builders, devex.tokens, devex.postags, devex.lemmas)
        if dl is not None:
            devloss += dl.scalar_value()
        predictions.append(predicted)

        devex_result = evaluate_example_targetid(list(devex.targetframedict.keys()), predicted)
        corpus_result = np.add(corpus_result, devex_result)
        devtagged += 1

    dev_p, dev_r, dev_f1 = calc_f(corpus_result)
    dev_tp, dev_fp, dev_fn = corpus_result
    
    return devloss/devtagged, (dev_p, dev_r, dev_f1), (dev_tp, dev_fp, dev_fn), predictions
    
    
    
best_dev_f1 = 0.0
# !!!change ***********
best_dev_p = 0.0
best_dev_r = 0.0
dev_eval_str=None
# !!!change ***********
if options.mode in ["refresh"]:
    logger.info("Reloading model from {} ...\n".format(model_file_best))
    model.populate(model_file_best)
    with open(os.path.join(model_dir, "best-dev-f1.txt"), "r") as fin:
#     !!!change *********** 
        lines = fin.readlines() 
    best_dev_f1 = float(lines[0].strip())
#     !!!change *********** 
    fin.close()
    logger.info("Best dev F1 so far = %.4f\n" % best_dev_f1)
    
if options.mode in ["train", "refresh"]:
    loss = 0.0
    dev_f1 = best_dev_f1
#     trainf = 0.0
    train_result = [0.0, 0.0, 0.0]

    last_updated_epoch = 0
    epochs_trained = 0
#    !!!change ***********
    steps_trained = 0
    last_updated_step = 0
    
    training_progress = {'steps':[],
                         'training_loss':[],
                         'train_loss':[],
                         'eval_loss':[],
                         'train_f1':[],
                         'eval_f1':[]
                        }
    
    if options.mode in ["refresh"]:
        
        epochs_trained = configuration['last_updated_epoch']
        last_updated_epoch = configuration['last_updated_epoch']
        
        steps_trained = configuration['last_updated_step']
        last_updated_step = configuration['last_updated_step']
            
        df = pd.read_csv(os.path.join(model_dir, "training_progress.csv"))
        df = df.loc[df['steps']<=steps_trained].copy()
        training_progress = df.to_dict(orient='list')
                         
        if log_tensorboard:
            for key in ['training_loss', 'eval_loss', 'eval_f1']:
                simple_key = key.split('_')[1] if key!="training_loss" else key
                for i in range(len(df)):      
                    tb_writer1.add_scalar(simple_key, training_progress[key][i], training_progress['steps'][i])
                         
            for key in ['training_loss', 'train_loss', 'train_f1']:
                simple_key = key.split('_')[1] if key!="training_loss" else key
                for i in range(len(df)):    
                    tb_writer2.add_scalar(simple_key, training_progress[key][i], training_progress['steps'][i])                 
#    !!!change *********** 
    starttime = time.time()
    
    epoch_iterator = tqdm.trange(0,
                                 NUM_EPOCHS,
                                 desc="TargetID Epoch")
    global_step = 0
    for epoch, _ in enumerate(epoch_iterator):
#         !!!change ***********
        epochtime = time.time()
#         !!!change ***********
        random.shuffle(combined_train)
        train_iterator = tqdm.tqdm(combined_train,
                                   desc="Train Iteration")
        trainer.status()
        for idx, trex in enumerate(train_iterator, 1):
            global_step = global_step + 1
            if global_step <= steps_trained: continue
                
            train_iterator.set_description(
                "epoch = %d steps=%d loss = %.4f val_f1 = %.4f best_val_f1 = %.4f" % (
                    epoch+1, global_step, loss/idx, dev_f1, best_dev_f1))
            inptoks = []
            unk_replace_tokens(trex.tokens, inptoks, VOCDICT, UNK_PROB, UNKTOKEN)

            trex_loss, trexpred = identify_targets(
                builders, inptoks, trex.postags, trex.lemmas, gold_targets=trex.targetframedict.keys())
            trainex_result = evaluate_example_targetid(list(trex.targetframedict.keys()), trexpred)
            train_result = np.add(train_result, trainex_result)

            if trex_loss is not None:
                loss += trex_loss.scalar_value()
                trex_loss.backward()
                trainer.update()
#                !!!change ***********
            steps_trained = global_step
            if steps_trained % DEV_EVAL_EPOCH == 0:
#             if idx % DEV_EVAL_EPOCH == 0:
                dev_iterator = tqdm.tqdm(combined_dev, desc="Dev-set Evaluation")
                dev_loss, (dev_p, dev_r, dev_f1), (dev_tp, dev_fp, dev_fn), predictions = evaluate_model(dev_iterator)
                
                # evaluate train
                tr_iterator = tqdm.tqdm(combined_train,
                              desc="Train-set Evaluation")
                train_loss, (train_p, train_r, train_f1), _, _ = evaluate_model(tr_iterator)
                                
#                 logger.info(" -- saving to {}".format(model_file_name))     
#                 model.save(model_file_name)  
                
                training_progress['steps'].append(steps_trained)
                training_progress['training_loss'].append(loss/idx)
                training_progress['train_loss'].append(train_loss)
                training_progress['eval_loss'].append(dev_loss)
                training_progress['train_f1'].append(train_f1)
                training_progress['eval_f1'].append(dev_f1)
                logger.info("\nSaving training progress.\n")
                df = pd.DataFrame.from_dict(training_progress)
                df.to_csv(os.path.join(model_dir, "training_progress.csv"), index=False)
                
                if log_tensorboard:
                    for key in ['training_loss', 'eval_loss', 'eval_f1']:
                        simple_key = key.split('_')[1] if key!="training_loss" else key
                        tb_writer1.add_scalar(simple_key, training_progress[key][-1], training_progress['steps'][-1])
                    for key in ['training_loss', 'train_loss', 'train_f1']:
                        simple_key = key.split('_')[1] if key!="training_loss" else key
                        tb_writer2.add_scalar(simple_key, training_progress[key][-1], training_progress['steps'][-1])
                        
                configuration['epochs_trained'] = epoch+1
                configuration['steps_trained'] = steps_trained

                logger.info("\nSaving configurations.\n")
                with open(configuration_file, 'w') as fout:
                    fout.write(json.dumps(configuration, indent=2))
                    fout.close() 
                  
#                 !!!change ***********

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    dev_eval_str = "[VAL best epoch=%d] loss = %.4f p = %.4f (%d/%d) r = %.4f (%d/%d) f1 = %.4f" % (
                        epoch, dev_loss, dev_p, dev_tp, dev_tp + dev_fp, dev_r, dev_tp, dev_tp + dev_fn, dev_f1)
#                     !!!change ***********
                    best_dev_p = dev_p
                    best_dev_r = dev_r
                    with open(os.path.join(model_dir, "best-dev-f1.txt"), "w") as fout:
                        fout.write("{}\n{}\n{}\n".format(best_dev_f1, best_dev_p, best_dev_r))
#                     !!!change ***********
                    logger.info(" -- saving to {}".format(model_file_best)) 
                    model.save(model_file_best)
                    print_as_conll(combined_dev, predictions)

                    last_updated_epoch = epoch+1
#                     !!!change ***********
                    last_updated_step = steps_trained
                
                    configuration['last_updated_epoch'] = last_updated_epoch                    
                    configuration['last_updated_step'] = last_updated_step
                    logger.info("\nSaving configurations.\n")
                    with open(configuration_file, 'w') as fout:
                        fout.write(json.dumps(configuration, indent=2))
                        fout.close()
                        
            if steps_trained >= NUM_STEPS:
                break
#                     !!!change ***********

# !!!change ***********
        logger.info(f"[ epoch  {epoch+1} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epochtime))} ]\n")
        configuration['epochs_trained'] = epoch+1
        configuration['steps_trained'] = steps_trained

        logger.info("\nSaving configurations.\n")
        with open(configuration_file, 'w') as fout:
            fout.write(json.dumps(configuration, indent=2))
            fout.close()
        
        if steps_trained == NUM_STEPS:
            logger.info("Training finished.\n")
            logger.info("Best model evaluation:\n{}\n".format(dev_eval_str))
            logger.info("Best model with F1 = {} saved to {}\n".format(best_dev_f1, model_file_best))
            logger.info(f"[ Training time : {time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))} ]\n")
            break
# !!!change ***********
        if epoch - last_updated_epoch > PATIENCE:
            logger.info("Ran out of patience, ending training.\n")
            logger.info("Best model evaluation:\n{}\n".format(dev_eval_str))
            logger.info("Best model saved to {}\n".format(model_file_best))
            logger.info(f"[ Training time : {time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))} ]\n")
            break
            
        loss = 0.0


elif options.mode == "test":
    logger.info("Reading model from {} ...\n".format(model_file_best))
    model.populate(model_file_best)
    corpus_tp_fp_fn = [0.0, 0.0, 0.0]

    test_predictions = []
    
    teststarttime = time.time()
    test_iterator = tqdm.tqdm(combined_dev,
                                   desc="Testing")
    for tidx, test_ex in enumerate(test_iterator, 1):
        if tidx+1 % 100 == 0:
            test_iterator.set_description(
                f"tidx = {tidx+1}")
        _, predicted = identify_targets(builders, test_ex.tokens, test_ex.postags, test_ex.lemmas)

        tp_fp_fn = evaluate_example_targetid(test_ex.targetframedict.keys(), predicted)
        corpus_tp_fp_fn = np.add(corpus_tp_fp_fn, tp_fp_fn)

        test_predictions.append(predicted)

    test_tp, test_fp, test_fn = corpus_tp_fp_fn
    test_prec, test_rec, test_f1 = calc_f(corpus_tp_fp_fn)
    logger.info("[test] p = %.4f (%.1f/%.1f) r = %.4f (%.1f/%.1f) f1 = %.4f\n" % (
        test_prec, test_tp, test_tp + test_fp,
        test_rec, test_tp, test_tp + test_fn,
        test_f1))
#     !!!change ***********
    with open(os.path.join(model_dir, "test-f1.txt"), "w") as fout:
        fout.write("f1:{}\np:{}\nr:{}\n".format(test_f1,test_prec,test_rec))
#     !!!change ***********
    logger.info(f"[ Testing time : {time.strftime('%H:%M:%S', time.gmtime(time.time() - teststarttime))} ]\n")
#     logger.info(" [took %.3fs]\n" % (time.time() - teststarttime))
    logger.info("Printing output in CoNLL format to {}\n".format(out_conll_file))
    print_as_conll(combined_dev, test_predictions)
    logger.info("Done!\n")

elif options.mode == "predict":
    logger.info("Reading model from {} ...\n".format(model_file_best))
    model.populate(model_file_best)

    predictions = []
    for instance in instances:
        _, prediction = identify_targets(builders, instance.tokens, instance.postags, instance.lemmas)
        predictions.append(prediction)
    logger.info("Printing output in CoNLL format to {}\n".format(out_conll_file))
    print_as_conll(instances, predictions)
    logger.info("Done!\n")

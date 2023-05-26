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

from .globalconfig import VERSION, PARSER_DATA_DIR, PARSER_OUTPUT_DIR, FRAMEID_LR

optpr = OptionParser()
optpr.add_option("--mode", dest="mode", type="choice", choices=["train", "test", "refresh", "predict"], default="train")
optpr.add_option("-n", "--model_name", help="Name of model directory to save model to.")
optpr.add_option("--hier", action="store_true", default=False)
optpr.add_option("--exemplar", action="store_true", default=False)
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
from .conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, FRAMEDICT, LUDICT, LUPOSDICT
from .dataio import get_wvec_map, read_conll, read_related_lus
from .evaluation import calc_f, evaluate_example_frameid
from .frame_semantic_graph import Frame
from .globalconfig import VERSION, TRAIN_FTE, UNK, DEV_CONLL, TEST_CONLL, TRAIN_EXEMPLAR
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
# !!!change ***********
model_dir = os.path.join(PARSER_OUTPUT_DIR, options.model_name) + '/'
# model_dir = "logs/{}/".format(options.model_name)
model_file_best = "{}best-frameid-{}-model".format(model_dir, VERSION)
model_file_name= "{}frameid-{}-model".format(model_dir, VERSION)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if options.exemplar:
    train_conll = TRAIN_EXEMPLAR
else:
    train_conll = TRAIN_FTE

USE_DROPOUT = True
if options.mode in ["test", "predict"]:
    USE_DROPOUT = False
USE_WV = True
USE_HIER = options.hier

logger = create_logger("frameid_log", model_dir, level=logging.INFO)

logger.info("_____________________\n")
logger.info("COMMAND: {}\n".format(" ".join(sys.argv)))
if options.mode in ["train", "refresh"]:
    logger.info("VALIDATED MODEL SAVED TO:\t{}\n".format(model_file_best))
else:
    logger.info("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_best))
logger.info("PARSING MODE:\t{}\n".format(options.mode))
logger.info(f"FIXSEED: {options.fixseed}\n")
logger.info("_____________________\n\n")


def find_multitokentargets(examples, split):
    multitoktargs = tottargs = 0.0
    for tr in examples:
        tottargs += 1
        if len(tr.targetframedict) > 1:
            multitoktargs += 1
            tfs = set(tr.targetframedict.values())
            if len(tfs) > 1:
                raise Exception("different frames for neighboring targets!", tr.targetframedict)
    logger.info("multi-token targets in %s: %.3f%% [%d / %d]\n"
                     % (split, multitoktargs*100/tottargs, multitoktargs, tottargs))

trainexamples, m, t = read_conll(train_conll)
find_multitokentargets(trainexamples, "train")

post_train_lock_dicts()
lufrmmap, relatedlus = read_related_lus()
if USE_WV:
    pretrained_embeddings_map = get_wvec_map()
    PRETRAINED_DIM = len(list(pretrained_embeddings_map.values())[0])

lock_dicts()
UNKTOKEN = VOCDICT.getid(UNK)


if options.mode in ["train", "refresh"]:
    devexamples, m, t = read_conll(DEV_CONLL)
    find_multitokentargets(devexamples, "dev/test")
    out_conll_file = "{}predicted-{}-frameid-dev.conll".format(model_dir, VERSION)
elif options.mode == "test":
    devexamples, m, t = read_conll(TEST_CONLL)
    find_multitokentargets(devexamples, "dev/test")
    out_conll_file = "{}predicted-{}-frameid-test.conll".format(model_dir, VERSION)
    fefile = "{}predicted-{}-frameid-test.fes".format(model_dir, VERSION)
elif options.mode == "predict":
    assert options.raw_input is not None
    instances, _, _ = read_conll(options.raw_input)
    out_conll_file = "{}predicted-frames.conll".format(model_dir)
else:
    raise Exception("Invalid parser mode", options.mode)

# Default configurations.
configuration = {'train': train_conll,
                 'use_exemplar': options.exemplar,
                 'use_hierarchy': USE_HIER,
                 'unk_prob': 0.1,
                 'dropout_rate': 0.01,
                 'token_dim': 100,
                 'pos_dim': 100,
                 'lu_dim': 100,
                 'lu_pos_dim': 100,
                 'lstm_input_dim': 100,
                 'lstm_dim': 100,
                 'lstm_depth': 2,
                 'hidden_dim': 100,
                 'use_dropout': USE_DROPOUT,
                 'pretrained_embedding_dim': PRETRAINED_DIM,
                 'num_epochs': 100 if not options.exemplar else 25,
                 "patience": 25,
#                  'eval_after_every_epochs': 100,
# !!!change ***********
#                  'dev_eval_epoch_frequency': 50 if options.exemplar else 5}
                 'dev_eval_epoch_frequency': 20000 if options.exemplar else 2000,
                 'fixseed': options.fixseed,
                 "epochs_trained":0,
                 "last_updated_epoch":0,
                 "num_steps":-1,
                 "steps_trained":0,
                 "last_updated_step":0,
                 "lr": FRAMEID_LR,
                 }

if options.mode == "train":
    max_epochs = configuration['num_epochs']
    max_steps = configuration['num_epochs']*len(trainexamples)    
    if options.num_steps!=-1:
        max_steps = options.num_steps
        max_epochs = math.ceil(options.num_steps / len(trainexamples))

    configuration["num_steps"] = max_steps
    configuration['num_epochs'] = max_epochs
    
    if configuration["num_steps"] <= 2000:
        configuration["dev_eval_epoch_frequency"] = 200
# !!!change ***********
configuration_file = os.path.join(model_dir, 'configuration.json')
if options.mode == "train":
    if options.config:
        config_json = open(options.config, "r")
        configuration = json.load(config_json)
    with open(configuration_file, 'w') as fout:
        fout.write(json.dumps(configuration, indent=2))
        fout.close()
else:
    json_file = open(configuration_file, "r")
    configuration = json.load(json_file)

UNK_PROB = configuration['unk_prob']
DROPOUT_RATE = configuration['dropout_rate']

TOKDIM = configuration['token_dim']
POSDIM = configuration['pos_dim']
LUDIM = configuration['lu_dim']
LPDIM = configuration['lu_pos_dim']
INPDIM = TOKDIM + POSDIM

LSTMINPDIM = configuration['lstm_input_dim']
LSTMDIM = configuration['lstm_dim']
LSTMDEPTH = configuration['lstm_depth']
HIDDENDIM = configuration['hidden_dim']

NUM_EPOCHS = configuration['num_epochs']
PATIENCE = configuration['patience']
# EVAL_EVERY_EPOCH = configuration['eval_after_every_epochs']
DEV_EVAL_EPOCH = configuration['dev_eval_epoch_frequency'] #* EVAL_EVERY_EPOCH

NUM_STEPS = configuration["num_steps"]
FRAMEID_LR = configuration["lr"]

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
print_data_status(LUDICT, "LUs")
print_data_status(LUPOSDICT, "LU POS tags")
print_data_status(FRAMEDICT, "Frames")
logger.info("\n_____________________\n\n")

logger.info(f"TRAIN_SET size \t{len(trainexamples)}")
logger.info(f"DEV_SET size \t{len(devexamples)}")

logger.info("\n_____________________\n\n")

model = Model()
trainer = SimpleSGDTrainer(model, FRAMEID_LR)
# trainer = AdamTrainer(model, 0.0001, 0.01, 0.9999, 1e-8)

v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))
lu_x = model.add_lookup_parameters((LUDICT.size(), LUDIM))
lp_x = model.add_lookup_parameters((LUPOSDICT.size(), LPDIM))
if USE_WV:
    e_x = model.add_lookup_parameters((VOCDICT.size(), PRETRAINED_DIM))
    for wordid in pretrained_embeddings_map:
        e_x.init_row(wordid, pretrained_embeddings_map[wordid])

    # Embedding for unknown pretrained embedding.
    u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')

    w_e = model.add_parameters((LSTMINPDIM, PRETRAINED_DIM+INPDIM))
    b_e = model.add_parameters((LSTMINPDIM, 1))

w_i = model.add_parameters((LSTMINPDIM, INPDIM))
b_i = model.add_parameters((LSTMINPDIM, 1))

builders = [
    LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
    LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
]

tlstm = LSTMBuilder(LSTMDEPTH, 2*LSTMDIM, LSTMDIM, model)

w_z = model.add_parameters((HIDDENDIM, LSTMDIM + LUDIM + LPDIM))
b_z = model.add_parameters((HIDDENDIM, 1))
w_f = model.add_parameters((FRAMEDICT.size(), HIDDENDIM))
b_f = model.add_parameters((FRAMEDICT.size(), 1))

def identify_frames(builders, tokens, postags, lexunit, targetpositions, goldframe=None):
    renew_cg()
    trainmode = (goldframe is not None)

    sentlen = len(tokens) - 1
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]

    emb2_xi = []
    for i in range(sentlen + 1):
        if tokens[i] in pretrained_embeddings_map:
            # If update set to False, prevents pretrained embeddings from being updated.
            emb_without_backprop = lookup(e_x, tokens[i], update=True)
            features_at_i = concatenate([emb_x[i], pos_x[i], emb_without_backprop])
        else:
            features_at_i = concatenate([emb_x[i], pos_x[i], u_x])
        emb2_xi.append(w_e * features_at_i + b_e)

    emb2_x = [rectify(emb2_xi[i]) for i in range(sentlen+1)]

    # initializing the two LSTMs
    if USE_DROPOUT and trainmode:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)
    f_init, b_init = [i.initial_state() for i in builders]

    fw_x = f_init.transduce(emb2_x)
    bw_x = b_init.transduce(reversed(emb2_x))

    # only using the first target position - summing them hurts :(
    targetembs = [concatenate([fw_x[targetidx], bw_x[sentlen - targetidx - 1]]) for targetidx in targetpositions]
    targinit = tlstm.initial_state()
    target_vec = targinit.transduce(targetembs)[-1]

    valid_frames = list(lufrmmap[lexunit.id])
    chosenframe = valid_frames[0]
    logloss = None
    if len(valid_frames) > 1:
        if USE_HIER and lexunit.id in relatedlus:
            lu_vec = esum([lu_x[luid] for luid in relatedlus[lexunit.id]])
        else:
            lu_vec = lu_x[lexunit.id]
        fbemb_i = concatenate([target_vec, lu_vec, lp_x[lexunit.posid]])
        # TODO(swabha): Add more Baidu-style features here.
        f_i = w_f * rectify(w_z * fbemb_i + b_z) + b_f
        if trainmode and USE_DROPOUT:
            f_i = dropout(f_i, DROPOUT_RATE)

        logloss = log_softmax(f_i, valid_frames)

        if not trainmode:
            chosenframe = np.argmax(logloss.npvalue())

    if trainmode:
        chosenframe = goldframe.id

    losses = []
    if logloss is not None:
        losses.append(pick(logloss, chosenframe))

    prediction = {tidx: (lexunit, Frame(chosenframe)) for tidx in targetpositions}

    objective = -esum(losses) if losses else None
    return objective, prediction

def print_as_conll(goldexamples, pred_targmaps):
    with codecs.open(out_conll_file, "w", "utf-8") as f:
        for g,p in zip(goldexamples, pred_targmaps):
            result = g.get_predicted_frame_conll(p) + "\n"
            f.write(result)
        f.close()


def evaluate_model(dev_iterator):
    corpus_result = [0.0, 0.0, 0.0]
    devtagged = devloss = 0.0
    predictions = []
    
    for didx, devex in enumerate(dev_iterator, 1):
        devludict = devex.get_only_targets()
        dl, predicted = identify_frames(
            builders, devex.tokens, devex.postags, devex.lu, list(devex.targetframedict.keys()))
        if dl is not None:
            devloss += dl.scalar_value()
        predictions.append(predicted)

        devex_result = evaluate_example_frameid(devex.frame, predicted)
        corpus_result = np.add(corpus_result, devex_result)
        devtagged += 1

    dev_p, dev_r, dev_f1 = calc_f(corpus_result)
    dev_tp, dev_fp, dev_fn = corpus_result
    
    return devloss/devtagged, (dev_p, dev_r, dev_f1), (dev_tp, dev_fp, dev_fn), predictions
    
    
best_dev_f1 = 0.0
# !!!change ***********
best_dev_p = 0.0
best_dev_r = 0.0
# !!!change ***********

if options.mode in ["refresh"]:
    logger.info("Reloading model from {} ...\n".format(model_file_best))
    model.populate(model_file_best)
    with open(os.path.join(model_dir, "best-dev-f1.txt"), "r") as fin:
#    !!!change ***********    
        lines = fin.readlines()
    best_dev_f1 = float(lines[0].strip())
#    !!!change ***********    
    fin.close()
    logger.info("Best dev F1 so far = %.4f\n" % best_dev_f1)

if options.mode in ["train", "refresh"]:
    loss, current_loss = 0.0, 0.0
    dev_f1 = best_dev_f1
    dev_tp = dev_fp = 0
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
                                 desc="FrameID Epoch")
    global_step = 0
    for epoch, _ in enumerate(epoch_iterator):
#         !!!change ***********
        epochtime = time.time()
#         !!!change ***********
        random.shuffle(trainexamples)
        train_iterator = tqdm.tqdm(trainexamples,
                              desc="Train Iteration")
        trainer.status()
        for idx, trex in enumerate(train_iterator, 1):
            global_step = global_step + 1
            if global_step <= steps_trained: continue
            train_iterator.set_description(
                "epoch = %d  steps = %d  loss = %.4f val_f1 = %.4f best_val_f1 = %.4f" % (
                    epoch+1, global_step, loss/idx, dev_f1, best_dev_f1))

            inptoks = []
            unk_replace_tokens(trex.tokens, inptoks, VOCDICT, UNK_PROB, UNKTOKEN)

            trexloss,_ = identify_frames(
                builders, inptoks, trex.postags, trex.lu, list(trex.targetframedict.keys()), trex.frame)

            if trexloss is not None:
                current_loss = trexloss.scalar_value()
                loss += current_loss
                trexloss.backward()
                try:
                    trainer.update()
                except Exception as ex:
                    print("\n******************")
                    print('loss:', current_loss)
                    print(trainer.status())
                    print("\n******************")
                    raise ex
#             !!!change ***********        
            steps_trained = global_step
            if steps_trained % DEV_EVAL_EPOCH == 0:
#             if idx % DEV_EVAL_EPOCH == 0:
                dev_iterator = tqdm.tqdm(devexamples, desc="Dev-set Evaluation")
                dev_loss, (dev_p, dev_r, dev_f1), (dev_tp, dev_fp, dev_fn), predictions = evaluate_model(dev_iterator)
                
                # evaluate train
                tr_iterator = tqdm.tqdm(trainexamples,
                              desc="Train-set Evaluation")
                train_loss, (train_p, train_r, train_f1), _ , _ = evaluate_model(tr_iterator)
                                
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
                                        
#                !!!change ***********
                
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
#                     !!!change ***********
                    best_dev_p = dev_p
                    best_dev_r = dev_r
                    with open(os.path.join(model_dir, "best-dev-f1.txt"), "w") as fout:
                        fout.write("{}\n{}\n{}\n".format(best_dev_f1, best_dev_p, best_dev_r))
#                     !!!change ***********
                        fout.close()
    
                    print_as_conll(devexamples, predictions)
                    logger.info(" -- saving to {}".format(model_file_best)) 
                    model.save(model_file_best)
                    last_updated_epoch = epoch+1
#                     !!!change ***********
                    last_updated_step = steps_trained
                    configuration['last_updated_epoch'] = last_updated_epoch                    
                    configuration['last_updated_step'] = last_updated_step
                    logger.info("\nSaving configurations.\n")
                    with open(configuration_file, 'w') as fout:
                        fout.write(json.dumps(configuration, indent=2))
                        fout.close()
                        
            if steps_trained == NUM_STEPS:
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
            logger.info("Best model with F1 = {} saved to {}\n".format(best_dev_f1, model_file_best))
            logger.info(f"[ Training time : {time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))} ]\n")
            break
# !!!change ***********        
        if epoch - last_updated_epoch > PATIENCE:
            logger.info("Ran out of patience, ending training.\n")
            logger.info("Best model with F1 = {} saved to {}\n".format(best_dev_f1, model_file_best))
            logger.info(f"[ Training time : {time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))} ]\n")
            break
            
        loss = 0.0
    
elif options.mode == "test":
    logger.info("Loading model from {} ...\n".format(model_file_best))
    model.populate(model_file_best)
    corpus_tpfpfn = [0.0, 0.0, 0.0]

    testpredictions = []

    sn = devexamples[0].sent_num
    sl = [0.0,0.0,0.0]
    prediction_logger = open("{}/frameid-prediction-analysis.log".format(model_dir), "w")
    prediction_logger.write("Sent#%d :\n" % sn)
    devexamples[0].print_internal_sent(prediction_logger)
    
    teststarttime = time.time()
    test_iterator = tqdm.tqdm(devexamples,
                                   desc="Testing")
    for tidx, testex in enumerate(test_iterator, 1):
        if tidx+1 % 100 == 0:
            test_iterator.set_description(
                f"tidx = {tidx+1}")
        _, predicted = identify_frames(builders, testex.tokens, testex.postags, testex.lu, list(testex.targetframedict.keys()))

        tpfpfn = evaluate_example_frameid(testex.frame, predicted)
        corpus_tpfpfn = np.add(corpus_tpfpfn, tpfpfn)

        testpredictions.append(predicted)

        sentnum = testex.sent_num
        if sentnum != sn:
            lp, lr, lf = calc_f(sl)
            prediction_logger.write("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
                  "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
                  "\n-----------------------------\n"
                  % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
                     sn,
                     lr, sl[0], sl[-1] + sl[0],
                     lp, sl[0], sl[1] + sl[0],
                     lf))
            sl = [0.0,0.0,0.0]
            sn = sentnum
            prediction_logger.write("Sent#%d :\n" % sentnum)
            testex.print_internal_sent(prediction_logger)

        prediction_logger.write("gold:\n")
        testex.print_internal_frame(prediction_logger)
        prediction_logger.write("prediction:\n")
        testex.print_external_frame(predicted, prediction_logger)

        sl = np.add(sl, tpfpfn)
        prediction_logger.write("{} / {} / {}\n".format(tpfpfn[0], tpfpfn[0]+tpfpfn[1], tpfpfn[0]+tpfpfn[-1]))

    # last sentence
    lp, lr, lf = calc_f(sl)
    prediction_logger.write("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
          "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
          "\n-----------------------------\n"
          % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
             sentnum,
             lp, sl[0], sl[1] + sl[0],
             lr, sl[0], sl[-1] + sl[0],
             lf))

    testp, testr, testf = calc_f(corpus_tpfpfn)
    testtp, testfp, testfn = corpus_tpfpfn
    logger.info("[test] p = %.4f (%.1f/%.1f) r = %.4f (%.1f/%.1f) f1 = %.4f\n" % (
        testp, testtp, testtp + testfp,
        testr, testtp, testtp + testfp,
        testf))
#     !!!change ***********
    with open(os.path.join(model_dir, "test-f1.txt"), "w") as fout:
        fout.write("f1:{}\np:{}\nr:{}\n".format(testf,testp,testr))
#     !!!change ***********
    logger.info(f"[ Testing time : {time.strftime('%H:%M:%S', time.gmtime(time.time() - teststarttime))} ]\n")
#     logger.info(" [took %.3fs]\n" % (time.time() - teststarttime))
    logger.info("Printing output conll to " + out_conll_file + " ... ")
    print_as_conll(devexamples, testpredictions)
    logger.info("Done!\n")

    logger.info("Printing frame-elements to " + fefile + " ...\n")
    convert_conll_to_frame_elements(out_conll_file, fefile)
    logger.info("Done!\n")
    prediction_logger.close()

elif options.mode == "predict":
    logger.info("Loading model from {} ...\n".format(model_file_best))
    model.populate(model_file_best)

    predictions = []
    for instance in instances:
        _, prediction = identify_frames(builders, instance.tokens, instance.postags, instance.lu, list(instance.targetframedict.keys()))
        predictions.append(prediction)
    logger.info("Printing output in CoNLL format to {}\n".format(out_conll_file))
    print_as_conll(instances, predictions)
    logger.info("Done!\n")

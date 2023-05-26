import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import time
from pathlib import Path

from .dataio import read_conll, get_wvec_map, read_ptb, read_frame_maps, read_frame_relations
from .conll09 import VOCDICT, FRAMEDICT, FEDICT, LUDICT, LUPOSDICT, DEPRELDICT, CLABELDICT, POSDICT, LEMDICT, lock_dicts, post_train_lock_dicts
from .globalconfig import VERSION, TRAIN_EXEMPLAR, TRAIN_FTE, TRAIN_FTE_CONSTITS, UNK, EMPTY_LABEL, EMPTY_FE, TEST_CONLL, DEV_CONLL, ARGID_LR
from .evaluation import evaluate_corpus_argid


import logging
log = logging.getLogger()


frmfemap, corefrmfemap, _ = read_frame_maps()
# Hack to handle FE in version 1.5 annotation!
frmfemap[FRAMEDICT.getid("Measurable_attributes")].append(FEDICT.getid("Dimension"))
frmfemap[FRAMEDICT.getid("Removing")].append(FEDICT.getid("Frequency"))


lock_dicts()
UNKTOKEN = VOCDICT.getid(UNK)
NOTANLU = LUDICT.getid(EMPTY_LABEL)
NOTANFEID = FEDICT.getid(EMPTY_FE)  # O in CoNLL format.


def print_eval_results(eval_results, work_dir):
    corp_up, corp_ur, corp_uf, \
    corp_lp, corp_lr, corp_lf, \
    corp_wp, corp_wr, corp_wf, \
    corp_ures, corp_labldres, corp_tokres = eval_results

    sys.stderr.write("\n[test] wpr = %.5f (%.1f/%.1f) wre = %.5f (%.1f/%.1f)\n"
                     "[test] upr = %.5f (%.1f/%.1f) ure = %.5f (%.1f/%.1f)\n"
                     "[test] lpr = %.5f (%.1f/%.1f) lre = %.5f (%.1f/%.1f)\n"
                     "[test] wf1 = %.5f uf1 = %.5f lf1 = %.5f\n"
                     % (corp_wp, corp_tokres[0], corp_tokres[1] + corp_tokres[0],
                        corp_wr, corp_tokres[0], corp_tokres[-1] + corp_tokres[0],
                        corp_up, corp_ures[0], corp_ures[1] + corp_ures[0],
                        corp_ur, corp_ures[0], corp_ures[-1] + corp_ures[0],
                        corp_lp, corp_labldres[0], corp_labldres[1] + corp_labldres[0],
                        corp_lr, corp_labldres[0], corp_labldres[-1] + corp_labldres[0],
                        corp_wf, corp_uf, corp_lf))

    with open(Path(work_dir) / "test-f1.txt", "w") as fout:
        fout.write("lf1:{},lpr:{},lre:{}\n".format(corp_lf, corp_lp, corp_lr))
        fout.write("uf1:{},upr:{},ure:{}\n".format(corp_uf, corp_up, corp_ur))
        fout.write("wf1:{},wpr:{},wre:{}\n".format(corp_wf, corp_wp, corp_wr))


def evaluate_parser(config, work_dir):
    log.info('Loading the gold standard corpus.')
    gold_answers, _, _ = read_conll(Path(config.data.data_dir) / config.data.test_name, None)
    log.info('Done.')
    
    log.info('Loading the predictions.')
    pred_answers, _, _ = read_conll(config.data.pred_answers_path, None)
    pred_answers = [e.invertedfes for e in pred_answers]
    log.info('Done.')
    
    with open(Path(work_dir) / 'argid-prediction-analysis.log', "w") as opensesame_logger:
        eval_res = evaluate_corpus_argid(
            gold_answers, pred_answers, corefrmfemap, NOTANFEID, opensesame_logger)
    
    print_eval_results(eval_res, work_dir)
    
        
@hydra.main(config_name=os.environ['HYDRA_CONFIG_PATH'])
def main(config : DictConfig) -> None:
    auto_generated_dir = os.getcwd()
    log.info(f'Work dir: {auto_generated_dir}')
    os.chdir(hydra.utils.get_original_cwd())
    
    evaluate_parser(config, auto_generated_dir)


if __name__ == "__main__":
    main()
    
from transformers.models.bert.tokenization_bert import (
    BertTokenizer,
)  # without it segmentation fault
import torch

import hydra
from omegaconf import DictConfig, OmegaConf
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Vocabulary
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import SlantedTriangular
from allennlp.data import allennlp_collate
from allennlp.training.util import evaluate
from allennlp.training.moving_average import ExponentialMovingAverage

from allennlp_models.structured_prediction.predictors.srl import (
    SemanticRoleLabelerPredictor,
)
from allennlp_models.structured_prediction.models.srl_bert import SrlBert

from srl_conll_reader_custom import SrlConllReaderCustom

import torch.optim as optim
from torch.utils.data import DataLoader as DataLoaderTorch

from transformers import AdamW
from transformers import set_seed
import transformers

from datetime import datetime
import math
from tqdm import tqdm
import yaml


import logging

log = logging.getLogger()


def train(config, work_dir, reader):
    log.info("Loading datasets...")
    train_dataset = list(reader.read(config.data.train_data_path))
    dev_dataset = list(reader.read(config.data.dev_data_path))
    test_dataset = list(
        reader.read(config.data.test_data_path)
    )  # For gathering the vocabulary of all roles
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset + test_dataset)
    log.info("Done.")
    
#     log.info("Saving config...")
#     with open(f'{config.hydra.run.dir}/configs.yaml', 'w') as fp:
#         yaml.dump(config, fp)
#     log.info("Done.")
    
    log.info("Loading model...")
    model = SrlBert(vocab, config.model.model_name)
    model = model.cuda()
    log.info("Done.")
    # set_seed(config.seed)
    steps_per_epoch = math.ceil(len(train_dataset) / config.training.bs)
    
    num_epochs = config.training.num_epochs
    total_steps = steps_per_epoch*num_epochs
    train_steps = config.training.train_steps
    if train_steps >= 0:
        total_steps =  math.ceil( train_steps / config.training.bs)
        num_epochs = math.ceil(total_steps / steps_per_epoch)
    
    warmup_steps = math.ceil(
        steps_per_epoch * num_epochs * config.training.warmup_ratio
    )
    log.info(f'steps_per_epoch:{steps_per_epoch}, num_epochs:{num_epochs}, total_steps:{total_steps}, train_dataset:{len(train_dataset)}')
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    train_data_loader = SimpleDataLoader(
        instances=train_dataset, batch_size=config.training.bs
    )
    train_data_loader.index_with(vocab)

    val_data_loader = SimpleDataLoader(
        instances=dev_dataset, batch_size=config.model.ebs
    )
    val_data_loader.index_with(vocab)

    lr_scheduler = LinearWithWarmup(
        optimizer,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        num_steps_per_epoch=steps_per_epoch,
    )

    ema_holder = None
    if config.training.ema_decay:
        ema_holder = ExponentialMovingAverage(
            model.named_parameters(), decay=config.training.ema_decay
        )

    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_data_loader,
        validation_data_loader=(
            val_data_loader if config.training.enable_validation else None
        ),
        validation_metric="+f1-measure-overall",
        num_epochs=num_epochs,
        cuda_device=torch.device("cuda:0"),
        learning_rate_scheduler=lr_scheduler,
        num_gradient_accumulation_steps=config.training.accum,
        serialization_dir=work_dir,
        grad_clipping=1.0,
        moving_average=ema_holder,
    )

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        pass

    if config.training.ema_decay:
        ema_holder.assign_average_value()

    if config.training.ema_decay:
        log.info("Updating batch norm parameters...")
        torch.optim.swa_utils.update_bn(train_data_loader, model)
        log.info("Done.")

    torch.save(model, Path(work_dir) / "pytorch_model.bin")
    vocab.save_to_files(Path(work_dir) / "vocab")

    return model, vocab


def save_in_conll_format(sents, output_file_path):
    with open(output_file_path, "w") as f:
        for i, sent in enumerate(sents):
            f.write("\n".join(["\t".join(w) for w in sent]))

            if i < len(sents) - 1:
                f.write("\n\n")


def save_results_conll(orig_file_path, predictions, output_file_path, use_predicate):
    with open(orig_file_path) as f:
        lines = f.readlines()

    sents = []
    sent = []
    for line in lines:
        line = line.rstrip()

        if line:
            sent.append(line.split("\t"))
        else:
            sents.append(sent)
            sent = []

    sents.append(sent)

    for i, pred in enumerate(predictions):
        if use_predicate:
            pred_tags = pred["tags"][:-2]
        else:
            pred_tags = pred["tags"]

        for j, tag in enumerate(pred_tags):
            sents[i][j][-1] = tag

    save_in_conll_format(sents, output_file_path)


def predict(config, work_dir, reader, model, vocab):
    pred_dataset = list(reader.read(config.data.pred_data_path))

    predictor = SemanticRoleLabelerPredictor(model, reader)
    pred_data_loader = DataLoaderTorch(
        pred_dataset, batch_size=config.model.ebs, shuffle=False, collate_fn=lambda a: a
    )

    preds = []
    for batch in tqdm(pred_data_loader):
        res = predictor.predict_instances(batch)
        preds += res["verbs"]
#     import pickle
#     pkl_file = Path(work_dir) / "preds.pkl"
#     with open(pkl_file, 'wb') as fp:
#         pickle.dump(preds, fp)
    save_results_conll(
        config.data.pred_data_path,
        preds,
        Path(work_dir) / "preds.conll",
        config.model.use_predicate,
    )


@hydra.main(config_name=os.environ["HYDRA_CONFIG_PATH"])
def main(config: DictConfig) -> None:
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    reader = SrlConllReaderCustom(
        bert_model_name=config.model.model_name,
        use_predicate=config.model.use_predicate,
    )

    # set_seed(config.seed) # I get a segfault for this

    model = None
    vocab = None
    if config.do_train:
        log.info("Start training.")

        model, vocab = train(config, work_dir=auto_generated_dir, reader=reader)

    if config.do_predict:
        log.info("Start predicting.")

        if model is None:
            log.info("Loading model...")
            model = torch.load(
                Path(config.model.model_path) / "pytorch_model.bin"
            ).cuda()
            vocab = Vocabulary.from_files(Path(config.model.model_path) / "vocab")
            log.info("Done.")

        predict(
            config, work_dir=auto_generated_dir, reader=reader, model=model, vocab=vocab
        )


if __name__ == "__main__":
    main()

# Semantic frame induction and parsing

This repository contains codes and datasets to reproduce results in the article 'Text augmentation for semantic frame induction and parsing'

**Download Data:** 
All pre-processed datasets, DTs and embedding files for non-contextualized models and Melamud's can be downloded [here](https://ltnas1.informatik.uni-hamburg.de:8081/owncloud/index.php/s/O3LftEWCil0s9Kq?path=%2F)

## Intrinsic Evaluation -- Augmenting Framenet Descriptions
### Required modules:
#### - ecg_framenet
Source: https://github.com/icsi-berkeley/ecg_framenet/ \
This library was used to aggreagte lexical-units for each frame in the FrameNet. Which is required to create gold term sets for final evaluation datasets. \

Pre-extracted files can be seen in [workdir/framenet_data](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/workdir/framenet_data) directory.

### Download and Preprocess FrameNet data
a) Can be requested from the FrameNet publisher: https://framenet.icsi.berkeley.edu/fndrupal/framenet_request_data \
b) via NLTK interface: 
```
import nltk
nltk.download(framenet_v17)
```
This will download the data into your home directory at: nltk_data/corpora/framenet_v17, rename it to 'fndata1.7' and move it to: parser_workdir/data directory

#### Preprocess
Assuming the FrameNet 1.7 data is downloaded and located under parser_workdir/data/fndata-1.7 directory. To create evaluations datasets, we first extracted all data from fulltext and lu (exemplars) subdirectories, which contains frame annotations in XML documents.\
To extract and preprocess framenet data, execute:
```
! python -m src.extract_framenet_data --input_dir=parser_workdir/data/fndata-1.7 --output_dir=workdir/framenet_data
```
This will create all required files and save them into output_dir. Now move to create evaluation datasets for intrinsic evaluation

###  Evaluation datasets for lexical expansion [Intrinsic Evaluation]
```
!python -m src.datasets_util create_source_datasets --input_dir='workdir/framenet_data' --output_dir='workdir/framenet_data' --data_types 'verbs,nouns,roles'

!python -m src.datasets_util create_final_datasets --input_dir='workdir/framenet_data' --output_dir='workdir/data' --data_types 'verbs,nouns,roles'
```
The final command will produce single word/token swv_T.pkl (verbs), swn_T.pkl(nouns) and swr_T.pkl(roles) datasets along with variations of dynamic patterns TandT etc, and their respective gold datasets. For relevant commands see: [create_all_datasets.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/create_all_datasets.ipynb)


You can also download all source and final datasets from [here](https://ltnas1.informatik.uni-hamburg.de:8081/owncloud/index.php/s/O3LftEWCil0s9Kq?path=%2Fworkdir%2Fdata) and [here](https://ltnas1.informatik.uni-hamburg.de:8081/owncloud/index.php/s/O3LftEWCil0s9Kq?path=%2Fworkdir%2Fframenet_data)


 
### Pre-trained models

- ##### Distributional Thesauri (DTs)
Original DTs are very large, we have already processed them for all single-token data from our evaluations datasets (verbs lexical units, nouns lexical units, semantic-roles) using following command:
```
!python -m src.dt --input_dir 'workdir/framenet_data' --dt_dir 'workdir' --output_dir 'workdir/dt'
```

These preprocessed DTs can eb downloaded [here](https://ltnas1.informatik.uni-hamburg.de:8081/owncloud/index.php/s/O3LftEWCil0s9Kq?path=%2Fworkdir%2Fdt) and should be saved to wordir/dsm directory. To add more DTs, see the module [src.dt](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/src/dt.py):

- #### Non-contextualized embeddings
can be dowloaded [here](https://ltnas1.informatik.uni-hamburg.de:8081/owncloud/index.php/s/O3LftEWCil0s9Kq?path=%2Fworkdir%2Fdsm), and should be saved to workdir/dsm directory. 

- #### Melamud
word and context embeddings can be downloaded from [here](https://ltnas1.informatik.uni-hamburg.de:8081/owncloud/index.php/s/O3LftEWCil0s9Kq?path=%2Fworkdir%2Fmelamud_lexsub) and sould be saved to workdir/melamud_lexsub directory. 
- #### Transformer-based models
use transformers library

### Experiments and and Evaluation

#### Predicting substitutes
- #### Non-contextualized models
```
!python -m src.run_predict \
--model_type='dt+workdir/dt_wiki.csv.gz' \
--dataset='workdir/data/swv_T.pk' \
--proc_column='masked_sent' \
--result_dir='workdir/results/paper_verbs_st/dt_wiki_nolem' \
--n_top=200 \
--do_lemmatize=False \
--batch_size=100000000000000
```
For word2vec and GloVe embeddings, replace the prefix 'dt' with 'dsm' for model_type, for fastText use the prefix 'fastext'. And change the path to vector file after '+'. Like this: "fasttext+workdir/dsm/fasttext_cc/cc.en.300.bin"
- #### Melamud

1. Execute the following command to extract relevant context for the target words:
```
Assuming StanfordCoreNLP server is running at port 9000. Execute the command:

! python -m src.context_extractor --input_file workdir/data/swv_T.pkl --output_file workdir/data/swv_Tp.pkl --jobs 16 --port 9000
```
```
2. Now execute the following command to predict substitutes using this context and the target word:

! python -m src.run_melamud_parallel --input_file workdir/data/swv_Tp.pkl --result_dir workdir/results/paper_verbs_st/melamud_balmult --metric balmult --jobs 36
```

- #### Transformer-based models
```
!python -m src.run_predict \
--model_type='bert-large-cased' \
--dataset='workdir/data/swv_T.pk' \
--proc_column='masked_sent' \
--result_dir='workdir/results/paper_verbs_st/blc-ntok1-mask-k200' \
--n_top=200 \
--mask_token=True \
--batch_size=10 
```
For dynamic patterns, just change the dataset.

- #### +embs models
We used [LexSubGen](https://github.com/Samsung/LexSubGen) to run experiments for XLNet and any +embs variants of BERT and XLNet.

#### Running multiple experiments
Execute the command:
```
!python -m src.run_experiments \
--config=workdir/experiment_configs/verb_preds_st.json \
--cuda_devices=0,1,0,1
```
How to define experiment configurations for these experiments: see [generate_experiment_configs.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/generate_experiment_configs.ipynb)


#### Postprocess

One example run for nouns lexical units is as follows:
```
!python -m  src.run_postprocessing_predictions --n_jobs=24 \
--gold_path=./workdir/data/swn_gold_dataset.pkl \
--test_indexes_path=workdir/framenet/swn_gold_dataset_test_split.json \
--results_path=./workdir/results/paper_nouns_st \
--proc_funcs='lemmatize,clean_noisy,remove_noun_stopwords,filter_nouns'\
--save_results_path=./workdir/results/test-paper_nouns_st_pattern_nounfilter\
--parser='pattern'\
--dataset_type='nouns'

```
default **parser** is 'pattern', other option is 'lemminflect' and 'nltk'. \
**dataset_type** can be 'nouns', 'roles', 'verbs' \
For more explanation see: [postprocess-nouns.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/postprocess-nouns.ipynb), [postprocess-roles.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/postprocess-roles.ipynb), [postprocess-verbs.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/postprocess-verbs.ipynb)


#### Evaluate

```
!python -m src.run_evaluate \
--results_path=$RESULTS_PATH
```
you can additonally pass a comma separated list of experiments to evaluate only that subset using **exp_names** parameter.
For examples see [paper_results.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/paper_results.ipynb)

#### Upperbound results: 
Relevant scripts: [upperbound.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/upperbound.ipynb) \
Results: [workdir/upperbound](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/workdir/upperbound)


### Manual evaluation datasets and results

Relevant scripts: [src.create_datasets_manual_evaluation](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/src/create_datasets_manual_evaluation.py) and [manual_evaluation.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/blob/main/manual_evaluation.ipynb)

Manual annotations: https://docs.google.com/spreadsheets/d/1me9YNaQpXJZ0p6pupd-IdmXTJ8AxbeavTIndfeROMpA/edit?usp=sharing

Final results: [workdir/manual_evaluation](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/workdir/manual_evaluation)


## Extrnisic Evaluation -- Frame-Semantic Parsing
### Parsers
#### Opensesame parser
Source: https://github.com/swabhs/open-sesame
Source code of this parser was slighlty modified, so better to use the code provided within our repository \
See [run_opensesame_parser.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/run_opensesame_parser.ipynb) for required data, and basic commands to run this parser
Other notebooks are as follows:

- [configs_parser.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/configs_parser.ipynb) explains how to generate configurations for augmentations and parser experiments 
- Results tables and figures for this part are created here: [results_opensesame_parser-VERBS.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/results_opensesame_parser-VERBS.ipynb), [results_opensesame_parser-NOUNS.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/results_opensesame_parser-NOUNS.ipynb)

#### Bert SRL parser
[srl_parser](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/srl_parser) contains code, and running example [run_parser_example.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/srl_parser/run_parser_example.ipynb) of this parser
- Results tables and figures for this part are created here: [results_bertsrl_parser-VERBS.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/results_bertsrl_parser-VERBS.ipynb)

### Datasets
Datasets for these experiments are available in [parser_workdir/data/fn1.7](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/parser_workdir/data/open_sesame_v1_data/fn1.7) 

The process to create these datasets is explained in notebooks as follows:

- [DataAugmentation.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/DataAugmentation.ipynb) : Relevant methods to mask conll data using different configurations, predict substitutes for masked words, their postprocessing and producing final augmented conll data
Here, the term mask means marking target word for predicting substitutes

- [DataSampling.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/DataSampling.ipynb) : How to sample data for augmentations

- [Mask-and-Postprocess.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/Mask-and-Postprocess.ipynb): Explains how, rather than doing end-to-end mask-predict-postprocess-augment cycle for each substitution model, creating a master file for each type of dataset (verbs, nouns) and then doing this cycle once on this file save times, and all configurations can be extracted from this file.

- [configs_augment.ipynb](https://github.com/uhh-lt/frame-induction-and-parsing/tree/main/opensesame/configs_augment.ipynb) explains how the datasets can eb generated 

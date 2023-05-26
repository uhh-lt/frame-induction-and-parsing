## Data location:
We assume that data is located at ```<project_root>/parser_workdir/data/open_sesame_v1_data/fn1.7```

## Run for training:  
From the srl_parser directory, run:  
```
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=./run_srl_parser.yaml python ./run_srl_parser.py do_train=True do_predict=True data.pred_data_path=../../parser_workdir/data/fn1.7_conll/fn1.7.test.syntaxnet.conll
```

## Run for prediction:  
From the srl_parser directory, run: 
```
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=./run_srl_parser.yaml python ./run_srl_parser.py do_train=False do_predict=True data.pred_data_path=../../parser_workdir/data/fn1.7_conll/fn1.7.test.syntaxnet.conll model.model_path=<path to the working dir for training>
```

## Get scores:  
From the opensesame directory, run:  
```
HYDRA_CONFIG_PATH=../configurations/run_evaluate_srlparser.yaml python -m sesame.run_evaluate data.pred_answers_path=../parser_workdir/srl_parser/2021-03-14/23-21-56/preds.conll data.test_name=fn1.7.test.syntaxnet.conll
```
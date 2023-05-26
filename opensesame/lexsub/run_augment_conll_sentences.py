import json
import fire
import os

import multiprocessing as mp

from collections.abc import Iterable
    
    
DEVICE_QUEUE = mp.Queue()
WORKER_DEVICE = None

def initialize_worker():
    global DEVICE_QUEUE
    global WORKER_DEVICE
    WORKER_DEVICE = DEVICE_QUEUE.get()
    print('Worker device:', WORKER_DEVICE)
    
    
def run_exp(*args, **kwargs):
    config = args[0]
    exp_name = config['name']
    arguments = ' '.join((f'--{arg_name}={arg_val}' for arg_name, arg_val in config['args'].items()))
    print(f'Experiment {exp_name} started')
#     print(f'args:{arguments}')
    res_code = os.system(f'python -m lexsub.augment_conll_sentences {arguments}')

    print(f'Experiment {exp_name} finished')
#     return res_code
    
def expand_experiments(configs, workers, exp_names=None):
    
    if type(configs) is str:
        with open(configs, 'r') as f:
            exp_configs = json.load(f)
    
    print('workers:', workers)
    for worker in range(workers):
        DEVICE_QUEUE.put(worker)

    if type(exp_names) is str:
        exp_names = exp_names.split(',')

    if exp_names is None: exp_names = [exp['name'] for exp in exp_configs]

    experiments = []
    for exp in exp_configs:
           if exp['name'] in exp_names:
                experiments.append(exp)
    
    print('exp_names:', exp_names)
    print('# of experiments:', len(experiments))
    pool = mp.Pool(workers, initializer=initialize_worker)
            
    pool.map(run_exp, experiments)
    
        
if __name__ == '__main__':
    fire.Fire(expand_experiments)
    
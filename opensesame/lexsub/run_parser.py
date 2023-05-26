import json
import fire
import os
import time
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
    mode = MODE

    seed = ''
    if 'fixseed' in config["args"].keys():
        if config["args"]['fixseed']:
            seed = '--fixseed'
    config["args"].pop('fixseed', None)
    model_id = config["args"].pop('model_id', None)
    arguments = ' '.join((f'--{arg_name}={arg_val}' for arg_name, arg_val in config['args'].items()))

    print(f'Experiment {exp_name} started')
    if mode in ['refresh', 'train']:
        print(f'Training ...')
        res_code = os.system(f'python -m sesame.{model_id} --mode {mode} {arguments} {seed}')
    
        if res_code == 0:
            print(f'Testing ...')
            res_code = os.system(f'python -m sesame.{model_id} --mode test {arguments}')
    else:
        print(f'Testing ...')
        res_code = os.system(f'python -m sesame.{model_id} --mode test {arguments}')
            
    print(f'Experiment {exp_name} finished')
    return res_code
    
def main(configs, workers, mode='train', exp_names=None):
    global MODE
    MODE = mode
    
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
    
    print('exp_names:',exp_names)
    pool = mp.Pool(workers, initializer=initialize_worker)
    pool.map(run_exp, experiments)
    
        
if __name__ == '__main__':
    fire.Fire(main)
    
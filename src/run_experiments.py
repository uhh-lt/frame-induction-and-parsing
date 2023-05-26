import json
import fire
import os
import sys
import copy 
import multiprocessing as mp
from collections.abc import Iterable


CUDA_DEVICES = mp.Queue()
WORKER_CUDA_DEVICE = None
def initialize_worker():
    global CUDA_DEVICES
    global WORKER_CUDA_DEVICE
    WORKER_CUDA_DEVICE = CUDA_DEVICES.get()
    print('Worker cuda device:', WORKER_CUDA_DEVICE)
    
    
def run_exp(*args, **kwargs):
    config = args[0]
    exp_name = config['name']
    print(f'Experiment {exp_name} started')
    #run_predict.main(**kwargs)
    arguments = ' '.join((f'--{arg_name}={arg_val}' for arg_name, arg_val in config['args'].items()))
#     print(arguments)
    res_code = os.system(f'CUDA_VISIBLE_DEVICES={WORKER_CUDA_DEVICE} python -m src.run_predict {arguments}')

    print(f'Experiment {exp_name} finished')
    return res_code
    
    
def run_experiments(config, cuda_devices, exp_names=None):
    
    if type(config) is str:
        with open(config, 'r') as f:
            config = json.load(f)
    
    if not isinstance(cuda_devices, Iterable):
        cuda_devices = [cuda_devices]
    print('Cuda devcies:', cuda_devices)
    for cuda_device in cuda_devices:
        CUDA_DEVICES.put(cuda_device)
    
    if exp_names is not None:
        if type(exp_names) is str:
            exp_names = exp_names.split(',')
        config = [e for e in config if e['name'] in exp_names]

    pool = mp.Pool(len(cuda_devices), initializer=initialize_worker)
    pool.map(run_exp, config)
    
#     for exp_num, exp in enumerate(config):   
#         pool.apply_async(run_exp, kwds=exp)
        #print(f'Experiment {exp_num} is finished!')
        #print('*******************************************************')

        
if __name__ == '__main__':
    fire.Fire(run_experiments)
    
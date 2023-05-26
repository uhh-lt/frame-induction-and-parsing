import os
import fire
import json


def run_command(command):
    print(command)
    res_code = os.system(command)
    if res_code != 0:
        raise RuntimeError(res_code)


def main(cuda_device, config, gold_path, proc_funcs, vocabulary_path, n_jobs, exp_names=None):
    if exp_names is None:
        with open(config) as f:
            exp_names = [e['name'] for e in json.load(f)]
    
    if type(exp_names) is str:
        exp_names = exp_names.split(',')
    
    for exp_name in exp_names:
        with open(config) as f:
            config_data = next(e for e in json.load(f) if e['name'] == exp_name)
        
        run_command(f'python -m src.run_experiments --cuda_devices={cuda_device} --config={config} --exp_names={exp_name}')

        result_dir = os.path.dirname(config_data['args']['result_dir'])

        run_command(f'python -m src.run_postprocessing_predictions --results_path={result_dir} --gold_path={gold_path} --proc_funcs={",".join(proc_funcs)} --n_jobs={n_jobs} --vocabulary_path={vocabulary_path} --exp_names={exp_name}')

        run_command(f'python -m src.run_evaluate --results_path={result_dir} --exp_names={exp_name}')
    
    
if __name__ == '__main__':
    fire.Fire(main)

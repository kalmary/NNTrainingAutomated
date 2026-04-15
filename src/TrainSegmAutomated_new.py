import matplotlib
matplotlib.use('Agg')


import pathlib as pth
import numpy as np
from pprint import pprint
from typing import Union, Sequence
import itertools
import sys
import argparse
import logging
import datetime
import multiprocessing


import torch
import torch.nn as nn
from torchinfo import summary
import optuna

from tqdm import tqdm

import gc

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from _train_single_case import train_model
from utils import load_json, save2json, save_model, convert_str_values
from utils import Plotter


def check_models(model,
                 model_configs_paths: list,
                 max_input_size: tuple,
                 max_memory_GB: int,
                 verbose: bool = False) -> tuple[list[dict], list[pth.Path]]:
    
    valid_configs = []
    valid_paths = []

    for path in model_configs_paths:
        path = pth.Path(path)
        instance = None
        try:
            config = load_json(path)
            config = convert_str_values(config)

            # Use the factory to build the model
            instance = model(config)
            instance.eval()

            # Move to CPU for summary if you want to be extra safe, 
            # or keep on device to check VRAM
            model_summary = summary(instance, input_size=max_input_size, verbose=0)
            estimated_memory_GB = (model_summary.total_param_bytes + model_summary.total_output_bytes) / (1024 ** 3)

            if estimated_memory_GB > max_memory_GB:
                raise MemoryError(f"Estimated {estimated_memory_GB:.2f} GB exceeds {max_memory_GB} GB limit.")

            valid_configs.append(config)
            valid_paths.append(path)
            if verbose:
                print(f"{path.name} passed.")

        except Exception as e:
            if verbose:
                print(f"{path.name} failed: {e}")
        
        finally:
            del instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return valid_configs, valid_paths


def get_step_list(params: list[Union[int, float]]) -> list[Union[int, float]]:
    """
    Generate a stepped sequence that includes the stop value if possible.

    Args:
        param_value_list: List of three numeric values [start, stop, step].

    Returns:
        List[Union[int, float]]: A list of numbers from start to stop (inclusive), 
            spaced by step. Returns integers if all inputs are integers, 
            otherwise returns floats.
    """
    start, stop, step = params

    if step == 0:
        raise ValueError("step cannot be zero")

    if all(isinstance(x, int) for x in [start, stop, step]):
        return list(range(int(start), int(stop) + int(step), int(step)))

    elif all(isinstance(x, (int, float)) for x in [start, stop, step]):
        values = np.arange(start, stop + step, step)
        return [float(x) for x in values]

    else:
        raise ValueError("All values must be int or float")

def get_factor_list(params: list[float]) -> list[float]:
    """
    Generates a sorted list of values using a geometric progression 
    that starts at a high value and decays until it hits a lower bound.

    Args:
        params: A list containing [start_value, stop_threshold, decay_factor].
        
    Returns:
        A list of floats sorted in ascending order.
    """
    start, stop, factor = params
    sequence = []
    
    current_val = start
    while current_val > stop:
        sequence.append(current_val)
        current_val *= factor

    # Sorting ensures the 'start' value ends up at the end of the returned list
    sequence.sort()
    return sequence

"""
{
    "learning_rate": [0.01, 0.0001, 0.1],
    "batch_size": [8, 32, 8],
    "num_classes": 10,
    "param_types": {
        "learning_rate": "factor",
        "batch_size": "step",
        "num_clases": "static"
    }
}
"""    

def generate_experiment_configs(training_config: dict, 
                                model_configs_list: Sequence[dict],
                                device_name: str = 'cpu') -> list[dict]:
    logger = logging.getLogger(__name__)
    logger.info('START: generate_experiment_config.')
    
    device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')
    logger.info(f'Using device: {device}')

    dynamic_params = {}
    static_params = {}

    # Separate dynamic and model, base_path=base_path, existing_ok=Truestatic parameters

    for key, value in training_config.items():
        
        if "comment" in key.lower():
            continue
        
        elif isinstance(value, list) and len(value) > 1 and 'samples_len' not in key.lower():
            if "learning_rate" in key.lower() or "weight_decay" in key.lower():
                dynamic_params[key] = get_factor_list(training_config[key])
            else:
                dynamic_params[key] = get_step_list(value)

        elif 'samples_len' in key.lower():
            samples_len = get_step_list(value)
            static_params[key] = samples_len
        else:
            static_params[key] = value

    static_params['device'] = device
    
    logger.info(f'Generated {len(dynamic_params)} dynamic parameters.')
    logger.info(f'Generated {len(static_params)} static parameters.')

    # Generate all combinations of dynamic parameters
    keys = dynamic_params.keys()
    values_lists = dynamic_params.values()
    combinations = list(itertools.product(*values_lists))

    
    # Create experiment configurations
    exp_configs = []
    for combo in combinations:

        for model_config in model_configs_list:
            
            # get model config
            dynamic_config = dict(zip(keys, combo))
            
            exp_config = static_params.copy()
            exp_config.update(dynamic_config)
            exp_config['model_config'] = model_config
            exp_config['model_config']['num_classes'] = exp_config['num_classes']
            
            exp_configs.append(exp_config)
        
    logger.info(f'Generated {len(exp_configs)} experiment configurations.')
    logger.info('STOP: generate_experiment_config')


    return exp_configs



def load_config(model, base_dir: Union[str, pth.Path], device_name: str, mode: int = 0) -> list[dict]:

    """
    Load configuration files and prepare experiment configurations for training.
    mode:
    0 - single_training
    1 - multiple trainings, grid_based
    2 - multiple trainings, with optuna
    
    """
    logger = logging.getLogger(__name__)
    logger.info('START: case_based_training.')

    if isinstance(mode, int):
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid mode: {mode}. Must be:\n" \
                             "0 - test" \
                             "1 - single_training," \
                             "2 - multiple trainings, grid_based," \
                             "3 - multiple trainings, with optuna.")

    training_config = None
    base_dir = pth.Path(base_dir)
    config_files_dir = base_dir.joinpath('training_configs')
    model_configs_dir = base_dir.joinpath('model_configs')

    model_configs_paths_list = list(model_configs_dir.rglob('*.json'))
    logger.info(f'Found {len(model_configs_paths_list)} model configs in dir {model_configs_dir}')


    if mode == 0 or mode == 1:
        training_config = load_json(config_files_dir.joinpath('config_train_single.json'))
        model_configs_paths_list = [p for p in model_configs_paths_list if "single" in p.stem]
    elif mode == 2 or mode == 3:
        training_config = load_json(config_files_dir.joinpath('config_train.json'))
        model_configs_paths_list = [p for p in model_configs_paths_list if "single" not in p.stem]
    
    logger.info(f'Loaded training config for mode: {mode}.')
        
    
    training_config = convert_str_values(training_config)
    model_configs_list, _ = check_models(model, model_configs_paths_list, max_input_size=(8, 2*8192, 4), max_memory_GB=32)
    
    assert model_configs_list != 0, "No models compiled. Check model_configs - most likely too big models are defined"

    if mode == 3:
        device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')
        training_config['device'] = device
        
        logger.info(f'Loaded device: {device}')
        logger.info('STOP: load_config. All files loaded.')

        training_config['model'] = None

        return [training_config, model_configs_list]
    else:
        exp_configs = generate_experiment_configs(training_config, 
                                            model_configs_list, 
                                            device_name = device_name)
        
        logger.info('STOP: load_config. All files loaded.')

        return exp_configs

def test_case(exp_config: dict) -> None:
    
    logger = logging.getLogger(__name__)
    logger.info('START: test_case.')
    
    """Test case for training model with reduced parameters for quick execution."""
    exp_config['train_repeat'] = 2
    exp_config['learning_rate'] = 0.01
    exp_config['epochs'] = 2
    
    try:
        for _, _ in train_model(training_dict=exp_config):
            pass
    except Exception as e:
        logger.error(f'ERROR: test_case. Error message: {e}')
        print(f"Error training model (TESTING_MODE):\n{e}")
    logger.info('STOP: test_case passed.')

class Checkpoint: 
    def __init__(self, existing_ok: bool = False) -> None:
        self.existing_ok = existing_ok
        self.final_val_best = 0.
        self.save_new = False
    
    def check_checkpoint(self, 
                         model: nn.Module,
                         model_name: str,
                         final_val: float,
                         exp_config: dict,
                         result_hist: dict) -> tuple[nn.Module, dict, pth.Path]:
    
        

        logger = logging.getLogger(__name__)

        model_dir =  pth.Path(__file__).parent.joinpath('training_results').joinpath(f'{model_name.rsplit('_', 1)[0]}')
        model_dir.mkdir(exist_ok=True, parents=True)

        model_path = model_dir.joinpath(f'{model_name}.pt')

        plot_dir = model_dir.joinpath('plots')
        if not plot_dir.exists():
            logger.info(f'Plots directory created: {plot_dir}')

        dict_files_dir = model_dir.joinpath('dict_files')
        if not dict_files_dir.exists():
            logger.info(f'Dict files directory created: {dict_files_dir}')
        
        plot_dir.mkdir(exist_ok=True, parents=True)
        if not plot_dir.exists():
            logger.info(f'Plots directory created: {plot_dir}')

        dict_files_dir.mkdir(exist_ok=True, parents=True)

        config_path = dict_files_dir.joinpath(f'{model_path.stem}_config.json')


        if final_val > self.final_val_best:
            self.save_new = True
            self.final_val_best = final_val
        
        if not self.save_new:
            return [model, exp_config, config_path]

        if not self.existing_ok:
                for file_path in plot_dir.iterdir():
                    file_path.unlink()
                for file_path in dict_files_dir.iterdir():
                    file_path.unlink()

        model_path = save_model(model_path, model, 
                                        existing_ok=self.existing_ok)
        model_name = model_path.stem
        logger.info(f'New best model saved to: {model_path}. model_name: {model_name}')


        best_config = exp_config
        best_config['device'] = str(best_config['device'])

        config_path = dict_files_dir.joinpath(f'{model_path.stem}_config.json')
        save2json(best_config, config_path)
        logger.info(f'New config for model {model_name} saved to: {config_path}')



        if len(result_hist['acc_hist']) > 1:
            plotter_obj = Plotter(best_config['num_classes'], plots_dir = plot_dir)


            plotter_obj.plot_metric_hist(f'Loss_{model_name}.png',
                                         result_hist['loss_hist'],
                                         result_hist['loss_v_hist'])

            plotter_obj.plot_metric_hist(f'Accuracy_{model_name}.png', 
                                    result_hist['acc_hist'],
                                    result_hist['acc_v_hist'])
            
            plotter_obj.plot_metric_hist(f'mIoU_{model_name}.png', 
                                    result_hist['miou_hist'],
                                    result_hist['miou_v_hist'])

            logger.info(f'Metrics history plots for {model_name} saved to: {plot_dir}')


        self.save_new = False
        return model, best_config, config_path

        




def case_based_training(exp_configs: list[dict],
                        model_name: str) -> None:
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: case_based_training.')

    pbar = tqdm(enumerate(exp_configs), total=len(exp_configs), desc="Training models", unit="model", position=0, leave=False)

    model_dir =  pth.Path(__file__).parent.joinpath('training_results').joinpath(f'{model_name.rsplit('_', 1)[0]}')
    model_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Model directory created: {model_dir}')

    model_path = model_dir.joinpath(f'{model_name}.pt')
    logger.info(f'Model path created: {model_path}')

    plot_dir = model_dir.joinpath('plots')
    dict_files_dir = model_dir.joinpath('dict_files')
    

    plot_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Plots directory created: {plot_dir}')
    dict_files_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Dict files directory created: {dict_files_dir}')
    

    logger.info('Case based training starting.')
    checkpoint = Checkpoint(existing_ok=False)


    for i, exp_config in pbar:
        logger.info(f'Case {i+1}/{len(exp_configs)}: {exp_config}')

        for model, result_hist in train_model(training_dict=exp_config):
            logger.info(f'Single model was generated. val_acc: {result_hist["acc_v_hist"][-1]:.3f}  val_loss: {result_hist["loss_v_hist"][-1]:.3f}')

            final_val = result_hist['acc_v_hist'][-1]*0.6 + (1 / (1 + result_hist['loss_v_hist'][-1]))*0.4

            model, best_config, config_path, result_hist = checkpoint.check_checkpoint(model, model_name, final_val, exp_config, result_hist)
    
    logger.info(f'Best model saved to: {model_path}')
    logger.info(f'Best config saved to: {config_path}')
    logger.info('STOP: case_based_training')

    summary(model)


def objective_function(trial: optuna.Trial,
                       exp_config: list[dict], # exp config, converted from str
                       model_name: str,
                       model_configs_list: list[dict],
                       checkpoint: object) -> float:
    
    """
    Objective function for Optuna
    """
    
    logger = logging.getLogger(__name__)
    logger.info('START: objective_function')

    existing_ok = False


    model_config_index = trial.suggest_categorical('model_config_index', range(len(model_configs_list)))
    model_config = model_configs_list[model_config_index]

    batch_size = get_step_list(exp_config['batch_size'])
    epochs = get_step_list(exp_config['epochs'])

    lr = trial.suggest_categorical('learning_rate', get_factor_list(exp_config['learning_rate']))
    weight_decay = trial.suggest_categorical('weight_decay', get_factor_list(exp_config['weight_decay']))

    batch_size = trial.suggest_int('batch_size', exp_config['batch_size'][0], exp_config['batch_size'][1], step=exp_config['batch_size'][2])
    epochs = trial.suggest_int('epochs', exp_config['epochs'][0], exp_config['epochs'][1], step=exp_config['epochs'][2] )
    focal_loss_gamma = trial.suggest_float('focal_loss_gamma', exp_config['focal_loss_gamma'][0], exp_config['focal_loss_gamma'][1], step=exp_config['focal_loss_gamma'][2])

    pc_start = trial.suggest_float('pc_start', exp_config['pc_start'][0], exp_config['pc_start'][1], step=exp_config['pc_start'][2])
    div_factor = trial.suggest_int('div_factor', exp_config['div_factor'][0], exp_config['div_factor'][1], log=True)
    fin_div_factor = trial.suggest_int('final_div_factor', exp_config['final_div_factor'][0], exp_config['final_div_factor'][1], log=True)
    num_neighbors = trial.suggest_int('num_neighbors', exp_config['num_neighbors'][0], exp_config['num_neighbors'][1], step = exp_config['num_neighbors'][2])
    num_points = trial.suggest_int('num_points', exp_config['num_points'][0], exp_config['num_points'][1], step = exp_config['num_points'][2])

    model_config.update({
        'num_neighbors': num_neighbors
    })

    model_config['num_classes'] = exp_config['num_classes']

    exp_config = exp_config.copy()
    exp_config.update({
        'model_config': model_config,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'epochs': epochs,
        'num_neighbors': num_neighbors,
        'num_points': num_points,
        'focal_loss_gamma': focal_loss_gamma,
        'pc_start': pc_start,
        'div_factor': div_factor,
        'final_div_factor': fin_div_factor,
        'train_repeat':1
    })

    logger.info(f'Generated exp_config for trial: {trial.number}')
    for key, value in exp_config.items():
        if key != 'model_config':
            logger.info(f'parameters: {key}: {value}')
    for key, value in model_config.items():
        logger.info(f'model_config: {key}: {value}')

    
    # start training
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    best_val_miou = 0.0

    for epoch_idx, (model, result_hist) in enumerate(train_model(training_dict=exp_config)):
        
        if model is None or result_hist == {}:
            logger.error('ERROR: objective_function. Empty result_hist. Trial pruned.')
            trial.report(0.0, step=epoch_idx)
            raise optuna.exceptions.TrialPruned()
            

        # goal is to maximize val_acc + 1/val_loss
        final_val_accuracy = result_hist['acc_v_hist'][-1]
        final_val_loss = result_hist['loss_v_hist'][-1]
        final_val_miou = result_hist['miou_v_hist'][-1]

        best_val_accuracy = max(best_val_accuracy, final_val_accuracy)
        best_val_miou = max(best_val_miou, final_val_miou)
        best_val_loss = min(best_val_loss, final_val_loss)


        normalized_loss = 1 / (1 + best_val_loss)
        final_val = 0.4 * best_val_accuracy + 0.2 * normalized_loss + 0.4 * best_val_miou # TODO tuning might be necessary

        checkpoint.check_checkpoint(model,
                                    model_name,
                                    final_val,
                                    exp_config,
                                    result_hist)

        logger.info(f'Epoch {epoch_idx+1}/{exp_config["epochs"]}: best_val_acc: {best_val_accuracy:.3f}, best_val_loss: {best_val_loss:.3f}, best_val_miou: {best_val_miou:.3f}, final_val: {final_val:.3f}')

        # report acutal results for prunning
        trial.report(final_val, step=epoch_idx)

        if trial.should_prune():
            logger.info(f'Pruning trial: {trial.number}')
            raise optuna.exceptions.TrialPruned()
        
    logger.info('STOP: objective_function')

    return final_val

def optuna_based_training(exp_config: list[dict], # only one, non converted conf given in list
                          model_name: str,
                          n_trials: int = 100) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logger = logging.getLogger(__name__)
    logger.info('START: optuna_based_training.')

    # able to automatically stop poor working exps
    n_startup = 3
    n_warmup_steps = 15
    interval_steps = 5

    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup_steps, interval_steps=interval_steps)
    logger.info(f'Pruner created: parameters: n_startup_trials: {n_startup}, n_warmup_step: {n_warmup_steps}, interval_steps: {interval_steps}')

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        study_name = f'{model_name}_{timestamp}',
        storage='sqlite:///db.sqlite3',
        directions=['maximize'],
        pruner=pruner)
    logger.info('Study created. Check ')


    model_configs = exp_config[1]
    exp_config = exp_config[0]

    train_repeat_old = exp_config['train_repeat']
    
    # Create progress bar
    pbar = tqdm(total=n_trials, desc="Optuna Optimization", unit="trial")
    
    def callback(study, trial):
        pbar.update(1)
        try:
            pbar.set_postfix({
                "Trial": trial.number,
                "Best Value": f"{study.best_value:.4f}",
                "Current": f"{trial.value:.4f}" if trial.value else "Pruned"
            })
        except ValueError:
            pbar.set_postfix({"Status": "Pruned"})

    checkpoint = Checkpoint(existing_ok=False)                          

    # Single optimize call with callback
    study.optimize(lambda trial: objective_function(trial,
                                                    exp_config=exp_config,
                                                    model_name = model_name,
                                                    model_configs_list=model_configs,
                                                    checkpoint=checkpoint),
                   n_trials=n_trials,
                   callbacks=[callback])
    
    pbar.close()
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    logger.info(f'Optimization finished. Best value of formula: 0.6 * val_acc + 0.4 * norm_val_loss: {best_value:.4f}')

    print(20*'=')
    print('Optuna optimalization finished')
    print(f'Best value of 0.6 * val_acc + 0.4 * norm_val_loss: {best_value:.4f}')
    pprint(f'Best trial params:\n{best_params}')
    print(20*'=')


    best_model_config = model_configs[best_params.pop('model_config_index')] 

    final_exp_config = exp_config.copy()
    final_exp_config.update(best_params)
    final_exp_config.update({'train_repeat': train_repeat_old})


    final_exp_config['model_config'] = best_model_config

    print('Training the best model last time: ')

    case_based_training([final_exp_config],
                        model_name=model_name) # FIXME inproper dict creation
    
    logger.info('STOP: optuna_based_training')
    
def argparser():
        
    """
    Parse command-line arguments for automated CNN training pipeline configuration.
    Accepts model naming, computational device selection (CPU/CUDA/GPU), and optional test mode activation.
    Returns parsed arguments with validation for device choices and formatted help text display.
    """

    default_name = 'ResNet_0'
    parser = argparse.ArgumentParser(
        description="Script for training the model based on predefined range of scenarios",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default=default_name,
        help=(
            "Base of the model's name.\n"
            "When iterating, name also gets an ID. \n"
            f"If not given, defaults to: {default_name}."
        )
    )

    
    # Flag definition
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'], # choice limit
        help=(
            "Device for tensor based computation.\n"
            "Pick 'cpu' or 'cuda'/ 'gpu'.\n"
        )
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4], # choice limit
        help=(
            "Device for tensor based computation.\n"
            'Pick:\n'
            '0: test\n'
            '1: single training\n'
            '2: multiple trainings, grid_based\n'
            '3: multiple trainings, with optuna\n'
            '4: only check models'
        )
    )

    return parser.parse_args()


def main():
    model = None
    
    multiprocessing.set_start_method('spawn', force=True)
    args = argparser()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = pth.Path(__file__).parent.joinpath('logs')
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file_name = f'{args.model_name}_{timestamp}_training.log'
    log_file_path = log_dir.joinpath(log_file_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"PROGRAM START: {args.model_name}")


    device = args.device.lower()
    model_name = args.model_name

    base_path = pth.Path(__file__).parent
    if args.mode == 4:
        model_configs_dir = base_path.joinpath('model_configs')
        model_configs_paths_list = list(model_configs_dir.rglob('*.json'))

        check_models(model, model_configs_paths=model_configs_paths_list, 
                     max_input_size=(1, 8192, 4), 
                     max_memory_GB=20,
                     verbose=True)
    
    exp_configs  = load_config(model, base_path, device, mode = args.mode)

    if args.mode == 0:
        test_case(exp_config=exp_configs[0])
    elif args.mode == 1 or args.mode==2:
        case_based_training(exp_configs=exp_configs,
                            model_name=model_name)
    elif args.mode == 3:
        optuna_based_training(exp_config=exp_configs,
                              model_name=model_name,
                              n_trials=80)
        

if __name__ == '__main__':
    
    main()  
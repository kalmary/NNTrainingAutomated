import pathlib as pth
import torch
import torch.nn as nn
from torchinfo import summary
import optuna
import gc
import logging
from typing import Literal, Union
from _train_single_case import train_model
from tqdm import tqdm
import datetime
import argparse

from utils import load_json, save2json, save_model, convert_str_values
from utils import Plotter


class Checkpoint:
    def __init__(self,
                 model_name: str,
                 base_dir: pth.Path,
                 existing_ok: bool = False) -> None:
        self.model_name = model_name
        self.existing_ok = existing_ok
        self.final_val_best = 0.0

        # Resolve all paths once
        self.model_dir = base_dir / 'training_results' / model_name.rsplit('_', 1)[0]
        self.model_path = self.model_dir / f'{model_name}.pt'
        self.plot_dir = self.model_dir / 'plots'
        self.dict_files_dir = self.model_dir / 'dict_files'
        self.config_path = self.dict_files_dir / f'{model_name}_config.json'

        for d in (self.model_dir, self.plot_dir, self.dict_files_dir):
            d.mkdir(exist_ok=True, parents=True)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Checkpoint initialized for {model_name} at {self.model_dir}')

    def check_checkpoint(self,
                         model: nn.Module,
                         final_val: float,
                         exp_config: dict,
                         result_hist: dict) -> None:
        if final_val <= self.final_val_best:
            return

        self.final_val_best = final_val

        if not self.existing_ok:
            for f in self.plot_dir.iterdir(): f.unlink()
            for f in self.dict_files_dir.iterdir(): f.unlink()

        save_model(self.model_path, model, existing_ok=self.existing_ok)
        self.logger.info(f'New best model saved: {self.model_path} (score={final_val:.3f})')

        cfg = {**exp_config, 'device': str(exp_config['device'])}
        save2json(cfg, self.config_path)

        if len(result_hist['acc_hist']) > 1:
            plotter = Plotter(cfg['num_classes'], plots_dir=self.plot_dir)
            plotter.plot_metric_hist(f'Loss_{self.model_name}.png',
                                     result_hist['loss_hist'], result_hist['loss_v_hist'])
            plotter.plot_metric_hist(f'Accuracy_{self.model_name}.png',
                                     result_hist['acc_hist'], result_hist['acc_v_hist'])
            plotter.plot_metric_hist(f'mIoU_{self.model_name}.png',
                                     result_hist['miou_hist'], result_hist['miou_v_hist'])
            self.logger.info(f'Plots updated in {self.plot_dir}')

def train_single_score(result_hist:dict) -> float:
    acc = result_hist['acc_v_hist'][-1]
    loss = result_hist['loss_v_hist'][-1]
    return 0.6 * acc + 0.4 / (1 + loss)

class TrialScoreTracker:
    """Tracks best-so-far metrics across epochs and computes a trial score."""

    def __init__(self, w_acc = 0.4, w_loss = 0.2, w_miou = 0.4):
        self.w_acc = w_acc
        self.w_loss = w_loss
        self.w_miou = w_miou
        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.best_miou = 0.0

    def update(self, result_hist: dict) -> float:
        """Update running bests from latest epoch, return current score."""
        self.best_acc = max(self.best_acc, result_hist['acc_v_hist'][-1])
        self.best_miou = max(self.best_miou, result_hist['miou_v_hist'][-1])
        self.best_loss = min(self.best_loss, result_hist['loss_v_hist'][-1])
        return self._score()

    def _score(self) -> float:
        normalized_loss = 1 / (1 + self.best_loss)
        return self.w_acc * self.best_acc + self.w_loss * normalized_loss + self.w_miou * self.best_miou

    @property
    def formula(self) -> str:
        return (f'{self.w_acc} * val_acc '
                f'+ {self.w_loss} * 1/(1+val_loss) '
                f'+ {self.w_miou} * val_miou')

class TrainAutomated():
    def __init__(self, 
                 model_cls: type, 
                 device: Literal['cuda', 'gpu', 'cpu'], 
                 max_memory_GB: int, 
                 max_input_size:tuple, 
                 base_dir: Union[str, pth.Path] = pth.Path(__file__).parent, 
                 n_trials: int = 100, 
                 n_startup: int = 3, 
                 n_warmup_steps: int = 15, 
                 interval_steps: int = 15) -> None:
        self.model_cls = model_cls
        self.device = torch.device('cuda') if (('cuda' in device or 'gpu' in device) and torch.cuda.is_available()) else torch.device('cpu')
        self.max_memory_GB = max_memory_GB
        self.max_input_size = max_input_size
        self.base_dir = pth.Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Train Automated: Using device: {self.device} (requested: '{device}').")
        self.n_trials = n_trials
        self.n_startup = n_startup
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
    
    def check_models(self, model_configs_paths: list) -> tuple[list[dict], list[pth.Path]]:

        valid_configs = []
        valid_paths = []

        for path in model_configs_paths:
            path = pth.Path(path)
            instance = None
            try:
                config = load_json(path)
                config = convert_str_values(config)

                instance = self.model_cls(config)
                instance.eval()

                model_summary = summary(instance, input_size=self.max_input_size, verbose=0)
                estimated_memory_GB = (model_summary.total_param_bytes + model_summary.total_output_bytes) / (1024 ** 3)

                if estimated_memory_GB > self.max_memory_GB:
                    raise MemoryError(f"Estimated {estimated_memory_GB:.2f} GB exceeds {self.max_memory_GB} GB limit.")

                valid_configs.append(config)
                valid_paths.append(path)
                self.logger.info(f"{path.name} passed — {estimated_memory_GB:.2f} GB.")

            except Exception as e:
                self.logger.warning(f"{path.name} failed: {e}")

            finally:
                del instance
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.info(f"check_models complete: {len(valid_configs)}/{len(model_configs_paths)} configs valid.")
        return valid_configs, valid_paths



    def load_config(self, mode: int = 0):

        """
        Load configuration files and prepare experiment configurations for training.
        mode:
        0 - test
        1 - single training
        2 - multiple trainings, with optuna
        
        """
        self.logger.info('START: case_based_training.')

        if mode not in [0, 1, 2]:
            raise ValueError(f"Invalid mode: {mode}. Must be:\n" \
                            "0 - test," \
                            "1 - single_training," \
                            "2 - multiple trainings, with optuna," \
                            "3 - Check models.")

        config_files_dir = self.base_dir.joinpath('training_configs')
        model_configs_dir = self.base_dir.joinpath('model_configs')

        model_configs_paths_list = list(model_configs_dir.rglob('*.json'))
        self.logger.info(f'Found {len(model_configs_paths_list)} model configs in dir {model_configs_dir}')


        if mode == 0 or mode == 1:
            training_config = load_json(config_files_dir.joinpath('config_train_single.json'))
            model_configs_paths_list = [p for p in model_configs_paths_list if "single" in p.stem]

        elif mode == 2:
            training_config = load_json(config_files_dir.joinpath('config_train.json'))
            model_configs_paths_list = [p for p in model_configs_paths_list if "single" not in p.stem]
        
        self.logger.info(f'Loaded training config for mode: {mode}.')
        
        training_config = convert_str_values(training_config)
        model_configs_list, _ = self.check_models(model_configs_paths_list)
        
        assert model_configs_list, "No models compiled. Check model_configs - most likely too big models are defined"

        if mode == 2:
            training_config['device'] = self.device
            
            self.logger.info(f'Loaded device: {self.device}')
            self.logger.info('STOP: load_config. All files loaded.')

            training_config['model'] = None

            return [training_config, model_configs_list]
        
        else:
            self.logger.info('STOP: load_config. All files loaded.')

            training_config['model'] = None
            training_config['model_config'] = model_configs_list[0]
            training_config['device'] = self.device

            exp_configs = [training_config]

            return exp_configs

    def test_case(self, exp_config: dict) -> None:
    
        self.logger = logging.getLogger(__name__)
        self.logger.info('START: test_case.')
        
        """Test case for training model with reduced parameters for quick execution."""
        exp_config['train_repeat'] = 2
        exp_config['learning_rate'] = 0.01
        exp_config['epochs'] = 2
        
        try:
            for _, _ in train_model(training_dict=exp_config):
                pass
        except Exception as e:
            self.logger.error(f'ERROR: test_case. Error message: {e}')
            print(f"Error training model (TESTING_MODE):\n{e}")
        self.logger.info('STOP: test_case passed.')


    def train_single(self, exp_config: dict, model_name: str) -> None:
        
        self.logger.info(f'START: train_single ({model_name}).')
        checkpoint = Checkpoint(model_name=model_name,
                                base_dir=self.base_dir,
                                existing_ok=False)

        pbar = tqdm(train_model(training_dict=exp_config),
                    total=exp_config.get('train_repeat', 1),
                    desc=f"Training {model_name}", unit="repeat")

        for model, result_hist in pbar:
            score = train_single_score(result_hist)
            self.logger.info(f'Repeat done — val_acc={result_hist['acc_v_hist'][-1]:.3f} val_loss={result_hist["loss_v_hist"][-1]:.3f} score={score:.3f}')
            checkpoint.check_checkpoint(model, score, exp_config, result_hist)

        self.logger.info(f'STOP: train_single. Best score: {checkpoint.final_val_best:.3f}')

    def _suggest_from_spec(self, trial, name, spec):

        if not isinstance(spec, list):
            return spec
        
        if all(isinstance(x, str) for x in spec):
            return trial.suggest_categorical(name, spec)

        if all(isinstance(x, bool) for x in spec):
            return trial.suggest_categorical(name, spec)
        
        is_int = all(isinstance(x, int) for x in spec)
        is_num = all(isinstance(x, (int, float)) for x in spec)

        if not is_num:
            raise ValueError(f"Spec '{name}' has mixed/unsupported types: {spec}")
        
        suggest = trial.suggest_int if is_int else trial.suggest_float
        low, high = spec[0], spec[1]

        if len(spec) == 3:
            return suggest(name, low, high, step=spec[2])
        if len(spec) == 2:
            return suggest(name, low, high, log = True)
        
        raise ValueError(f"Spec '{name}' must have 2 or 3 elements, got {len(spec)}: {spec}")


    def objective_function(self, trial, exp_config, model_configs_list, checkpoint):
        
        model_config_index = trial.suggest_categorical('model_config_index', list(range(len(model_configs_list))))
        model_config = model_configs_list[model_config_index]

        trial_config = {
            key: self._suggest_from_spec(trial, key, value)
            for key, value in exp_config.items()
        }

        # these parameters should be already writen in the configs
        trial_config['train_repeat'] = 1
        model_config['num_neighbors'] = trial_config['num_neighbors']
        model_config['num_classes'] = trial_config['num_classes']


        trial_config["model_config"] = model_config
        self.logger.info(f'Generated exp_config for trial: {trial.number}')

        self.logger.info(f'Trial {trial.number} config:')
        for key, value in trial_config.items():
            if key != 'model_config':
                self.logger.info(f' {key}: {value}')
        for key, value in model_config.items():
            self.logger.info(f' model_config.{key}: {value}')

        tracker = TrialScoreTracker()
        score = 0.0

        for epoch_idx, (model, result_hist) in enumerate(train_model(training_dict=trial_config)):

            if model is None or not result_hist:
                self.logger.error(f'Trial {trial.number}: empty result_hist at epoch {epoch_idx}, pruning.')
                trial.report(0.0, step=epoch_idx)
                raise optuna.exceptions.TrialPruned()

            score = tracker.update(result_hist)
            checkpoint.check_checkpoint(model, score, trial_config, result_hist)

            self.logger.info(
                f'Trial {trial.number} epoch {epoch_idx+1}/{trial_config["epochs"]}: '
                f'score={score:.3f} {vars(tracker)}'
            )

            trial.report(score, step=epoch_idx)
            if trial.should_prune():
                self.logger.info(f'Pruning trial {trial.number} at epoch {epoch_idx+1}.')
                raise optuna.exceptions.TrialPruned()

        self.logger.info(f'STOP: objective_function (trial {trial.number}, final score {score:.3f})')
        return score

    def optuna_based_training(self, exp_config, model_name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger.info('START: optuna_based_training.')

        pruner = optuna.pruners.MedianPruner(n_startup_trials=self.n_startup, n_warmup_steps=self.n_warmup_steps, interval_steps=self.interval_steps)
        self.logger.info(f'Pruner created: parameters: n_startup_trials: {self.n_startup}, n_warmup_step: {self.n_warmup_steps}, interval_steps: {self.interval_steps}')

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            study_name = f'{model_name}_{timestamp}',
            storage='sqlite:///db.sqlite3',
            directions=['maximize'],
            pruner=pruner)
        self.logger.info('Study created. Check ')

        model_configs = exp_config[1]
        exp_config = exp_config[0]

        train_repeat_old = exp_config['train_repeat']

        # Create progress bar
        pbar = tqdm(total=self.n_trials, desc="Optuna Optimization", unit="trial")

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

        checkpoint = Checkpoint(model_name=model_name, 
                                base_dir=self.base_dir, 
                                existing_ok=False)  

        # Single optimize call with callback
        study.optimize(lambda trial: self.objective_function(trial,
                                                        exp_config=exp_config,
                                                        model_configs_list=model_configs,
                                                        checkpoint=checkpoint),
                    n_trials=self.n_trials,
                    callbacks=[callback])

        pbar.close()
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        tracker = TrialScoreTracker()
        self.logger.info(f'Optimization finished. Best value of formula: {tracker.formula} = {best_value:.4f}')

        best_model_config = model_configs[best_params.pop('model_config_index')] 

        final_exp_config = exp_config.copy()
        final_exp_config.update(best_params)
        final_exp_config.update({'train_repeat': train_repeat_old})


        final_exp_config['model_config'] = best_model_config

        print('Training the best model last time: ')

        self.train_single(final_exp_config,
                            model_name=model_name) # FIXME inproper dict creation
        
        self.logger.info('STOP: optuna_based_training')


    def argparser():
        
        """
        Parse command-line arguments for automated CNN training pipeline configuration.
        Accepts model naming, computational device selection (CPU/CUDA/GPU), and optional test mode activation.
        Returns parsed arguments with validation for device choices and formatted help text display.
        """

        parser = argparse.ArgumentParser(
            description="Script for training the model based on predefined range of scenarios",
            formatter_class=argparse.RawTextHelpFormatter
        )

        parser.add_argument(
            '--model_name',
            type=str,
            help=(
                "Base of the model's name.\n"
                "When iterating, name also gets an ID."
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
            choices=[0, 1, 2, 3], # choice limit
            help=(
                "Device for tensor based computation.\n"
                'Pick:\n'
                '0: test\n'
                '1: single training\n'
                '2: multiple trainings, with optuna\n'
                '3: only check models'
            )
        )

        return parser.parse_args()

    def run(self, model_name:str, mode:Literal[0, 1, 2], device):
        pass
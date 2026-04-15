import pathlib as pth
import torch
import torch.nn as nn
from torchinfo import summary
import optuna
import gc
import logging
from typing import Literal, Union
from _train_single_case import train_model
import tqdm

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



class TrainAutomated():
    def __init__(self, model_cls: type, device: Literal['cuda', 'gpu', 'cpu'], max_memory_GB: int, max_input_size:tuple, base_dir: Union[str, pth.Path] = pth.Path(__file__).parent) -> None:
        self.model_cls = model_cls
        self.device = torch.device('cuda') if (('cuda' in device or 'gpu' in device) and torch.cuda.is_available()) else torch.device('cpu')
        self.max_memory_GB = max_memory_GB
        self.max_input_size = max_input_size
        self.base_dir = pth.Path(base_dir)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Train Automated: Using device: {self.device} (requested: '{device}').")
    
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

        if isinstance(mode, int):
            if mode not in [0, 1, 2]:
                raise ValueError(f"Invalid mode: {mode}. Must be:\n" \
                                "0 - test" \
                                "1 - single_training," \
                                "2 - multiple trainings, with optuna.")

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
            val_acc = result_hist['acc_v_hist'][-1]
            val_loss = result_hist['loss_v_hist'][-1]
            score = 0.6 * val_acc + 0.4 * (1 / (1 + val_loss))
            self.logger.info(f'Repeat done — val_acc={val_acc:.3f} val_loss={val_loss:.3f} score={score:.3f}')
            checkpoint.check_checkpoint(model, score, exp_config, result_hist)

        self.logger.info(f'STOP: train_single. Best score: {checkpoint.final_val_best:.3f}')



    def run(self, model_name:str, mode:Literal[0, 1, 2], device):
        pass
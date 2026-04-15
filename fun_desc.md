**`check_models`**
Validates a list of model config files by actually building each model and estimating its memory footprint. Filters out configs that either fail to compile or exceed the memory limit. Returns only the configs and paths that passed.

---

**`get_step_list`**
Converts a `[start, stop, step]` spec into a full list of values. Works for both int and float. Used to expand range-style config values into concrete lists for grid search.

---

**`get_factor_list`**
Similar to `get_step_list` but multiplies instead of adds. Converts a `[start, stop, factor]` spec into a list that shrinks multiplicatively — used for learning rate and weight decay which are better searched on a log scale.

---

**`generate_experiment_configs`**
Takes a training config and a list of model configs and produces every combination of parameters as individual experiment dictionaries. Separates params into dynamic (to be combined) and static (same in every experiment), then runs a cartesian product over the dynamic ones.

---

**`load_config`**
Entry point for loading everything from disk. Reads the training config JSON and model config JSONs, validates the models, then either returns raw configs (for Optuna mode) or calls `generate_experiment_configs` to produce ready-to-run experiment dicts.

---

**`test_case`**
Runs a single experiment with reduced settings (2 epochs, fixed learning rate) just to verify the pipeline doesn't crash. A smoke test.

---

**`Checkpoint`**
A stateful class that tracks the best model seen so far across a training run. On each call to `check_checkpoint` it decides whether to save — only saving when the new score beats the previous best. Also handles saving plots, configs, and model weights to disk.

---

**`case_based_training`**
Iterates through a list of experiment configs, trains each one, and uses `Checkpoint` to keep only the best model across all runs. The outer loop is the grid search execution.

---

**`objective_function`**
The function Optuna calls on each trial. Samples hyperparameters from the search space defined in the config, runs training, reports intermediate results back to Optuna for pruning decisions, and returns a final score.

---

**`optuna_based_training`**
Sets up and runs the Optuna study — creates the pruner, sampler, and study object, then calls `study.optimize` which repeatedly calls `objective_function`. After optimization finishes, takes the best found params and does one final training run with `case_based_training`.

---

**`argparser`**
Defines and parses command-line arguments: model name, device (cpu/cuda), and mode (0-4).

---

**`main`**
Entry point. Sets up logging, parses args, loads configs, then routes to the right training function based on mode.
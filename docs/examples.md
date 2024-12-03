# Examples and Tutorials
## Table of Contents
- [Example 1: Running a Basic Experiment](#example-1-running-a-basic-experiment)
- [Example 2: Custom Data Loader](#example-2-custom-data-loader)
- [Example 3: Custom Model](#example-3-custom-model)

## Example 1: Running a Basic Experiment

This example demonstrates how to run a basic experiment using MooClassification.

### Step 1: Set Up the Environment

First, ensure you have installed all dependencies as described in the [installation guide](installation.md).

### Step 2: Define the Configuration
Select a model and environment configuration file from the `configs/` directory.
In this example we use a LightGBM model and an image classification environment.
Adjust a configuration file `lgbm_config.yaml` with the following content:
```yaml
model_type: lgbm
torch_interface: false
params:
  n_jobs: -1
  verbose: -1
  early_stopping_round:
  max_depth: 5

```
Adjust a configuration file `image_classification.yaml` with the following content:
```yaml
name: classification
data_loader_config:
  training_data_path: /path/to/your/data/train
  eval_data_path: /path/to/your/data/test
```
Adjust a configuration file `config.yaml` with the following content:

```yaml
name: "basic_experiment"
model_config_file_name: lgbm_config.yml
env_config_file_name: image_classification.yml
```


### Step 3: Run the Experiment
Execute the run_training.py script with the configuration file:
    
```bash
poetry run python scripts/run_training.py
```

## Example 2: Custom Data Loader

This example shows how to create and use a custom data loader for image classification.
### Step 1: Implement the Data Loader
Create a new file custom_data_loader.py in the environments/data_loaders/ directory:

```python
from optilearn.environments.data_loaders.base_data_loader import BaseDataLoader


class CustomDataLoader(BaseDataLoader):
    def load_data(self):
        # Implement data loading logic here
        pass
```
### Step 2: Update the Configuration
Modify the objects_maps.py to use the custom data loader:

```python
class DataLoaders(BaseObject):
    """
    A class to hold different types of data loaders.
    """
    abstract_base_type: ClassVar = AbstractDataLoader

    simple: Type[AbstractDataLoader] = SimpleImageLoader
    contamination: Type[AbstractDataLoader] = LearningImageLoader
    tabular: Type[AbstractDataLoader] = TabularDataLoader
    
    custom_loader: Type[AbstractDataLoader] = CustomDataLoader
```

### Step 3: Adjust the Experiment Configuration
create a copy of the `image_classification.yaml` configuration file and name it `custom_image_classification.yaml`.
Update the configuration file `custom_image_classification.yaml` to use the custom data loader:
```yaml
name: classification
training_data_loader: custom_loader
eval_data_loader: custom_loader
```
Adjust a configuration file `config.yaml` with the following content:

```yaml
name: "custom_data_loader_experiment"
env_config_file_name: custom_image_classification.yml
```
### Step 4: Run the Experiment
Execute the run_training.py script with the updated configuration file:
```bash
poetry run python scripts/run_training.py
```

## Example 3: Custom Model
This example demonstrates how to create and use a custom model.  
### Step 1: Implement the Model
Create a new file custom_model.py in the models/ directory:

```python
from optilearn.models.base_model import BaseModel


class CustomModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Define model architecture here

    def forward(self, x):
        # Implement forward pass here
        pass
```
### Step 2: Update the Configuration
Modify the objects_maps.py to use the custom model:
```python
class Models(BaseObject):
    """
    A class to hold different types of models.
    """
    abstract_base_type: ClassVar = AbstractModel

    nn: Type[AbstractModel] = ClassificationNN
    sk: Type[AbstractModel] = ClassificationSK
    lgbm: Type[AbstractModel] = ClassificationLGBM
    
    custom_model: Type[AbstractModel] = CustomModel
```

### Step 3: Adjust the Experiment Configuration
create a configuration file `custom_model_config.yaml` with the following content:
```yaml
model_type: custom_model
torch_interface: false # Set to true if using a PyTorch model
params:
    # Add custom model parameters here
```
Adjust a configuration file `config.yaml` with the following content:

```yaml
name: "custom_model_experiment"
model_config_file_name: custom_model_config.yml
```
### Step 4: Run the Experiment
Execute the run_training.py script with the updated configuration file:
```bash
poetry run python scripts/run_training.py
```
